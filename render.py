from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SPHEREx Infrared Explorer", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
# IRSA's SIA v2 servlet. /sia2query/SPHEREx is not a valid IRSA route.
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA2/servlet/Query"
SPHEREX_COLLECTIONS = ("spherex_qr3", "spherex_dr1")


def _radec_from_horizons(target: str, day: date) -> tuple[float, float]:
    """Resolve a Horizons object name/ID to apparent geocentric RA/Dec."""
    start = datetime.combine(day, datetime.min.time())
    stop = start + timedelta(minutes=2)
    params = {
        "format": "json", "COMMAND": f"'{target}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'500@399'",
        "QUANTITIES": "'1'", "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{start:%Y-%m-%d %H:%M}'", "STOP_TIME": f"'{stop:%Y-%m-%d %H:%M}'",
        "STEP_SIZE": "'1 m'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(3, 12))
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise ValueError(payload.get("error", "Horizons returned no position for this target/date."))

    # Horizons quantity 1 is RA/Dec. Match the sexagesimal pair instead of
    # guessing CSV offsets that can vary between targets and API versions.
    section = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    pattern = re.compile(
        r"(?<!\d)(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s*,?\s+"
        r"([+-]?\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)"
    )
    for match in pattern.finditer(section):
        h, m, s = map(float, match.group(1, 2, 3))
        sign = -1 if match.group(4).startswith("-") else 1
        d = abs(float(match.group(4)))
        dm, ds = float(match.group(5)), float(match.group(6))
        ra = 15 * (h + m / 60 + s / 3600)
        dec = sign * (d + dm / 60 + ds / 3600)
        if 0 <= ra < 360 and -90 <= dec <= 90:
            return ra, dec
    raise ValueError("Could not parse RA/Dec from Horizons. Try a planet name or Horizons ID.")


def _parse_votable(content: bytes) -> list[dict]:
    root = ET.fromstring(content)
    fields = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "FIELD":
            fields.append(node.attrib.get("name") or node.attrib.get("ID") or f"column_{len(fields)}")
    results = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "TR":
            continue
        values = [child.text or "" for child in node if child.tag.rsplit("}", 1)[-1] == "TD"]
        if values:
            results.append({fields[i] if i < len(fields) else f"column_{i}": value
                            for i, value in enumerate(values)})
    return results


def _value(record: dict, *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return str(value).strip()
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def query_spherex(ra: float, dec: float, radius: float) -> list[dict]:
    """Search actual SPHEREx products via IRSA SIA v2, trying known releases."""
    found = []
    seen = set()
    errors = []
    for collection in SPHEREX_COLLECTIONS:
        try:
            response = requests.get(
                IRSA_SIA_URL,
                params={
                    "POS": f"CIRCLE {ra:.8f} {dec:.8f} {radius}",
                    "BAND": "0.75e-6 5.0e-6",
                    "COLLECTION": collection,
                },
                headers={"Accept": "application/x-votable+xml"},
                timeout=(4, 15),
            )
            response.raise_for_status()
            if not response.content.strip():
                continue
            for row in _parse_votable(response.content):
                row["_queried_collection"] = collection
                key = _value(row, "obs_publisher_did", "obs_id", "access_url", "access_reference")
                if key not in seen:
                    seen.add(key)
                    found.append(row)
        except requests.RequestException as exc:
            errors.append(f"{collection}: {exc}")
    if not found and errors:
        raise RuntimeError("IRSA SIA query failed: " + " | ".join(errors))
    return found


@st.cache_data(ttl=1800, show_spinner=False)
def download_fits_product(url: str) -> bytes:
    """Download one selected archive product with a strict size cap and timeout."""
    if not url.startswith("https://"):
        raise ValueError("IRSA returned a non-HTTPS product link; refusing to download it.")
    with requests.get(url, stream=True, timeout=(4, 20)) as response:
        response.raise_for_status()
        length = int(response.headers.get("Content-Length", "0") or 0)
        max_size = 80 * 1024 * 1024
        if length > max_size:
            raise ValueError("Product is larger than 80 MB. Use the IRSA product link to download it directly.")
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                total += len(chunk)
                if total > max_size:
                    raise ValueError("Product exceeded the 80 MB download limit.")
                chunks.append(chunk)
        return b"".join(chunks)


def _extract_image_hdus(fits_bytes: bytes) -> list[dict]:
    # Lazy import keeps first page render lightweight; astropy is needed only
    # after the user chooses to fetch and inspect an actual FITS data product.
    from astropy.io import fits

    images = []
    with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None or not isinstance(data, np.ndarray) or data.ndim not in (2, 3):
                continue
            if not np.issubdtype(data.dtype, np.number):
                continue
            array = np.asarray(data, dtype=np.float32)
            if array.ndim == 3:
                # Keep the spectral/channel axis and spatial axes for the UI.
                array = array.reshape((-1, array.shape[-2], array.shape[-1]))
            images.append({
                "label": f"HDU {index}: {hdu.name or 'IMAGE'} · {array.shape}",
                "data": array,
                "unit": str(hdu.header.get("BUNIT", "instrument data units")),
                "header": {key: str(hdu.header.get(key, "")) for key in ("EXTNAME", "BUNIT", "OBJECT", "DATE-OBS", "TELESCOP", "INSTRUME")},
            })
    return images


def _finite_display_array(image: np.ndarray) -> np.ndarray:
    """Downsample very large images for browser responsiveness, preserving measured pixel values."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Selected FITS plane is not 2D.")
    stride = max(1, int(np.ceil(max(image.shape) / 700)))
    return image[::stride, ::stride]


def show_products(records: list[dict]):
    st.success(f"Found {len(records)} actual SPHEREx archive product(s) in this sky area.")
    summary = []
    for index, row in enumerate(records):
        try:
            lo = float(_value(row, "em_min")) * 1e6
            hi = float(_value(row, "em_max")) * 1e6
            wave = f"{lo:.3f}–{hi:.3f}" if np.isfinite(lo) and np.isfinite(hi) else ""
        except (ValueError, TypeError):
            wave = ""
        summary.append({
            "#": index + 1,
            "Collection": row.get("_queried_collection", ""),
            "Product": _value(row, "obs_id", "obs_publisher_did", "obs_title", "filename") or "SPHEREx product",
            "Type": _value(row, "dataproduct_type"),
            "Wavelength (μm)": wave,
            "Observation": _value(row, "t_min", "t_max"),
        })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    labels = []
    url_by_label = {}
    for index, row in enumerate(records):
        url = _value(row, "access_url", "access_reference", "accessurl")
        title = _value(row, "obs_id", "obs_title", "filename") or f"Product {index + 1}"
        label = f"{index + 1}. {title[:100]}"
        if url.startswith("https://"):
            labels.append(label)
            url_by_label[label] = url
    if not labels:
        st.info("Products were listed, but IRSA did not return a usable HTTPS FITS access URL. Open the product metadata in IRSA.")
        return

    chosen = st.selectbox("Choose an actual SPHEREx FITS product to inspect", labels)
    if st.button("Load infrared image / spectrum from this FITS product"):
        try:
            with st.spinner("Downloading selected SPHEREx FITS product…"):
                raw = download_fits_product(url_by_label[chosen])
                image_hdus = _extract_image_hdus(raw)
            if not image_hdus:
                st.error("This product contains no numeric 2D/3D FITS image extension that can be plotted.")
                return
            st.session_state["loaded_spherex_product"] = chosen
            st.session_state["loaded_spherex_hdus"] = image_hdus
        except Exception as exc:
            st.error(f"Could not load this FITS product: {exc}")

    image_hdus = st.session_state.get("loaded_spherex_hdus", [])
    if image_hdus and st.session_state.get("loaded_spherex_product") == chosen:
        hdu_label = st.selectbox("Image / spectral extension", [item["label"] for item in image_hdus])
        selected_hdu = next(item for item in image_hdus if item["label"] == hdu_label)
        cube = selected_hdu["data"]
        if cube.ndim == 3:
            channel = st.slider("Spectral channel / plane", 0, cube.shape[0] - 1, 0)
            plane = cube[channel]
        else:
            channel = None
            plane = cube
        display = _finite_display_array(plane)
        finite = display[np.isfinite(display)]
        if not finite.size:
            st.error("This image plane contains no finite pixel data.")
            return
        figure = go.Figure(go.Heatmap(
            z=display, colorscale="Inferno", colorbar=dict(title=selected_hdu["unit"]),
            hovertemplate="x=%{x}, y=%{y}<br>measured value=%{z:.5g}<extra></extra>",
            zmin=float(np.nanpercentile(finite, 1)), zmax=float(np.nanpercentile(finite, 99)),
        ))
        figure.update_layout(
            title=f"SPHEREx measured FITS pixel values · {selected_hdu['unit']}" + (f" · plane {channel}" if channel is not None else ""),
            xaxis_title="Image pixel x", yaxis_title="Image pixel y", yaxis_autorange="reversed",
            paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="white", height=650,
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Inferno colors encode the selected FITS image plane's actual stored pixel values. This is an image-plane visualization, not automatically a planet-only spectrum or calibrated disk-integrated flux; consult the product's BUNIT and archive documentation.")
        st.json(selected_hdu["header"])
        st.link_button("Open original SPHEREx product", url_by_label[chosen])


def main():
    st.title("SPHEREx Infrared Data Explorer")
    st.write("Search a target's sky position in JPL Horizons, find real SPHEREx archive observations, then load a FITS product and display its measured image values with Inferno colors.")
    with st.sidebar:
        target = st.text_input("Planet / object / Horizons ID", value="Mars", help="Examples: Mars, Jupiter, Ceres, 499, or another Horizons target.")
        observed_on = st.date_input("Target position date (UTC)", value=date.today())
        radius = st.select_slider("Search radius", options=[0.02, 0.05, 0.1, 0.25, 0.5], value=0.1,
                                  format_func=lambda value: f"{value}°")

    st.caption("Initial load makes no network requests. Submit the form to query Horizons and IRSA.")
    with st.form("archive_search"):
        submitted = st.form_submit_button("Find actual SPHEREx infrared data", type="primary")

    if submitted:
        if not target.strip():
            st.error("Enter a target name or Horizons ID.")
            return
        try:
            with st.spinner("Resolving target's apparent position with JPL Horizons…"):
                ra, dec = _radec_from_horizons(target.strip(), observed_on)
            st.write(f"**{target.strip()}** on {observed_on}: RA {ra:.6f}°, Dec {dec:.6f}°")
            with st.spinner("Searching IRSA SPHEREx QR3 / DR1 coverage…"):
                records = query_spherex(ra, dec, float(radius))
            st.session_state["spherex_search_records"] = records
            st.session_state["spherex_search_key"] = (target.strip(), observed_on.isoformat(), float(radius))
            if records:
                show_products(records)
            else:
                st.info("IRSA returned no SPHEREx data products at this position/radius. This is not a zero infrared measurement; it means no matching archive product was found.")
        except requests.Timeout:
            st.error("NASA archive timed out. The app remains usable; retry the search or try again later.")
        except requests.RequestException as exc:
            st.error(f"A NASA service could not be reached: {exc}")
        except Exception as exc:
            st.error(f"Search failed: {exc}")
    elif st.session_state.get("spherex_search_records"):
        show_products(st.session_state["spherex_search_records"])

    st.markdown("Sources: [IRSA SPHEREx archive](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [IRSA SIA documentation](https://irsa.ipac.caltech.edu/docs/program_interface/SIA2.html) · [JPL Horizons API](https://ssd.jpl.nasa.gov/api/horizons.api.html)")


if __name__ == "__main__":
    main()
