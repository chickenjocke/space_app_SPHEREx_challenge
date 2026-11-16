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
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA2/servlet/Query"
SPHEREX_COLLECTIONS = ("spherex_qr3", "spherex_dr1", "spherex_qr2")

# Horizons names such as "Mars" are ambiguous (Mars barycenter, Mars, and
# spacecraft can all match). Use the NAIF body IDs for common Solar System targets.
HORIZONS_TARGET_IDS = {
    "sun": "10", "mercury": "199", "venus": "299", "earth": "399",
    "moon": "301", "mars": "499", "jupiter": "599", "saturn": "699",
    "uranus": "799", "neptune": "899", "pluto": "999",
}
RA_CELL = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s*$")
DEC_CELL = re.compile(r"^\s*([+-])\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s*$")


def _radec_from_horizons(target: str, obs_date: date) -> tuple[float, float]:
    """Resolve target's apparent geocentric RA/Dec using JPL Horizons."""
    cleaned_target = target.strip()
    # Map planet names to unambiguous NAIF IDs; allow numeric Horizons IDs as entered.
    horizons_id = HORIZONS_TARGET_IDS.get(cleaned_target.casefold(), cleaned_target)
    next_day = obs_date + timedelta(days=1)
    params = {
        "format": "json", "COMMAND": f"'{horizons_id}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'500@399'",
        "QUANTITIES": "'1'", "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{obs_date.isoformat()}'", "STOP_TIME": f"'{next_day.isoformat()}'",
        "STEP_SIZE": "'1 d'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(4, 15))
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        api_error = payload.get("error") or result.strip() or "Horizons returned no ephemeris rows."
        raise ValueError(f"JPL Horizons could not resolve {cleaned_target!r} (queried ID {horizons_id}) for {obs_date}: {api_error[:500]}")

    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    for row in csv.reader(io.StringIO(block)):
        cells = [cell.strip().strip('"') for cell in row]
        for index, cell in enumerate(cells):
            ra_match = RA_CELL.fullmatch(cell)
            if not ra_match:
                continue
            for dec_cell in cells[index + 1:index + 4]:
                dec_match = DEC_CELL.fullmatch(dec_cell)
                if dec_match:
                    hours, minutes, seconds = map(float, ra_match.groups())
                    sign = -1.0 if dec_match.group(1) == "-" else 1.0
                    degrees, arcminutes, arcseconds = map(float, dec_match.groups()[1:])
                    ra_deg = 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)
                    dec_deg = sign * (degrees + arcminutes / 60.0 + arcseconds / 3600.0)
                    if 0 <= ra_deg < 360 and -90 <= dec_deg <= 90:
                        return ra_deg, dec_deg

    sample = " | ".join(line.strip() for line in block.splitlines() if line.strip())
    raise ValueError(f"Horizons responded but RA/Dec parsing failed. Response: {sample[:400]}")


def _parse_votable(content: bytes) -> list[dict]:
    root = ET.fromstring(content)
    fields = [node.attrib.get("name") or node.attrib.get("ID") or f"column_{i}"
              for i, node in enumerate(n for n in root.iter() if n.tag.rsplit("}", 1)[-1] == "FIELD")]
    records = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "TR":
            cells = [child.text or "" for child in node if child.tag.rsplit("}", 1)[-1] == "TD"]
            if cells:
                records.append({fields[i] if i < len(fields) else f"column_{i}": value
                                for i, value in enumerate(cells)})
    return records


def _value(record: dict, *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def query_spherex_products(ra_deg: float, dec_deg: float, radius_deg: float) -> list[dict]:
    """Search IRSA SIA v2 for SPHEREx products covering this sky position."""
    products, seen, errors = [], set(), []
    for collection in SPHEREX_COLLECTIONS:
        try:
            response = requests.get(
                IRSA_SIA_URL,
                params={"POS": f"CIRCLE {ra_deg:.8f} {dec_deg:.8f} {radius_deg}",
                       "BAND": "0.75e-6 5.0e-6", "COLLECTION": collection},
                headers={"Accept": "application/x-votable+xml"}, timeout=(4, 15),
            )
            response.raise_for_status()
            if not response.content.strip():
                continue
            for record in _parse_votable(response.content):
                key = _value(record, "obs_publisher_did", "obs_id", "access_url", "access_reference")
                if key and key not in seen:
                    seen.add(key)
                    record["_collection"] = collection
                    products.append(record)
        except (requests.RequestException, ET.ParseError) as exc:
            errors.append(f"{collection}: {exc}")
    if not products and errors:
        raise RuntimeError("IRSA search failed: " + " | ".join(errors)[:700])
    return products


@st.cache_data(ttl=1800, show_spinner=False)
def download_fits(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS IRSA product links are accepted.")
    max_bytes = 80 * 1024 * 1024
    chunks, total = [], 0
    with requests.get(url, timeout=(4, 25), stream=True) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", "0") or 0) > max_bytes:
            raise ValueError("Product exceeds the 80 MB display limit; download it directly from IRSA.")
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Product exceeded the 80 MB display limit.")
                chunks.append(chunk)
    return b"".join(chunks)


def _fits_planes(raw: bytes) -> list[dict]:
    from astropy.io import fits
    planes = []
    with fits.open(io.BytesIO(raw), memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            data = getattr(hdu, "data", None)
            if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype, np.number):
                continue
            data = np.asarray(data, dtype=np.float32)
            unit = str(hdu.header.get("BUNIT", "FITS stored units (BUNIT absent)"))
            if data.ndim == 2:
                planes.append({"label": f"HDU {index}: {hdu.name} {data.shape}", "data": data, "unit": unit})
            elif data.ndim == 3:
                for channel, plane in enumerate(data):
                    planes.append({"label": f"HDU {index}: {hdu.name}, plane {channel} {plane.shape}",
                                   "data": plane, "unit": unit})
    return planes


def _show_products(products: list[dict]):
    if not products:
        st.info("No SPHEREx products cover this position and wavelength range. That is not a zero infrared measurement.")
        return
    st.success(f"IRSA found {len(products)} actual SPHEREx archive product(s).")
    rows, labels, url_map = [], [], {}
    for index, record in enumerate(products, 1):
        name = _value(record, "obs_id", "obs_publisher_did", "obs_title", "filename") or f"Product {index}"
        url = _value(record, "access_url", "access_reference", "accessurl")
        try:
            wave = f"{float(_value(record, 'em_min'))*1e6:.3f}–{float(_value(record, 'em_max'))*1e6:.3f} μm"
        except (TypeError, ValueError):
            wave = "not supplied"
        rows.append({"#": index, "Release": record.get("_collection", ""), "Product": name,
                     "Type": _value(record, "dataproduct_type"), "Wavelength range": wave,
                     "Observation time": _value(record, "t_min", "t_max")})
        if url.startswith("https://"):
            label = f"{index}: {name[:90]}"
            labels.append(label)
            url_map[label] = url
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if not labels:
        st.warning("The returned metadata did not include usable HTTPS product links.")
        return

    selected = st.selectbox("Select a real FITS product", labels)
    st.link_button("Open/download selected product from IRSA", url_map[selected])
    if st.button("Display selected product with Inferno", type="primary"):
        try:
            with st.spinner("Downloading and opening the selected FITS product…"):
                planes = _fits_planes(download_fits(url_map[selected]))
            if planes:
                st.session_state["fits_product_url"] = url_map[selected]
                st.session_state["fits_planes"] = planes
            else:
                st.error("No numeric 2-D image or 3-D cube was found in that FITS product.")
        except Exception as exc:
            st.error(f"Could not display the FITS product: {exc}")

    planes = st.session_state.get("fits_planes", [])
    if planes and st.session_state.get("fits_product_url") == url_map[selected]:
        plane_label = st.selectbox("FITS image/cube plane", [plane["label"] for plane in planes])
        plane = next(item for item in planes if item["label"] == plane_label)
        image = plane["data"]
        stride = max(1, int(np.ceil(max(image.shape) / 700)))
        image = image[::stride, ::stride]
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = np.nanpercentile(finite, [1, 99])
            if low == high:
                low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
            if low == high:
                high = low + 1
            figure = go.Figure(go.Heatmap(z=image, colorscale="Inferno", zmin=low, zmax=high,
                colorbar={"title": plane["unit"]}, hovertemplate="x=%{x}, y=%{y}<br>pixel value=%{z:.6g}<extra></extra>"))
            figure.update_layout(title=f"Actual SPHEREx FITS pixel values · {plane['unit']}",
                xaxis_title="Pixel x", yaxis_title="Pixel y", yaxis_autorange="reversed",
                paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="white", height=650)
            st.plotly_chart(figure, use_container_width=True)
            st.caption("Inferno colors encode the actual values stored in the selected FITS plane. This is not automatically a planet-only spectrum or disk-integrated flux.")


def main():
    st.title("SPHEREx Infrared Explorer")
    st.caption("Network queries run only after submitting a search. Common planet names are mapped to unambiguous Horizons IDs (Mars → 499).")
    with st.sidebar:
        st.header("Search")
        target = st.text_input("Object name or JPL Horizons ID", value="Mars")
        obs_date = st.date_input("Observation date (UTC)", value=date.today())
        radius = st.select_slider("Search radius", options=[0.02, 0.05, 0.1, 0.25, 0.5], value=0.1,
                                  format_func=lambda value: f"{value}°")

    with st.form("spherex_search"):
        submitted = st.form_submit_button("Find real SPHEREx infrared data", type="primary")
    if submitted:
        if not target.strip():
            st.error("Enter a target name or Horizons ID.")
        else:
            try:
                with st.spinner("Getting target coordinates from JPL Horizons…"):
                    ra_deg, dec_deg = _radec_from_horizons(target.strip(), obs_date)
                st.write(f"**{target.strip()}**, {obs_date}: RA {ra_deg:.6f}°, Dec {dec_deg:.6f}°")
                with st.spinner("Searching IRSA SPHEREx releases…"):
                    products = query_spherex_products(ra_deg, dec_deg, float(radius))
                st.session_state["spherex_products"] = products
            except requests.Timeout:
                st.error("JPL or IRSA timed out. Retry the search; the page remains usable.")
            except Exception as exc:
                st.error(f"Search failed: {exc}")
    if "spherex_products" in st.session_state:
        _show_products(st.session_state["spherex_products"])

    st.markdown("Sources: [JPL Horizons API](https://ssd.jpl.nasa.gov/api/horizons.api.html) · [IRSA SPHEREx archive](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [IRSA SIA v2 documentation](https://irsa.ipac.caltech.edu/docs/program_interface/SIA2.html)")


if __name__ == "__main__":
    main()
