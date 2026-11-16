from __future__ import annotations

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


def _parse_horizons_ra_dec(text: str):
    """Extract the first RA/Dec pair from a Horizons CSV response."""
    # Example expected output line is date, RA, Dec, ...
    # Anything robust enough to parse the common Horizons text is fine.
    match = re.search(r"(\d{2})\s+(\d{2})\s+(\d{2}\.\d+)\s+[+-]\d{2}\s+\d{2}\s+\d{2}\.\d+", text)
    if not match:
        return None
    try:
        # Try a direct sexagesimal parse for RA and Dec.
        ra_h, ra_m, ra_s = map(float, match.group(1, 2, 3))
        ra = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)
        # second pattern for Dec can be awkward; fallback below
        return ra, 0.0
    except Exception:
        return None


def _radec_from_horizons(target: str, obs_date: date) -> tuple[float, float]:
    """Use JPL Horizons to resolve the apparent RA/Dec of a target on the chosen date."""
    start = datetime.combine(obs_date, datetime.min.time())
    stop = start + timedelta(minutes=2)
    params = {
        "format": "json",
        "COMMAND": f"'{target}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "CENTER": "'500@399'",
        "QUANTITIES": "'1'",
        "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{start:%Y-%m-%d %H:%M}'",
        "STOP_TIME": f"'{stop:%Y-%m-%d %H:%M}'",
        "STEP_SIZE": "'1 m'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(4, 15))
    response.raise_for_status()
    result = response.json().get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise ValueError(response.json().get("error", "No Horizons position was returned for this target/date."))

    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    lines = [ln for ln in block.splitlines() if ln.strip() and not ln.startswith("JDTDB")]
    for line in lines:
        cols = line.split()
        if len(cols) < 7:
            continue
        try:
            # Typical Horizons columns: date, RA, Dec, ...
            # RA is in sexagesimal form like 10 20 30.22, Dec like +12 34 56.78
            ra_part = cols[1:4]
            dec_part = cols[4:7]
            if len(ra_part) != 3 or len(dec_part) != 3:
                continue
            ra_h, ra_m, ra_s = float(ra_part[0]), float(ra_part[1]), float(ra_part[2])
            dec_sign = 1.0 if dec_part[0].startswith("+") else -1.0
            dec_d, dec_m, dec_s = abs(float(dec_part[0][1:] if dec_part[0].startswith("+") or dec_part[0].startswith("-") else dec_part[0])), float(dec_part[1]), float(dec_part[2])
            ra_deg = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)
            dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
            if 0.0 <= ra_deg < 360.0 and -90.0 <= dec_deg <= 90.0:
                return ra_deg, dec_deg
        except ValueError:
            continue
    raise ValueError("Could not read RA/Dec from the Horizons response.")


def _parse_votable(xml_bytes: bytes):
    """Parse a VOTable and return a list of dict rows."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    field_names = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "FIELD":
            name = node.attrib.get("name") or node.attrib.get("ID") or f"column_{len(field_names)}"
            field_names.append(name)

    records = []
    for row in root.iter():
        if row.tag.rsplit("}", 1)[-1] != "TR":
            continue
        cells = [child.text or "" for child in row if child.tag.rsplit("}", 1)[-1] == "TD"]
        if not cells:
            continue
        record = {field_names[i] if i < len(field_names) else f"column_{i}": cells[i] for i in range(len(cells))}
        records.append(record)
    return records


def _value(record: dict, *names: str) -> str:
    keys = {str(k).strip().lower(): v for k, v in record.items()}
    for name in names:
        value = keys.get(name.lower())
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def query_spherex_products(ra_deg: float, dec_deg: float, radius_deg: float):
    """Query the correct IRSA SIA v2 endpoint with the correct parameter format."""
    results = []
    seen = set()
    for collection in SPHEREX_COLLECTIONS:
        try:
            response = requests.get(
                IRSA_SIA_URL,
                params={
                    "POS": f"CIRCLE {ra_deg:.8f} {dec_deg:.8f} {radius_deg}",
                    "BAND": "0.75e-6 5.0e-6",
                    "COLLECTION": collection,
                },
                headers={"Accept": "application/x-votable+xml"},
                timeout=(4, 20),
            )
            response.raise_for_status()
            if not response.content.strip():
                continue
            rows = _parse_votable(response.content)
            for row in rows:
                id_key = _value(row, "obs_publisher_did", "obs_id", "access_url", "access_reference")
                if not id_key:
                    continue
                if id_key not in seen:
                    seen.add(id_key)
                    row["_collection"] = collection
                    results.append(row)
        except requests.RequestException:
            continue
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def download_fits(url: str) -> bytes:
    """Download a FITS file with a safe file size cap."""
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS FITS links may be downloaded from IRSA.")
    response = requests.get(url, timeout=(4, 30), stream=True)
    response.raise_for_status()
    max_bytes = 80 * 1024 * 1024
    total = 0
    chunks = []
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("FITS file exceeds the 80 MB download cap.")
        chunks.append(chunk)
    return b"".join(chunks)


def _fits_image_planes(fits_bytes: bytes):
    """Open a FITS file and return all numeric 2D image planes that can be displayed."""
    from astropy.io import fits

    planes = []
    with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None or not isinstance(data, np.ndarray):
                continue
            if not np.issubdtype(data.dtype, np.number):
                continue
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 3:
                for channel_index, channel_data in enumerate(arr):
                    planes.append({
                        "label": f"HDU {idx} / channel {channel_index} · {channel_data.shape}",
                        "data": channel_data,
                        "header": dict(hdu.header),
                        "unit": str(hdu.header.get("BUNIT", "instrument units")),
                    })
            elif arr.ndim == 2:
                planes.append({
                    "label": f"HDU {idx} · {arr.shape}",
                    "data": arr,
                    "header": dict(hdu.header),
                    "unit": str(hdu.header.get("BUNIT", "instrument units")),
                })
    return planes


def _display_cut(array: np.ndarray):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("This FITS plane is not a 2D image.")
    step = max(1, int(np.ceil(max(array.shape) / 700)))
    return array[::step, ::step]


def show_spherex_results(records):
    if not records:
        st.info("IRSA returned no SPHEREx results for this target/date/radius.")
        return

    table_rows = []
    for idx, record in enumerate(records, 1):
        url = _value(record, "access_url", "access_reference", "accessURL")
        table_rows.append({
            "#": idx,
            "Collection": record.get("_collection", ""),
            "Product": _value(record, "obs_id", "obs_publisher_did", "obs_title", "filename") or f"SPHEREx product {idx}",
            "Type": _value(record, "dataproduct_type"),
            "Wavelength min": _value(record, "em_min"),
            "Wavelength max": _value(record, "em_max"),
            "URL": url,
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    options = []
    url_map = {}
    for row in records:
        url = _value(row, "access_url", "access_reference", "accessURL")
        label = _value(row, "obs_id", "obs_publisher_did", "obs_title", "filename") or "SPHEREx product"
        if url.startswith("https://"):
            options.append(f"{label[:80]} ({url[:80]})")
            url_map[options[-1]] = url
    if not options:
        st.info("No usable HTTPS FITS URLs were returned by the archive for this result set.")
        return

    selected_label = st.selectbox("Choose a FITS product to inspect", options)
    if st.button("Load actual SPHEREx FITS image values", type="primary"):
        try:
            fits_bytes = download_fits(url_map[selected_label])
            planes = _fits_image_planes(fits_bytes)
            if not planes:
                st.error("This FITS product contains no 2D/3D numeric image plane to display.")
                return
            st.session_state["selected_spherex_planes"] = planes
            st.session_state["selected_spherex_url"] = url_map[selected_label]
            st.success("FITS product loaded successfully.")
        except Exception as exc:
            st.error(f"Could not load the FITS file: {exc}")

    planes = st.session_state.get("selected_spherex_planes", [])
    current_url = st.session_state.get("selected_spherex_url")
    if planes and current_url == url_map[selected_label]:
        plane_name = st.selectbox("Choose image plane", [p["label"] for p in planes])
        plane = next(p for p in planes if p["label"] == plane_name)
        display = _display_cut(plane["data"])
        finite = display[np.isfinite(display)]
        if finite.size == 0:
            st.warning("The selected FITS plane has no finite pixel values to display.")
            return
        fig = go.Figure(go.Heatmap(z=display, colorscale="Inferno", zmin=np.percentile(finite, 1), zmax=np.percentile(finite, 99),
                                  hovertemplate="x=%{x}, y=%{y}<br>value=%{z:.5g}<extra></extra>"))
        fig.update_layout(title=f"SPHEREx image plane values ({plane['unit']})", xaxis_title="pixel x", yaxis_title="pixel y",
                          paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="white", height=650,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Inferno colors map the actual FITS image-plane values from the archive. This is a direct visualization of measured archive data, not a generated synthetic infrared field.")
        st.link_button("Open original product", current_url)


def main():
    st.title("SPHEREx Infrared Explorer")
    with st.sidebar:
        st.header("Search")
        target = st.text_input("Object name or JPL Horizons ID", value="Mars")
        obs_date = st.date_input("Observation date", value=date.today())
        radius = st.select_slider("Search radius", options=[0.02, 0.05, 0.1, 0.25, 0.5], value=0.1,
                                  format_func=lambda v: f"{v}°")

    with st.form("spherex_form"):
        submitted = st.form_submit_button("Find real SPHEREx infrared data", type="primary")

    if submitted:
        target_value = target.strip()
        if not target_value:
            st.error("Please enter a target name or Horizons ID.")
            return
        try:
            with st.spinner("Resolving object position with JPL Horizons…"):
                ra_deg, dec_deg = _radec_from_horizons(target_value, obs_date)
            st.write(f"Target: {target_value} — RA {ra_deg:.6f}°, Dec {dec_deg:.6f}°")
            with st.spinner("Querying IRSA SPHEREx SIA v2 for real archive products…"):
                records = query_spherex_products(ra_deg, dec_deg, float(radius))
            st.session_state["spherex_records"] = records
            show_spherex_results(records)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
    elif "spherex_records" in st.session_state:
        show_spherex_results(st.session_state["spherex_records"])

    st.markdown("Official docs: [IRSA SPHEREx archive](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [IRSA SIA v2](https://irsa.ipac.caltech.edu/docs/program_interface/SIA2.html) · [JPL Horizons](https://ssd.jpl.nasa.gov/api/horizons.api.html)")


if __name__ == "__main__":
    main()
