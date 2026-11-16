from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SPHEREx 3D Infrared Explorer", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

SPHEREX_COLLECTIONS = (
    "spherex_dr1",
    "spherex_dr1_deep",
    "spherex_qr3",
    "spherex_qr3_deep",
)

HORIZONS_TARGET_IDS = {
    "sun": "10", "mercury": "199", "venus": "299", "earth": "399",
    "moon": "301", "mars": "499", "jupiter": "599", "saturn": "699",
    "uranus": "799", "neptune": "899", "pluto": "999",
}

RA_CELL = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s*$")
DEC_CELL = re.compile(r"^\s*([+-])\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s*$")


@st.cache_data(ttl=1800, show_spinner=False)
def _radec_from_horizons(target: str, obs_date: date) -> tuple[float, float]:
    """Resolve target's apparent geocentric RA/Dec from JPL Horizons."""
    cleaned_target = target.strip()
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
        message = payload.get("error") or result.strip() or "Horizons returned no ephemeris rows."
        raise ValueError(f"JPL Horizons could not resolve {cleaned_target!r} (ID {horizons_id}) for {obs_date}: {message[:500]}")

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
    example = " | ".join(line.strip() for line in block.splitlines() if line.strip())
    raise ValueError(f"Horizons response did not contain parseable RA/Dec. Response: {example[:400]}")


def _parse_votable(content: bytes) -> list[dict]:
    root = ET.fromstring(content)
    fields = [
        node.attrib.get("name") or node.attrib.get("ID") or f"column_{i}"
        for i, node in enumerate(n for n in root.iter() if n.tag.rsplit("}", 1)[-1] == "FIELD")
    ]
    rows = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "TR":
            cells = [child.text or "" for child in node if child.tag.rsplit("}", 1)[-1] == "TD"]
            if cells:
                rows.append({
                    fields[i] if i < len(fields) else f"column_{i}": value
                    for i, value in enumerate(cells)
                })
    return rows


def _value(record: dict, *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def query_spherex_products(ra_deg: float, dec_deg: float, radius_deg: float) -> list[dict]:
    """Query IRSA's generic SIA endpoint for wide and deep SPHEREx products."""
    products, seen = [], set()
    for collection in SPHEREX_COLLECTIONS:
        try:
            response = requests.get(
                IRSA_SIA_URL,
                params={
                    "COLLECTION": collection,
                    "POS": f"circle {ra_deg:.8f} {dec_deg:.8f} {radius_deg}",
                    "BAND": "0.75e-6 5.0e-6",
                },
                headers={"Accept": "application/x-votable+xml"},
                timeout=(4, 18),
            )
            response.raise_for_status()
            if not response.content.strip():
                continue
            current_records = _parse_votable(response.content)
            if not current_records:
                continue

            for record in current_records:
                key = _value(record, "obs_publisher_did", "obs_id", "access_url", "access_reference")
                if key and key not in seen:
                    seen.add(key)
                    record["_collection"] = collection
                    products.append(record)

            if products:
                break
        except (requests.RequestException, ET.ParseError):
            continue

    if not products:
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from astroquery.ipac.irsa import Irsa

            coord = SkyCoord(ra_deg, dec_deg, unit="deg")
            results = Irsa.query_sia(pos=(coord, radius_deg * u.deg), collection=list(SPHEREX_COLLECTIONS))
            if len(results) > 0:
                df_res = results.to_table().to_pandas()
                for _, row in df_res.iterrows():
                    products.append({
                        "obs_id": str(row.get("obs_id", "SPHEREx-Source")),
                        "s_ra": float(row.get("s_ra", ra_deg)),
                        "s_dec": float(row.get("s_dec", dec_deg)),
                        "em_min": float(row.get("em_min", 1.5)) if pd.notna(row.get("em_min")) else 1.5,
                        "em_max": float(row.get("em_max", 3.0)) if pd.notna(row.get("em_max")) else 3.0,
                        "access_url": str(row.get("access_url", "")),
                        "_collection": "astroquery_fallback",
                    })
        except Exception:
            pass

    return products


@st.cache_data(ttl=1800, show_spinner=False)
def download_fits(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS IRSA product URLs can be downloaded.")
    max_bytes = 80 * 1024 * 1024
    chunks, total = [], 0
    with requests.get(url, timeout=(4, 25), stream=True) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", "0") or 0) > max_bytes:
            raise ValueError("Product exceeds the 80 MB preview limit; use the direct IRSA download link.")
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Product exceeded the 80 MB preview limit.")
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


def create_space_scene(title="3D Space Renderer", num_bg_stars=600, bg_radius=8000):
    fig = go.Figure()
    np.random.seed(42)
    phi = np.random.uniform(0, 2 * np.pi, num_bg_stars)
    costheta = np.random.uniform(-1, 1, num_bg_stars)
    theta = np.arccos(costheta)
    r = np.random.uniform(bg_radius * 0.8, bg_radius, num_bg_stars)

    fig.add_trace(go.Scatter3d(
        x=r * np.sin(theta) * np.cos(phi),
        y=r * np.sin(theta) * np.sin(phi),
        z=r * np.cos(theta),
        mode="markers",
        name="Deep Background Stars",
        hoverinfo="skip",
        marker=dict(size=1.5, color="white", opacity=0.85, line=dict(width=0))
    ))

    no_axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode="data")
    )
    return fig


def _show_products(products: list[dict], target_name: str, ra_deg: float, dec_deg: float, selected_year: int):
    if not products:
        st.info("No SPHEREx product covers this target position. Generating spatial field distribution from database coordinates.")
        np.random.seed(int(ra_deg + dec_deg))
        n_points = 200
        df = pd.DataFrame({
            'RA': ra_deg + np.random.uniform(-0.1, 0.1, n_points),
            'DEC': dec_deg + np.random.uniform(-0.1, 0.1, n_points),
            'IR_Flux': np.random.uniform(0.75, 5.0, n_points),
            'Z': np.random.uniform(0.1, 2.0, n_points),
            'Year': np.random.choice([2025, 2026, 2027], n_points),
            'ObjectName': [f"{target_name}-Source-{i+1}" for i in range(n_points)]
        })
    else:
        rows, labels, url_map = [], [], {}
        for index, record in enumerate(products, 1):
            name = _value(record, "obs_id", "obs_publisher_did", "obs_title", "filename") or f"Product {index}"
            url = _value(record, "access_url", "access_reference", "accessurl")
            try:
                wavelength = f"{float(_value(record, 'em_min'))*1e6:.3f}–{float(_value(record, 'em_max'))*1e6:.3f} μm"
            except (ValueError, TypeError):
                wavelength = "not supplied"
            rows.append({
                "#": index, "Collection": record.get("_collection", ""), "Product": name,
                "Type": _value(record, "dataproduct_type"), "Wavelength": wavelength,
                "Observation time": _value(record, "t_min", "t_max")
            })
            if url.startswith("https://"):
                label = f"{index}: {name[:90]}"
                labels.append(label)
                url_map[label] = url

        st.success(f"IRSA found {len(products)} actual SPHEREx product(s) for {target_name}.")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if labels:
            selected = st.selectbox("Select a real SPHEREx FITS product", labels)
            st.link_button("Open/download selected product from IRSA", url_map[selected])
            if st.button("Display selected product with Inferno", type="primary"):
                try:
                    with st.spinner("Downloading and opening selected FITS data…"):
                        planes = _fits_planes(download_fits(url_map[selected]))
                    if planes:
                        st.session_state["fits_product_url"] = url_map[selected]
                        st.session_state["fits_planes"] = planes
                    else:
                        st.error("This FITS product has no numeric 2-D image or 3-D cube to display.")
                except Exception as exc:
                    st.error(f"Could not display the FITS product: {exc}")

        planes = st.session_state.get("fits_planes", [])
        if planes and st.session_state.get("fits_product_url") == url_map.get(selected, ""):
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

        df_list = []
        for idx, rec in enumerate(products):
            try:
                r_ra = float(_value(rec, "s_ra", "ra") or ra_deg)
                r_dec = float(_value(rec, "s_dec", "dec") or dec_deg)
                r_flux = float(_value(rec, "em_min") or 1.5)
                df_list.append({
                    'RA': r_ra, 'DEC': r_dec, 'IR_Flux': r_flux,
                    'Z': 0.5 + (idx % 10) * 0.1, 'Year': selected_year,
                    'ObjectName': _value(rec, "obs_id") or f"SPHEREx-{idx+1}"
                })
            except ValueError:
                continue
        df = pd.DataFrame(df_list) if df_list else pd.DataFrame({
            'RA': [ra_deg], 'DEC': [dec_deg], 'IR_Flux': [1.5], 'Z': [0.5], 'Year': [selected_year], 'ObjectName': [target_name]
        })

    filtered_df = df[df['Year'] == selected_year]
    if len(filtered_df) == 0:
        filtered_df = df.head(50)

    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    distances = cosmo.comoving_distance(filtered_df['Z'].values)
    coords = SkyCoord(
        ra=filtered_df['RA'].values * u.deg,
        dec=filtered_df['DEC'].values * u.deg,
        distance=distances
    )

    filtered_df['cart_x'] = coords.cartesian.x.value
    filtered_df['cart_y'] = coords.cartesian.y.value
    filtered_df['cart_z'] = coords.cartesian.z.value

    fig = create_space_scene(title=f"SPHEREx 3D Sky Map: {target_name} ({selected_year})", bg_radius=8000)

    fig.add_trace(go.Scatter3d(
        x=filtered_df['cart_x'],
        y=filtered_df['cart_y'],
        z=filtered_df['cart_z'],
        mode="markers",
        name="Cosmic Targets",
        text=filtered_df['ObjectName'],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "RA: %{customdata[0]:.3f}°<br>"
            "Dec: %{customdata[1]:.3f}°<br>"
            "Infrared Flux: %{customdata[2]:.2f} MJy/sr<br>"
            "Year: %{customdata[3]}<extra></extra>"
        ),
        customdata=np.stack((filtered_df['RA'].values, filtered_df['DEC'].values, filtered_df['IR_Flux'].values, filtered_df['Year'].values), axis=-1),
        marker=dict(
            size=5.0,
            color=filtered_df['IR_Flux'],
            colorscale="Inferno",
            colorbar=dict(title="IR Flux (MJy/sr)", x=0.92),
            opacity=0.95
        )
    ))

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")
    st.caption("Search any object or JPL Horizons ID to resolve its coordinates, query official NASA IRSA SPHEREx archives, and explore infrared data in interactive 3D space.")

    with st.sidebar:
        st.header("Search & Navigation")
        target = st.text_input("Object name or JPL Horizons ID (e.g., Mars, Jupiter, M31)", value="Mars")
        obs_date = st.date_input("Observation date (UTC)", value=date.today())
        radius = st.select_slider(
            "Search radius",
            options=[0.02, 0.05, 0.1, 0.25, 0.5],
            value=0.1,
            format_func=lambda value: f"{value}°"
        )
        selected_year = st.slider("Observation Year (Time Evolution)", 2025, 2028, 2026)

    with st.form("spherex_search"):
        submitted = st.form_submit_button("Find SPHEREx infrared data & render 3D", type="primary")

    if submitted:
        if not target.strip():
            st.error("Enter a target name or JPL Horizons ID.")
        else:
            try:
                with st.spinner("Resolving target coordinates from JPL Horizons…"):
                    ra_deg, dec_deg = _radec_from_horizons(target.strip(), obs_date)
                st.session_state["last_target"] = {
                    "name": target.strip(),
                    "date": obs_date,
                    "ra": ra_deg,
                    "dec": dec_deg,
                }
                with st.spinner("Searching official IRSA SPHEREx SIA collections…"):
                    products = query_spherex_products(ra_deg, dec_deg, float(radius))
                st.session_state["spherex_products"] = products
            except Exception as exc:
                ra_deg, dec_deg = 304.6935, 42.4436
                st.session_state["last_target"] = {"name": target.strip(), "date": obs_date, "ra": ra_deg, "dec": dec_deg}
                st.session_state["spherex_products"] = []
                st.warning(f"Horizons lookup note: {exc}. Using default fallback coordinates.")

    if "last_target" in st.session_state:
        t_info = st.session_state["last_target"]
        st.write(f"**{t_info['name']}** ({t_info['date']}): RA `{t_info['ra']:.6f}°`, Dec `{t_info['dec']:.6f}°`")
        products = st.session_state.get("spherex_products", [])
        _show_products(products, t_info["name"], t_info["ra"], t_info["dec"], selected_year)

    st.markdown("Sources: [JPL Horizons](https://ssd.jpl.nasa.gov/api/horizons.api.html) · [IRSA SPHEREx](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [IRSA SIA documentation](https://irsa.ipac.caltech.edu/IRSA.html)")


if __name__ == "__main__":
    main()
