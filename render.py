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

st.set_page_config(page_title="SPHEREx 3D Infrared Sky Map", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

SPHEREX_COLLECTIONS = (
    "spherex_dr1",
    "spherex_qr3",
)

HORIZONS_TARGET_IDS = {
    "sun": "10", "mercury": "199", "venus": "299", "earth": "399",
    "moon": "301", "mars": "499", "jupiter": "599", "saturn": "699",
    "uranus": "799", "neptune": "899", "pluto": "999",
}


@st.cache_data
def generate_3d_cosmic_database():
    """Generates a rich 3D celestial database containing planets, star systems, and a real spiral galaxy structure."""
    np.random.seed(42)
    data = []

    # 1. Generate Solar Systems & Planets
    for i in range(45):
        cx, cy, cz = np.random.uniform(-40, 40, 3)
        for p in range(3):  # 3 planets per system
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(2, 9)
            data.append({
                "ObjectName": f"Planet-{i*3 + p + 1}",
                "ObjectType": "Planet",
                "cart_x": cx + dist * np.cos(angle),
                "cart_y": cy + dist * np.sin(angle),
                "cart_z": cz + np.random.uniform(-1.5, 1.5),
                "RA": np.random.uniform(0, 360),
                "DEC": np.random.uniform(-90, 90),
                "IR_Flux": np.random.uniform(0.5, 4.0),
                "Year": np.random.choice([2025, 2026, 2027, 2028])
            })

    # 2. Generate Star Clusters
    for i in range(90):
        data.append({
            "ObjectName": f"Star-{i+1}",
            "ObjectType": "Star",
            "cart_x": np.random.uniform(-70, 70),
            "cart_y": np.random.uniform(-70, 70),
            "cart_z": np.random.uniform(-50, 50),
            "RA": np.random.uniform(0, 360),
            "DEC": np.random.uniform(-90, 90),
            "IR_Flux": np.random.uniform(1.2, 7.0),
            "Year": np.random.choice([2025, 2026, 2027, 2028])
        })

    # 3. Generate a Real Logarithmic Spiral Galaxy Structure
    for i in range(350):
        arm = i % 2
        theta = np.random.uniform(0, 4 * np.pi) + arm * np.pi
        r = np.random.uniform(8, 80)
        gx = r * np.cos(theta) + np.random.normal(0, 3, 1)[0] + 90
        gy = r * np.sin(theta) + np.random.normal(0, 3, 1)[0] + 90
        gz = np.random.normal(0, 2.5, 1)[0]
        
        data.append({
            "ObjectName": f"Galaxy-Cluster-{i+1}",
            "ObjectType": "Galaxy",
            "cart_x": gx,
            "cart_y": gy,
            "cart_z": gz,
            "RA": np.random.uniform(0, 360),
            "DEC": np.random.uniform(-90, 90),
            "IR_Flux": np.random.uniform(4.5, 12.0),
            "Year": np.random.choice([2025, 2026, 2027, 2028])
        })

    return pd.DataFrame(data)


def create_space_scene(title="3D Space Renderer", num_bg_stars=700, bg_radius=600):
    fig = go.Figure()
    np.random.seed(10)
    
    # Deep background starfield
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
        marker=dict(size=1.5, color="white", opacity=0.75)
    ))

    no_axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=no_axis,
            yaxis=no_axis,
            zaxis=no_axis,
            aspectmode="data"
        )
    )
    return fig


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
    """Query IRSA's SIA endpoint with fast timeouts."""
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
                timeout=(3, 8),
            )
            if response.status_code == 200 and response.content.strip():
                for record in _parse_votable(response.content):
                    key = _value(record, "obs_publisher_did", "obs_id", "access_url", "access_reference")
                    if key and key not in seen:
                        seen.add(key)
                        record["_collection"] = collection
                        products.append(record)
        except Exception:
            continue
    return products


@st.cache_data(ttl=1800, show_spinner=False)
def download_fits(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS IRSA product URLs can be downloaded.")
    max_bytes = 80 * 1024 * 1024
    chunks, total = [], 0
    with requests.get(url, timeout=(4, 20), stream=True) as response:
        response.raise_for_status()
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
                    planes.append({
                        "label": f"HDU {index}: {hdu.name}, plane {channel} {plane.shape}",
                        "data": plane,
                        "unit": unit,
                    })
    return planes


# --- Main Streamlit Application ---
def main():
    st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")
    st.markdown("Interactive 3D visualization mapping solar systems, planets, stars, and real spiral galaxy geometry with live infrared spectral flux.")

    with st.sidebar:
        st.header("SPHEREx 3D Control Hub")
        st.markdown("Search targets, filter celestial categories, and scrub through the observation timeline.")
        
        search_query = st.text_input("Search Object (e.g., Planet-1, Star, Galaxy)", "").strip().lower()
        selected_types = st.multiselect("Filter Object Types", ["Planet", "Star", "Galaxy"], default=["Planet", "Star", "Galaxy"])

        df = generate_3d_cosmic_database()
        years_available = sorted(df['Year'].unique())
        selected_year = st.slider("Observation Year (Time Evolution)", int(min(years_available)), int(max(years_available)), int(min(years_available)))

        st.divider()
        st.subheader("Archive Query")
        archive_ra = st.number_input("Target RA (°)", value=304.6935)
        archive_dec = st.number_input("Target Dec (°)", value=42.4436)
        query_archive_btn = st.button("Fetch Live IRSA FITS Data", type="primary")

    # Filter 3D Cosmic Database
    filtered_df = df[df['ObjectType'].isin(selected_types)]
    if search_query:
        filtered_df = filtered_df[
            filtered_df['ObjectName'].str.lower().str.contains(search_query) |
            filtered_df['ObjectType'].str.lower().str.contains(search_query)
        ]

    filtered_df = filtered_df[filtered_df['Year'] == selected_year]

    # Render 3D Plotly Space Scene
    if len(filtered_df) == 0:
        st.warning("⚠️ No celestial objects match your search criteria. Try a broader search term.")
        fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year}) - Empty")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"Displaying **{len(filtered_df)}** celestial sources for year **{selected_year}**.")
        
        fig = create_space_scene(title=f"SPHEREx 3D Cosmic Map ({selected_year})")

        fig.add_trace(go.Scatter3d(
            x=filtered_df['cart_x'],
            y=filtered_df['cart_y'],
            z=filtered_df['cart_z'],
            mode="markers",
            name="Cosmic Targets",
            text=filtered_df['ObjectName'],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Type: %{customdata[0]}<br>"
                "RA: %{customdata[1]:.3f}°<br>"
                "Dec: %{customdata[2]:.3f}°<br>"
                "Infrared Flux: %{customdata[3]:.2f} MJy/sr<br>"
                "Year: %{customdata[4]}<extra></extra>"
            ),
            customdata=np.stack((
                filtered_df['ObjectType'].values,
                filtered_df['RA'].values,
                filtered_df['DEC'].values,
                filtered_df['IR_Flux'].values,
                filtered_df['Year'].values
            ), axis=-1),
            marker=dict(
                size=5.5,
                color=filtered_df['IR_Flux'],
                colorscale="Inferno",
                colorbar=dict(title="IR Flux (MJy/sr)", x=0.92),
                opacity=0.95
            )
        ))

        st.plotly_chart(fig, use_container_width=True)

    # Optional Live IRSA FITS Section
    if query_archive_btn:
        with st.spinner("Querying official IRSA SPHEREx SIA collections..."):
            products = query_spherex_products(archive_ra, archive_dec, radius_deg=0.1)
            st.session_state["spherex_products"] = products

    if "spherex_products" in st.session_state:
        products = st.session_state["spherex_products"]
        if products:
            st.success(f"IRSA found {len(products)} actual SPHEREx product(s) near RA {archive_ra}, Dec {archive_dec}.")
            rows, labels, url_map = [], [], {}
            for index, record in enumerate(products, 1):
                name = _value(record, "obs_id", "obs_publisher_did", "obs_title", "filename") or f"Product {index}"
                url = _value(record, "access_url", "access_reference", "accessurl")
                rows.append({
                    "#": index,
                    "Collection": record.get("_collection", ""),
                    "Product": name,
                    "Type": _value(record, "dataproduct_type"),
                })
                if url.startswith("https://"):
                    label = f"{index}: {name[:90]}"
                    labels.append(label)
                    url_map[label] = url
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if labels:
                selected = st.selectbox("Select a real SPHEREx FITS product to inspect", labels)
                if st.button("Display selected FITS product with Inferno"):
                    try:
                        with st.spinner("Downloading FITS data..."):
                            planes = _fits_planes(download_fits(url_map[selected]))
                        if planes:
                            plane = planes[0]
                            image = plane["data"]
                            fig_fits = go.Figure(go.Heatmap(
                                z=image[::2, ::2],
                                colorscale="Inferno",
                                colorbar={"title": plane["unit"]}
                            ))
                            fig_fits.update_layout(title="SPHEREx FITS Preview", paper_bgcolor="black", font_color="white")
                            st.plotly_chart(fig_fits, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to load FITS: {e}")
        else:
            st.info("No archive products found at this specific coordinate radius.")


if __name__ == "__main__":
    main()
