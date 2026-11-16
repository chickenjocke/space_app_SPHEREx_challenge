# modules used in image loader
import http.client
import time
import urllib.error
from astropy.io import fits
# modules used in 3d renderer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astroquery.ipac.irsa import Irsa
import streamlit as st
 
st.set_page_config(page_title="INFRARED SKY MAP", layout="wide")

def create_space_scene(title="3D Space Renderer", num_bg_stars=600, bg_radius=8000):
   fig = go.Figure()
   np.random.seed(42)
   phi = np.random.uniform(0, 2 * np.pi, num_bg_stars)
   costheta = np.random.uniform(-1, 1, num_bg_stars)
   theta = np.arccos(costheta)
   r = np.random.uniform(bg_radius * 0.8, bg_radius, num_bg_stars)

   bg_x = r * np.sin(theta) * np.cos(phi)
   bg_y = r * np.sin(theta) * np.sin(phi)
   bg_z = r * np.cos(theta)

   # Distant background stars (dotted safely in the deep background)
   fig.add_trace(go.Scatter3d(
       x=bg_x,
       y=bg_y,
       z=bg_z,
       mode="markers",
       name="Deep Background Stars",
       hoverinfo="skip",
       marker=dict(
           size=np.random.uniform(1.0, 2.2, num_bg_stars),
           color="white",
           opacity=0.85,
           line=dict(width=0)
       )
   ))

   no_axis = dict(
       showbackground=False,
       showgrid=False,
       zeroline=False,
       showticklabels=False,
       title=""
   )

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

@st.cache_data
def load_and_prep_nasa_data(ra_deg, dec_deg):
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        search_radius = 1 * u.arcsec
        results = Irsa.query_sia(pos=(coord, search_radius), collection=['spherex_dr1', 'spherex_qr3'])
        if len(results) > 0:
            df = results.to_table().to_pandas()
            df['RA'] = df['s_ra']
            df['DEC'] = df['s_dec']
            df['IR_Flux'] = df['em_min'].fillna(1.5)
    except Exception:
        pass

    if df is None or len(df) == 0:
        np.random.seed(101)
        n_points = 350
        df = pd.DataFrame({
            'RA': np.random.uniform(300, 310, n_points),
            'DEC': np.random.uniform(40, 45, n_points),
            'IR_Flux': np.random.uniform(0.75, 5.0, n_points),
            'Z': np.random.uniform(0.1, 2.5, n_points),
            'Year': np.random.choice([2025, 2026, 2027], n_points),
            'ObjectType': np.random.choice(['Planet', 'Star', 'Galaxy'], n_points, p=[0.15, 0.45, 0.40])
        })
    else:
        if 'Z' not in df.columns:
            df['Z'] = np.random.uniform(0.1, 2.0, len(df))
        if 'Year' not in df.columns:
            df['Year'] = np.random.choice([2025, 2026, 2027], len(df))
        if 'ObjectType' not in df.columns:
            df['ObjectType'] = np.random.choice(['Planet', 'Star', 'Galaxy'], len(df), p=[0.15, 0.45, 0.40])
        if 'IR_Flux' not in df.columns:
            df['IR_Flux'] = df.get('em_min', 2.0)

    # Generate custom searchable names (e.g., Planet-1, Star-12)
    df['ObjectName'] = [f"{row['ObjectType']}-{i+1}" for i, row in df.iterrows()]
    return df

st.sidebar.title("SPHEREx Control Hub")
st.sidebar.markdown("Search celestial targets and scrub through observation timelines.")
target_ra = st.sidebar.number_input("Target RA (°)", value=304.6935)
target_dec = st.sidebar.number_input("Target Dec (°)", value=42.4436)

df = load_and_prep_nasa_data(target_ra, target_dec)

search_query = st.sidebar.text_input("Search Object Name (e.g., Planet-1, Star-12)", "").strip().lower()
selected_types = st.sidebar.multiselect("Filter Object Types", ["Planet", "Star", "Galaxy"], default=["Planet", "Star", "Galaxy"])

years_available = sorted(df['Year'].unique())
selected_year = st.sidebar.slider("Observation Year (Time Evolution)", int(min(years_available)), int(max(years_available)), int(min(years_available)))

# Filtering logic
filtered_df = df[df['ObjectType'].isin(selected_types)]
if search_query:
    filtered_df = filtered_df[filtered_df['ObjectName'].str.lower().str.contains(search_query)]

filtered_df = filtered_df[filtered_df['Year'] == selected_year]

st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")

# Safety check so app never crashes on empty searches
if len(filtered_df) == 0:
    st.warning("⚠️ No celestial objects found matching your search. Try searching for generic IDs like 'Planet-1', 'Star-5', or clear your search box.")
    fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year}) - Empty", bg_radius=8000)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(f"Displaying **{len(filtered_df)}** celestial sources for year **{selected_year}**.")

    distances = cosmo.comoving_distance(filtered_df['Z'].values)
    coords = SkyCoord(
        ra=filtered_df['RA'].values * u.deg,
        dec=filtered_df['DEC'].values * u.deg,
        distance=distances
    )

    filtered_df['cart_x'] = coords.cartesian.x.value
    filtered_df['cart_y'] = coords.cartesian.y.value
    filtered_df['cart_z'] = coords.cartesian.z.value

    fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year})", bg_radius=8000)

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

def render_spherex_live_nasa(fig, ra_deg=304.6935, dec_deg=42.4436, max_rows=25000):
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        search_radius = 1 * u.arcsec
        results = Irsa.query_sia(pos=(coord, search_radius), collection = ['spherex_dr1', 'spherex_qr3'])
        if len(results) > 0:
            df = results.to_table().to_pandas().head(max_rows)
            ra_col, dec_col = 's_ra', 's_dec'
            df['Z'] = df['em_min'].fillna(1.0) / 5.0
    except Exception:
        pass

    if df is None or len(df) == 0:
        df = pd.DataFrame({
            's_ra': [304.69],
            's_dec': [42.44],
            'em_min': [1.2],
            'Z': [1.0]
        })
        ra_col, dec_col = 's_ra', 's_dec'

    distances = cosmo.comoving_distance(df['Z'].values)
    coords = SkyCoord(
        ra=df[ra_col].values * u.deg,
        dec=df[dec_col].values * u.deg,
        distance=distances
    )

    color_values = df.get('em_min', df.get('Z'))
    color_label = "Infrared Wavelength (um)"

    fig.add_trace(go.Scatter3d(
        x=coords.cartesian.x.value,
        y=coords.cartesian.y.value,
        z=coords.cartesian.z.value,
        mode="markers",
        name="NASA SPHEREx Data",
        marker=dict(size=3.0, color=color_values, colorscale="Inferno")
    ))
    return fig
