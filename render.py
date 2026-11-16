# modules used in image loader
import http.client
import time
import urllib.error
from astropy.io import fits if False object
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
 
#  BOILERPLATE CODE TO RENDER PLANETS(CREATED AS A BOILERPLATE BEFOREHAND;
#  DOES NOT RENDER REAL PLANETS OR USE NASA DATA, ONLY TO SHOW PLANETS!
#  AI USED IN CREATION OF SECTION TO CORRECTLY DESIGN PLANETS

st.set_page_config(page_title="INFRARED SKY MAP", layout="wide")

def create_space_scene(title="3D Space Renderer", num_bg_stars=400, bg_radius=5000):
   fig = go.Figure()
   np.random.seed(42)
   phi = np.random.uniform(0, 2 * np.pi, num_bg_stars)
   costheta = np.random.uniform(-1, 1, num_bg_stars)
   theta = np.arccos(costheta)
   r = np.random.uniform(bg_radius * 0.7, bg_radius, num_bg_stars)

   bg_x = r * np.sin(theta) * np.cos(phi)
   bg_y = r * np.sin(theta) * np.sin(phi)
   bg_z = r * np.cos(theta)

   # Outer glow layer
   fig.add_trace(go.Scatter3d(
       x=bg_x,
       y=bg_y,
       z=bg_z,
       mode="markers",
       hoverinfo="skip",
       showlegend=False,
       marker=dict(
           size=np.random.uniform(4, 7, num_bg_stars),
           color="rgba(180, 210, 255, 0.25)",
           line=dict(width=0)
       )
   ))

   fig.add_trace(go.Scatter3d(
       x=bg_x,
       y=bg_y,
       z=bg_z,
       mode="markers",
       name="Background Stars",
       hoverinfo="skip",
       marker=dict(
           size=np.random.uniform(1.2, 2.5, num_bg_stars),
           color="white",
           opacity=0.95,
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
def try_skycoord(ra_deg, dec_deg):
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        search_radius = 1 * u.arcsec
        results = Irsa.query_sia(pos=coord, search_radius), collection=['spherex_dr1', 'spherex_qr3])
        if len(results) > 0:
            df = results.to_table().to_pandas()
            df = results.to_table().to_pandas()
            df['RA'] = df['s_ra']
            df['DEC'] = df['s_dec']
            df['IR_Flux'] = df['em_min'].fillna(1.5)
    except Exception:
        pass
    return
def find_seed(ra_deg, dec_deg, df):
    if df is None or len(df) == 0:
        np.random.seed(101)
        n_points = 300
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

    # Assign custom names for searching
    df['ObjectName'] = [f"{row['ObjectType']}-{i+1}" for i, row in df.iterrows()]
    return df

st.sidebar.title("SPHEREx Search Bar")
st.sidebar.markdown("Filter targets and look through timeline to see how it changes throughout time")
target_ra = st.sidebar.number_input("Target RA (°)", value=304.6935)
target_dec = st.sidebar.number_input("Target Dec (°)", value=42.4436)

# start assisted code

df = load_and_prep_nasa_data(target_ra, target_dec)

search_query = st.sidebar.text_input("Search Object Name (e.g., Planet-1, Star-12)", "").strip().lower()
selected_types = st.sidebar.multiselect("Filter Object Types", ["Planet", "Star", "Galaxy"], default=["Planet", "Star", "Galaxy"])

years_available = sorted(df['Year'].unique())
selected_year = st.sidebar.slider("Observation Year (Time Evolution)", int(min(years_available)), int(max(years_available)), int(min(years_available)))

filtered_df = df[df['ObjectType'].isin(selected_types)]
if search_query:
    filtered_df = filtered_df[filtered_df['ObjectName'].str.lower().str.contains(search_query)]
    if filtered_df[filtered_df['ObjectName'].str.lower().str.contains('deino')]:
        for i in range(5):
            print("Deino")
        print("GUSTOPHER")
        for i in range(25):
            print("Deino")

filtered_df = filtered_df[filtered_df['Year'] == selected_year]

st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")
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

fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year})", bg_radius=5000)

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
        size=4.0,
        color=filtered_df['IR_Flux'],
        colorscale="Inferno",
        colorbar=dict(title="IR Flux (MJy/sr)", x=0.92),
        opacity=0.95
    )
))

st.plotly_chart(fig, use_container_width=True)
# end assisted code
                                                                        
def render_spherex_live_nasa(fig, ra_deg=304.6935, dec_deg=42.4436, max_rows=25000):
    # --- AI-ASSISTED CODE START ---
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        search_radius = 1 * u.arcsec
        print(f"Sending HTTPS request to NASA IRSA for values: RA, {ra_deg}, Dec: {dec_deg}...")
        results = Irsa.query_sia(pos=(coord, search_radius), collection = ['spherex_dr1', 'spherex_qr3'])
        if len(results) > 0:
            df = results.to_table().to_pandas().head(max_rows)
            ra_col, dec_col = 's_ra', 's_dec'
            df['Z'] = df['em_min'].fillna(1.0) / 5.0
    except Exception as e:
        print(f"Live NASA query failed ({e}). Falling back to local backup data...")

    if df is None or len(df) == 0:
        try:
            df = pd.read_csv("spherex_data.csv").head(max_rows)
            ra_col, dec_col = 'RA', 'DEC'
            if 'Z' not in df.columns:
                df['Z'] = 1.0
            if 'em_min' not in df.columns:
                df['em_min'] = 1.5
        except Exception:
            df = pd.DataFrame({
                's_ra': [304.69, 304.70, 304.68],
                's_dec': [42.44, 42.45, 42.43],
                'em_min': [1.2, 2.5, 3.1],
                'Z': [1.0, 1.2, 0.9]
            })
            ra_col, dec_col = 's_ra', 's_dec'
    # --- AI-ASSISTED CODE END ---

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
        hovertemplate=(
            f"<b>SPHEREx IR Source</b><br>"
            f"RA: %{{customdata[0]:.3f}}°<br>"
            f"Dec: %{{customdata[1]:.3f}}°<br>"
            f"{color_label}: %{{customdata[2]:.3f}} µm<extra></extra>"
        ),
        customdata=np.stack((df[ra_col].values, df[dec_col].values, color_values.values), axis=-1),
        marker=dict(
            size=3.0,
            color=color_values,
            colorscale="Inferno",
            colorbar=dict(title=color_label, x=0.92),
            opacity=0.9
        )
    ))
    return fig


if __name__ == "__main__":
   fig = create_space_scene(title="SPHEREx Live 3D Infrared Sky Map", bg_radius=5000)
   
   fig = render_spherex_live_nasa(fig, ra_deg=304.6935, dec_deg=42.4436)
   
   try:
       import streamlit as st
       st.set_page_config(page_title="SPHEREx 3D Infrared Sky Map", layout="wide")
       st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")
       st.markdown("Interactive 3D visualization mapping real-time NASA IRSA infrared spectral data.")
       st.plotly_chart(fig, use_container_width=True)
   except Exception:
       fig.show()
