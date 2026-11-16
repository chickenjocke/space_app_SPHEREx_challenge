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

""" 
  BOILERPLATE CODE TO RENDER PLANETS(CREATED AS A BOILERPLATE BEFOREHAND;
  DOES NOT RENDER REAL PLANETS OR USE NASA DATA, ONLY TO SHOW PLANETS!
  AI USED IN CREATION OF SECTION TO CORRECTLY DESIGN PLANETS
"""

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


def render_spherex_live_nasa(fig, ra_deg=304.6935, dec_deg=42.4436, max_rows=25000):
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    search_radius = 1 * u.arsec
    print(f"Sending HTTPS request to NASA IRSA for values: RA, {ra_deg}, Dec: {dec_deg}...")
    results = Irsa.query_sia(pos=(coord, search_radius), collection = ['spherex_dr1', 'spherex_qr3'])
    if len(results) == 0:
        raise ValueError("No SPHEREx data found for these coordinates.")
      
    df = results.to_table().to_pandas().head(max_rows)
    ra_col  = 's_ra'
    dec_col = 's_deg'

    df['Z'] = df['em_min'].fillna(1.0) / 5.0
    distances = cosmo.comoving_distance(df['Z'].values)

    coords = SkyCoord(
        ra=df[ra_col].values * u.deg,
        dec=df[dec_col].values * u.deg,
        distance=distances
    )

    color_values = df['em_min'] # Infrared wavelength band indicator
    color_label = "Infrared Wavelength (um)"

    fig.add_trace(go.Scatter3d(
        x=coords.cartesian.x.value,
        y=coords.cartesian.y.value,
        z=coords.cartesian.z.value,
        mode="markers",
        name="NASA SPHEREx Live Data",
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
