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




def render_spherex_table(fig, filepath, ra_col="RA", dec_col="DEC", z_col="Z", ir_col=None, max_rows=25000):
   if filepath.endswith((".fits", ".fit")):
       
        max_retries = 3
        spectral_image_url = "YOUR_TARGET_SPHEREx_FITS_URL" # ADD FUNCTION HERE!
        
        for attempt in range(max_retries):
            try:
                with fits.open(spectral_image_url, memmap=False) as hdulist:
                    for hdu in hdulist:
                        hdu.data
                break
            except (TimeoutError, urllib.error.HTTPError, http.client.IncompleteRead):
                if attempt == max_retries - 1:
                    raise
                time.sleep(10 * (attempt + 1))

    
    """if filepath.endswith((".fits", ".fit")):
       df = Table.read(filepath).to_pandas()
   else:
       df = pd.read_csv(filepath)
"""

   cols = [ra_col, dec_col, z_col] + ([ir_col] if ir_col else [])
   df = df[cols].dropna().head(max_rows)


   distances = cosmo.comoving_distance(df[z_col].values)
   coords = SkyCoord(
       ra=df[ra_col].values * u.deg,
       dec=df[dec_col].values * u.deg,
       distance=distances
   )


   color_values = df[ir_col] if ir_col else df[z_col]
   color_label = ir_col if ir_col else "Redshift (z)"


   fig.add_trace(go.Scatter3d(
       x=coords.cartesian.x.value,
       y=coords.cartesian.y.value,
       z=coords.cartesian.z.value,
       mode="markers",
       name="SPHEREx Infrared Sources",
       hovertemplate=(
           f"<b>SPHEREx Object</b><br>"
           f"RA: %{{customdata:.3f}}°<br>"
           f"Dec: %{{customdata:.3f}}°<br>"
           f"{color_label}: %{{customdata:.3f}}<extra></extra>"
       ),
       customdata=df[[ra_col, dec_col]].assign(val=color_values).values,
       marker=dict(
           size=2.5,
           color=color_values,
           colorscale="Inferno",
           colorbar=dict(title=color_label, x=0.92),
           opacity=0.85
       )
   ))
   return fig


if __name__ == "__main__":
   fig = create_space_scene(title="SPHEREx 3D Infrared Sky Map", bg_radius=5000)
   fig = render_spherex_table(
       fig,
       filepath="spherex_data.fits",
       ra_col="RA",
       dec_col="DEC",
       z_col="Z",
       ir_col="FLUX_BAND_1"
   )
   fig.show()


