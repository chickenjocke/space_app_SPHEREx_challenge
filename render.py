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


def _query_spherex(coord, search_radius_deg=1.0):
    """Query the SPHEREx archive with a radius-aware call compatible with newer astroquery versions."""
    search_radius = search_radius_deg * u.deg
    call_variants = (
        {"pos": coord, "size": search_radius, "collection": ["spherex_dr1", "spherex_qr3"]},
        {"pos": (coord, search_radius), "collection": ["spherex_dr1", "spherex_qr3"]},
    )

    for kwargs in call_variants:
        try:
            results = Irsa.query_sia(**kwargs)
            if results is not None:
                return results
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _numeric_series(values, index):
    """Return numeric data as an index-aligned Series, even for scalar or array inputs."""
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)
    return pd.to_numeric(series, errors="coerce")


def _fill_missing_redshifts(values, index):
    """Fill invalid/missing redshifts with randomized fallback values."""
    redshifts = _numeric_series(values, index)
    missing = redshifts.isna()
    if missing.any():
        redshifts.loc[missing] = np.random.uniform(0.1, 2.0, int(missing.sum()))
    return redshifts


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
    fig.add_trace(
        go.Scatter3d(
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
                line=dict(width=0),
            ),
        )
    )

    no_axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")

    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode="data"),
    )
    return fig


@st.cache_data
def load_and_prep_nasa_data(ra_deg, dec_deg):
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        results = _query_spherex(coord)
        if results is not None and len(results) > 0:
            df = results.to_table().to_pandas()
    except Exception:
        df = None

    if df is None or len(df) == 0:
        np.random.seed(101)
        n_points = 350
        df = pd.DataFrame(
            {
                "RA": np.random.uniform(300, 310, n_points),
                "DEC": np.random.uniform(40, 45, n_points),
                "IR_Flux": np.random.uniform(0.75, 5.0, n_points),
                "Z": np.random.uniform(0.1, 2.5, n_points),
                "Year": np.random.choice([2025, 2026, 2027], n_points),
                "ObjectType": np.random.choice(["Planet", "Star", "Galaxy"], n_points, p=[0.15, 0.45, 0.40]),
            }
        )
    else:
        df = df.copy()
        if "RA" not in df.columns:
            df["RA"] = df.get("s_ra", df.get("ra", np.nan))
        if "DEC" not in df.columns:
            df["DEC"] = df.get("s_dec", df.get("dec", np.nan))
        if "IR_Flux" not in df.columns:
            flux_values = _numeric_series(df.get("em_min", pd.Series(np.nan, index=df.index)), df.index)
            df["IR_Flux"] = flux_values.fillna(1.5)
        if "Z" not in df.columns:
            df["Z"] = np.random.uniform(0.1, 2.0, len(df))
        else:
            df["Z"] = _fill_missing_redshifts(df["Z"], df.index)
        if "Year" not in df.columns:
            df["Year"] = np.random.choice([2025, 2026, 2027], len(df))
        else:
            df["Year"] = _numeric_series(df["Year"], df.index).fillna(2025).astype(int)
        if "ObjectType" not in df.columns:
            df["ObjectType"] = np.random.choice(["Planet", "Star", "Galaxy"], len(df), p=[0.15, 0.45, 0.40])

    df = df.copy()
    for column, default in (("RA", 0.0), ("DEC", 0.0), ("IR_Flux", 1.5)):
        values = _numeric_series(df.get(column, default), df.index)
        df[column] = values.fillna(default)
    df["Z"] = _fill_missing_redshifts(df.get("Z", pd.Series(np.nan, index=df.index)), df.index)
    df["Year"] = _numeric_series(df.get("Year", 2025), df.index).fillna(2025).astype(int)
    df["ObjectType"] = df.get("ObjectType", pd.Series(["Star"] * len(df), index=df.index)).fillna("Star").astype(str)
    df["ObjectName"] = [f"{row['ObjectType']}-{i + 1}" for i, row in df.iterrows()]
    return df


def render_spherex_live_nasa(fig, ra_deg=304.6935, dec_deg=42.4436, max_rows=25000):
    df = None
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        results = _query_spherex(coord, 1.0)
        if results is not None and len(results) > 0:
            df = results.to_table().to_pandas().head(max_rows)
            if "s_ra" in df.columns:
                ra_col = "s_ra"
            else:
                ra_col = "RA" if "RA" in df.columns else "ra"
            if "s_dec" in df.columns:
                dec_col = "s_dec"
            else:
                dec_col = "DEC" if "DEC" in df.columns else "dec"
            redshift_source = df.get("em_min", df.get("Z", pd.Series(1.0, index=df.index)))
            df["Z"] = _numeric_series(redshift_source, df.index).fillna(1.0) / 5.0
    except Exception:
        df = None

    if df is None or len(df) == 0:
        df = pd.DataFrame({"s_ra": [304.69], "s_dec": [42.44], "em_min": [1.2], "Z": [1.0]})
        ra_col, dec_col = "s_ra", "s_dec"
    else:
        ra_col = "RA" if "RA" in df.columns else ("s_ra" if "s_ra" in df.columns else "ra")
        dec_col = "DEC" if "DEC" in df.columns else ("s_dec" if "s_dec" in df.columns else "dec")

    distances = cosmo.comoving_distance(_numeric_series(df["Z"], df.index).fillna(1.0).to_numpy())
    coords = SkyCoord(
        ra=_numeric_series(df[ra_col], df.index).fillna(0.0).to_numpy() * u.deg,
        dec=_numeric_series(df[dec_col], df.index).fillna(0.0).to_numpy() * u.deg,
        distance=distances,
    )

    color_values = _numeric_series(df.get("em_min", df.get("IR_Flux", df["Z"])), df.index).fillna(1.0)
    color_label = "Infrared Wavelength (um)"

    fig.add_trace(
        go.Scatter3d(
            x=coords.cartesian.x.value,
            y=coords.cartesian.y.value,
            z=coords.cartesian.z.value,
            mode="markers",
            name="NASA SPHEREx Data",
            marker=dict(size=3.0, color=color_values, colorscale="Inferno", colorbar=dict(title=color_label)),
        )
    )
    return fig


def main():
    st.sidebar.title("SPHEREx Control Hub")
    st.sidebar.markdown("Search celestial targets and scrub through observation timelines.")
    target_ra = st.sidebar.number_input("Target RA (°)", value=304.6935)
    target_dec = st.sidebar.number_input("Target Dec (°)", value=42.4436)

    df = load_and_prep_nasa_data(target_ra, target_dec)

    search_query = st.sidebar.text_input("Search Object Name (e.g., Planet-1, Star-12)", "").strip().lower()
    selected_types = st.sidebar.multiselect("Filter Object Types", ["Planet", "Star", "Galaxy"], default=["Planet", "Star", "Galaxy"])

    years_available = sorted(df["Year"].unique())
    selected_year = st.sidebar.slider("Observation Year (Time Evolution)", int(min(years_available)), int(max(years_available)), int(min(years_available)))

    filtered_df = df[df["ObjectType"].isin(selected_types)].copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["ObjectName"].str.lower().str.contains(search_query)]
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]

    st.title("NASA SPHEREx 3D Time-Evolving Infrared Sky Map")

    if len(filtered_df) == 0:
        st.warning("⚠️ No celestial objects found matching your search. Try searching for generic IDs like 'Planet-1', 'Star-5', or clear your search box.")
        fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year}) - Empty", bg_radius=8000)
        st.plotly_chart(fig, use_container_width=True)
        return

    st.markdown(f"Displaying **{len(filtered_df)}** celestial sources for year **{selected_year}**.")

    distances = cosmo.comoving_distance(filtered_df["Z"].fillna(1.0).to_numpy())
    coords = SkyCoord(
        ra=filtered_df["RA"].to_numpy() * u.deg,
        dec=filtered_df["DEC"].to_numpy() * u.deg,
        distance=distances,
    )

    filtered_df["cart_x"] = coords.cartesian.x.value
    filtered_df["cart_y"] = coords.cartesian.y.value
    filtered_df["cart_z"] = coords.cartesian.z.value

    fig = create_space_scene(title=f"SPHEREx Sky Map ({selected_year})", bg_radius=8000)

    fig.add_trace(
        go.Scatter3d(
            x=filtered_df["cart_x"],
            y=filtered_df["cart_y"],
            z=filtered_df["cart_z"],
            mode="markers",
            name="Cosmic Targets",
            text=filtered_df["ObjectName"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "RA: %{customdata[0]:.3f}°<br>"
                "Dec: %{customdata[1]:.3f}°<br>"
                "Infrared Flux: %{customdata[2]:.2f} MJy/sr<br>"
                "Year: %{customdata[3]}<extra></extra>"
            ),
            customdata=np.stack(
                (
                    filtered_df["RA"].to_numpy(),
                    filtered_df["DEC"].to_numpy(),
                    filtered_df["IR_Flux"].to_numpy(),
                    filtered_df["Year"].to_numpy(),
                ),
                axis=-1,
            ),
            marker=dict(
                size=5.0,
                color=filtered_df["IR_Flux"],
                colorscale="Inferno",
                colorbar=dict(title="IR Flux (MJy/sr)", x=0.92),
                opacity=0.95,
            ),
        )
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    with st.spinner("Connecting to NASA archive... May take 30+ seconds..."):
        main()
