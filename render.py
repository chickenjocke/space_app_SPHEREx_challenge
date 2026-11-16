import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import streamlit as st

st.set_page_config(page_title="INFRARED SKY MAP", layout="wide")


@st.cache_data

def load_demo_data():
    """Create a small deterministic offline dataset so the app never waits on a remote archive."""
    rng = np.random.default_rng(101)
    n_points = 350
    df = pd.DataFrame(
        {
            "RA": rng.uniform(300, 310, n_points),
            "DEC": rng.uniform(40, 45, n_points),
            "IR_Flux": rng.uniform(0.75, 5.0, n_points),
            "Z": rng.uniform(0.1, 2.5, n_points),
            "Year": rng.choice([2025, 2026, 2027], n_points),
            "ObjectType": rng.choice(
                ["Planet", "Star", "Galaxy"], n_points, p=[0.15, 0.45, 0.40]
            ),
        }
    )
    df["ObjectName"] = [f"{kind}-{i + 1}" for i, kind in enumerate(df["ObjectType"])]
    return df


def create_space_scene(title, num_bg_stars=250, bg_radius=8000):
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, num_bg_stars)
    costheta = rng.uniform(-1, 1, num_bg_stars)
    theta = np.arccos(costheta)
    radius = rng.uniform(bg_radius * 0.8, bg_radius, num_bg_stars)

    fig = go.Figure(
        go.Scatter3d(
            x=radius * np.sin(theta) * np.cos(phi),
            y=radius * np.sin(theta) * np.sin(phi),
            z=radius * np.cos(theta),
            mode="markers",
            name="Background stars",
            hoverinfo="skip",
            marker=dict(size=1.5, color="white", opacity=0.8),
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


def main():
    st.sidebar.title("SPHEREx Control Hub")
    st.sidebar.caption("Offline demo data loads immediately; no NASA archive request is made at startup.")
    target_ra = st.sidebar.number_input("Target RA (°)", value=304.6935)
    target_dec = st.sidebar.number_input("Target Dec (°)", value=42.4436)

    # Keep controls meaningful while keeping this offline demo responsive.
    df = load_demo_data().copy()
    ra_offset = target_ra - 304.6935
    dec_offset = target_dec - 42.4436
    df["RA"] = (df["RA"] + ra_offset) % 360
    df["DEC"] = (df["DEC"] + dec_offset).clip(-90, 90)

    search_query = st.sidebar.text_input(
        "Search Object Name (e.g., Planet-1, Star-12)", ""
    ).strip().lower()
    object_types = ["Planet", "Star", "Galaxy"]
    selected_types = st.sidebar.multiselect(
        "Filter Object Types", object_types, default=object_types
    )
    years_available = sorted(df["Year"].unique())
    selected_year = st.sidebar.slider(
        "Observation Year (Time Evolution)",
        int(min(years_available)),
        int(max(years_available)),
    )

    filtered_df = df[df["ObjectType"].isin(selected_types)]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["ObjectName"].str.lower().str.contains(search_query, regex=False)
        ]
    filtered_df = filtered_df[filtered_df["Year"] == selected_year].copy()

    st.title("SPHEREx 3D Time-Evolving Infrared Sky Map")
    st.info("Showing a deterministic offline demo dataset, not live NASA archive results.")

    if filtered_df.empty:
        st.warning("No objects match these filters. Try another year or clear your search.")
        st.plotly_chart(
            create_space_scene(f"Sky Map ({selected_year}) - Empty"),
            use_container_width=True,
        )
        return

    st.markdown(f"Displaying **{len(filtered_df)}** demo sources for **{selected_year}**.")
    distances = cosmo.comoving_distance(filtered_df["Z"].to_numpy())
    coords = SkyCoord(
        ra=filtered_df["RA"].to_numpy() * u.deg,
        dec=filtered_df["DEC"].to_numpy() * u.deg,
        distance=distances,
    )
    filtered_df["cart_x"] = coords.cartesian.x.value
    filtered_df["cart_y"] = coords.cartesian.y.value
    filtered_df["cart_z"] = coords.cartesian.z.value

    fig = create_space_scene(f"Sky Map ({selected_year})")
    fig.add_trace(
        go.Scatter3d(
            x=filtered_df["cart_x"],
            y=filtered_df["cart_y"],
            z=filtered_df["cart_z"],
            mode="markers",
            name="Demo targets",
            text=filtered_df["ObjectName"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "RA: %{customdata[0]:.3f}°<br>"
                "Dec: %{customdata[1]:.3f}°<br>"
                "Infrared Flux: %{customdata[2]:.2f} MJy/sr<br>"
                "Year: %{customdata[3]}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    filtered_df["RA"],
                    filtered_df["DEC"],
                    filtered_df["IR_Flux"],
                    filtered_df["Year"],
                ]
            ),
            marker=dict(
                size=4,
                color=filtered_df["IR_Flux"],
                colorscale="Inferno",
                colorbar=dict(title="IR Flux (MJy/sr)"),
                opacity=0.95,
            ),
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


if __name__ == "__main__":
    main()
