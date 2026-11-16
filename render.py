from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="REAL SOLAR SYSTEM SKY MAP", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
# Horizons numeric IDs for the planetary centers. The Sun itself is drawn at the origin.
PLANETS = [
    ("Mercury", "199", "#b8b8b8", 4),
    ("Venus", "299", "#f5c46b", 6),
    ("Earth", "399", "#4da6ff", 7),
    ("Mars", "499", "#ff7043", 5),
    ("Jupiter", "599", "#e8b17a", 12),
    ("Saturn", "699", "#f2dc8c", 10),
    ("Uranus", "799", "#83e3e3", 8),
    ("Neptune", "899", "#577dff", 8),
]


def _vector_for_target(target_id: str, start: str, stop: str) -> dict:
    """Fetch one real heliocentric JPL Horizons vector, with a bounded network wait."""
    params = {
        "format": "json",
        "COMMAND": f"'{target_id}'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",
        "REF_PLANE": "'FRAME'",
        "VEC_TABLE": "'2'",
        "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "'1 d'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(3, 8))
    response.raise_for_status()
    payload = response.json()
    text = payload.get("result", "")
    if not text or "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError(payload.get("error", "JPL Horizons returned no vector data."))

    table_text = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
    rows = list(csv.reader(io.StringIO(table_text)))
    for row in rows:
        # CSV vector rows start with Julian date and calendar date, then X, Y, Z.
        if len(row) >= 5:
            try:
                return {"x": float(row[2]), "y": float(row[3]), "z": float(row[4])}
            except (TypeError, ValueError):
                continue
    raise ValueError("Could not parse the JPL Horizons vector response.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_solar_system(date_utc: str) -> tuple[list[dict], list[str]]:
    """Fetch the planets concurrently; successful results are cached for one hour."""
    start = f"{date_utc} 00:00"
    next_day = (datetime.strptime(date_utc, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    stop = f"{next_day} 00:00"
    bodies = [{"name": "Sun", "id": "10", "x": 0.0, "y": 0.0, "z": 0.0, "color": "#ffd54f", "size": 18}]
    errors = []
    with ThreadPoolExecutor(max_workers=len(PLANETS)) as pool:
        tasks = {
            pool.submit(_vector_for_target, body_id, start, stop): (name, body_id, color, size)
            for name, body_id, color, size in PLANETS
        }
        for task in as_completed(tasks):
            name, body_id, color, size = tasks[task]
            try:
                vector = task.result()
                bodies.append({"name": name, "id": body_id, **vector, "color": color, "size": size})
            except Exception as exc:
                errors.append(f"{name} (Horizons ID {body_id}): {exc}")
    return bodies, errors


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia_background(limit: int = 2500) -> tuple[list[dict], str | None]:
    """Load real all-sky Gaia DR3 star positions for a sparse, bright night-sky layer."""
    query = (
        "SELECT TOP " + str(int(limit)) + " ra,dec,phot_g_mean_mag "
        "FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 6.5 "
        "AND ra IS NOT NULL AND dec IS NOT NULL ORDER BY phot_g_mean_mag ASC"
    )
    try:
        response = requests.get(
            GAIA_TAP_URL,
            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
            timeout=(3, 8),
        )
        response.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(response.text)))
        stars = [
            {"ra": float(row["ra"]), "dec": float(row["dec"]), "mag": float(row["phot_g_mean_mag"])}
            for row in rows
            if row.get("ra") and row.get("dec") and row.get("phot_g_mean_mag")
        ]
        return stars, None
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia_star(source_id: str) -> dict:
    """Resolve one Gaia DR3 source ID to its real sky position and parallax."""
    if not source_id.isdigit():
        raise ValueError("Enter the numeric Gaia DR3 source_id for a star.")
    query = (
        "SELECT TOP 1 source_id,ra,dec,parallax,phot_g_mean_mag "
        "FROM gaiadr3.gaia_source WHERE source_id=" + source_id
    )
    response = requests.get(
        GAIA_TAP_URL,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=(3, 8),
    )
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows:
        raise ValueError("No Gaia DR3 source was found with that ID.")
    row = rows[0]
    return {
        "id": row["source_id"],
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "parallax": float(row["parallax"]) if row.get("parallax") else float("nan"),
        "mag": float(row["phot_g_mean_mag"]) if row.get("phot_g_mean_mag") else float("nan"),
    }


def add_star_sky(fig, stars: list[dict], radius: float = 80.0):
    """Plot real Gaia sky directions on a compressed shell so planets remain visible."""
    if not stars:
        return
    ra = np.radians([star["ra"] for star in stars])
    dec = np.radians([star["dec"] for star in stars])
    x = radius * np.cos(dec) * np.cos(ra)
    y = radius * np.cos(dec) * np.sin(ra)
    z = radius * np.sin(dec)
    mags = np.array([star["mag"] for star in stars])
    sizes = np.clip(5.2 - mags * 0.55, 1.2, 4.0)
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="Gaia DR3 stars (sky directions)",
            text=[f"Gaia DR3 star · G={mag:.2f}" for mag in mags],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sizes, color="#f4f3e8", opacity=0.82),
        )
    )


def make_figure(bodies: list[dict], stars: list[dict], target_star: dict | None = None):
    fig = go.Figure()
    add_star_sky(fig, stars)
    if target_star:
        ra, dec = np.radians(target_star["ra"]), np.radians(target_star["dec"])
        radius = 80.0
        parallax = target_star["parallax"]
        parallax_text = f"{parallax:.3f} mas" if np.isfinite(parallax) else "unavailable"
        fig.add_trace(
            go.Scatter3d(
                x=[radius * np.cos(dec) * np.cos(ra)],
                y=[radius * np.cos(dec) * np.sin(ra)],
                z=[radius * np.sin(dec)],
                mode="markers+text", name="Selected Gaia star",
                text=[f"Gaia {target_star['id']}"], textposition="top center",
                marker=dict(size=10, color="#ff66cc", symbol="diamond"),
                customdata=[[target_star["ra"], target_star["dec"], parallax_text, target_star["mag"]]],
                hovertemplate=("Gaia DR3 %{text}<br>RA %{customdata[0]:.5f}° · Dec %{customdata[1]:.5f}°"
                               "<br>Parallax %{customdata[2]} · G %{customdata[3]:.2f}<extra></extra>"),
            )
        )

    if bodies:
        fig.add_trace(
            go.Scatter3d(
                x=[body["x"] for body in bodies], y=[body["y"] for body in bodies],
                z=[body["z"] for body in bodies], mode="markers+text", name="JPL Horizons bodies",
                text=[body["name"] for body in bodies], textposition="top center",
                marker=dict(size=[body["size"] for body in bodies], color=[body["color"] for body in bodies],
                            line=dict(color="white", width=0.5)),
                customdata=[[body["id"], np.sqrt(body["x"]**2 + body["y"]**2 + body["z"]**2)] for body in bodies],
                hovertemplate=("<b>%{text}</b><br>Horizons ID %{customdata[0]}<br>"
                               "Heliocentric distance %{customdata[1]:.4f} AU<extra></extra>"),
            )
        )

    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(
        title="Real Solar System · JPL Horizons + Gaia DR3",
        paper_bgcolor="#02030a", plot_bgcolor="#02030a", font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(bgcolor="rgba(0,0,0,0.35)"),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
    )
    return fig


def main():
    st.title("Real Solar System & Night Sky")
    st.caption("Planet positions: NASA/JPL Horizons · Background and star lookup: ESA Gaia DR3")
    st.sidebar.header("Explore")
    mode = st.sidebar.radio("View", ["Solar system", "Look up a star (Gaia DR3)"])
    star_id = ""
    if mode == "Look up a star (Gaia DR3)":
        st.sidebar.markdown("Enter a **Gaia DR3 source_id**. The complete Solar System stays in view.")
        with st.sidebar.form("star_lookup"):
            star_id = st.text_input("Gaia DR3 source_id", placeholder="For example: 5853498713190525696")
            lookup = st.form_submit_button("Find star")
        if lookup and star_id.strip():
            st.session_state["selected_gaia_id"] = star_id.strip()

    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    st.caption(f"Ephemeris date: {date_utc} UTC · Planet markers are enlarged; the star shell is compressed, not to scale.")
    with st.spinner("Fetching today's planetary positions from JPL Horizons and the Gaia star catalog…"):
        bodies, body_errors = fetch_solar_system(date_utc)
        stars, star_error = fetch_gaia_background()

    selected_star = None
    selected_id = st.session_state.get("selected_gaia_id")
    if mode == "Look up a star (Gaia DR3)" and selected_id:
        try:
            with st.spinner("Looking up the Gaia DR3 star…"):
                selected_star = fetch_gaia_star(selected_id)
        except Exception as exc:
            st.error(f"Could not load that Gaia star: {exc}")

    if not bodies:
        st.error("JPL Horizons could not be reached, so current planet positions are unavailable. Check network access and retry.")
        if body_errors:
            with st.expander("Horizons request details"):
                st.write("\n".join(body_errors))
    else:
        st.success(f"Loaded {len(bodies)} Solar System bodies from JPL Horizons.")
    if body_errors:
        st.warning("Some planet requests failed: " + "; ".join(body_errors))
    if star_error:
        st.warning(f"Gaia's real star catalog is temporarily unavailable: {star_error}. Planet positions are still shown.")
    elif stars:
        st.caption(f"Background contains {len(stars):,} real, bright Gaia DR3 star positions (the all-sky directions are shown on a compressed shell).")

    st.plotly_chart(make_figure(bodies, stars, selected_star), use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})

    st.subheader("Infrared observations (real mission data)")
    st.info(
        "JPL Horizons provides real positions, not measured infrared spectra or fluxes. There is no single "
        "uniform real-time infrared measurement for every planet/star. To avoid inventing values, this map "
        "does not label modeled colors as observed infrared data. Search actual mission observations in the "
        "NASA Planetary Data System and NASA/IPAC Infrared Science Archive below. Those observations are "
        "instrument-, wavelength-, date-, and target-specific."
    )
    st.markdown(
        "- [NASA Planetary Data System (PDS)](https://pds.nasa.gov/) — spacecraft observations and calibrated products.\n"
        "- [NASA/IPAC Infrared Science Archive (IRSA)](https://irsa.ipac.caltech.edu/) — infrared mission archives.\n"
        "- [JPL Horizons API documentation](https://ssd.jpl.nasa.gov/api/horizons.api.html) — ephemerides used for positions.\n"
        "- [ESA Gaia Archive](https://gea.esac.esa.int/archive/) — real star positions used in the background."
    )


if __name__ == "__main__":
    main()
