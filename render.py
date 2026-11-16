from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Solar System Sky Map", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
GAIA_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# Approximate JPL secular orbital elements for 1800-2050. These are used only
# for an immediate local preview; the app labels them as estimates, not live data.
# name, Horizons ID, a(AU), e, inclination, mean longitude, longitude perihelion,
# longitude ascending node, color, marker size
PLANETS = [
    ("Mercury", "199", .387099, .205636, 7.005, 252.251, 77.456, 48.331, "#b8b8b8", 4),
    ("Venus", "299", .723332, .006772, 3.395, 181.980, 131.602, 76.680, "#f5c46b", 6),
    ("Earth", "399", 1.000000, .016708, 0.000, 100.464, 102.937, 0.000, "#4da6ff", 7),
    ("Mars", "499", 1.523679, .093400, 1.849, 355.453, 336.041, 49.558, "#ff7043", 5),
    ("Jupiter", "599", 5.202600, .048498, 1.303, 34.404, 14.753, 100.464, "#e8b17a", 11),
    ("Saturn", "699", 9.554910, .055508, 2.489, 49.944, 92.432, 113.665, "#f2dc8c", 9),
    ("Uranus", "799", 19.218400, .046295, .773, 313.232, 170.964, 74.006, "#83e3e3", 8),
    ("Neptune", "899", 30.110400, .008988, 1.770, 304.880, 44.971, 131.784, "#577dff", 8),
]


def _kepler_position(body, day_offset: float = 0.0) -> tuple[float, float, float]:
    """Fast approximate heliocentric position, calculated locally with no network."""
    name, _, a, eccentricity, inclination, mean_longitude, perihelion, node = body[:8]
    periods_days = {
        "Mercury": 87.969, "Venus": 224.701, "Earth": 365.256,
        "Mars": 686.980, "Jupiter": 4332.59, "Saturn": 10759.2,
        "Uranus": 30688.5, "Neptune": 60182.0,
    }
    mean_anomaly = np.radians((mean_longitude - perihelion) % 360)
    mean_anomaly += 2 * np.pi * day_offset / periods_days[name]
    mean_anomaly = (mean_anomaly + np.pi) % (2 * np.pi) - np.pi
    eccentric_anomaly = mean_anomaly
    for _ in range(6):
        eccentric_anomaly -= (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly) / (1 - eccentricity * np.cos(eccentric_anomaly))
    x_orbit = a * (np.cos(eccentric_anomaly) - eccentricity)
    y_orbit = a * np.sqrt(1 - eccentricity**2) * np.sin(eccentric_anomaly)
    arg_perihelion = np.radians(perihelion - node)
    inc, asc_node = np.radians(inclination), np.radians(node)
    x = x_orbit * (np.cos(asc_node) * np.cos(arg_perihelion) - np.sin(asc_node) * np.sin(arg_perihelion) * np.cos(inc)) + y_orbit * (-np.cos(asc_node) * np.sin(arg_perihelion) - np.sin(asc_node) * np.cos(arg_perihelion) * np.cos(inc))
    y = x_orbit * (np.sin(asc_node) * np.cos(arg_perihelion) + np.cos(asc_node) * np.sin(arg_perihelion) * np.cos(inc)) + y_orbit * (-np.sin(asc_node) * np.sin(arg_perihelion) + np.cos(asc_node) * np.cos(arg_perihelion) * np.cos(inc))
    z = x_orbit * np.sin(arg_perihelion) * np.sin(inc) + y_orbit * np.cos(arg_perihelion) * np.sin(inc)
    return float(x), float(y), float(z)


def _horizons_vector(body_id: str, day: str) -> tuple[float, float, float]:
    params = {
        "format": "json", "COMMAND": f"'{body_id}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@10'",
        "REF_PLANE": "'FRAME'", "VEC_TABLE": "'2'", "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'", "START_TIME": f"'{day} 00:00'",
        "STOP_TIME": f"'{day} 00:01'", "STEP_SIZE": "'1 d'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(2, 5))
    response.raise_for_status()
    text = response.json().get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("Horizons response contained no ephemeris data")
    for row in csv.reader(io.StringIO(text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip())):
        if len(row) >= 5:
            try:
                return float(row[2]), float(row[3]), float(row[4])
            except ValueError:
                continue
    raise ValueError("Could not parse Horizons vector")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_horizons_positions(day: str) -> tuple[dict, list[str]]:
    positions, errors = {}, []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_horizons_vector, body[1], day): body[0] for body in PLANETS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                positions[name] = future.result()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    return positions, errors


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia_stars(limit: int) -> tuple[list[dict], str | None]:
    query = (f"SELECT TOP {int(limit)} ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source "
             "WHERE phot_g_mean_mag < 6.5 AND ra IS NOT NULL AND dec IS NOT NULL "
             "ORDER BY phot_g_mean_mag ASC")
    try:
        response = requests.get(GAIA_URL, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}, timeout=(2, 5))
        response.raise_for_status()
        rows = csv.DictReader(io.StringIO(response.text))
        stars = [{"ra": float(r["ra"]), "dec": float(r["dec"]), "mag": float(r["phot_g_mean_mag"])}
                 for r in rows if r.get("ra") and r.get("dec") and r.get("phot_g_mean_mag")]
        return stars, None
    except Exception as exc:
        return [], str(exc)


def make_figure(day: date, positions: dict, use_horizons: bool, selected: str,
                trail_days: int, stars: list[dict]):
    fig = go.Figure()
    # Actual Gaia positions are only queried after the user asks for them.
    if stars:
        ra = np.radians([s["ra"] for s in stars])
        dec = np.radians([s["dec"] for s in stars])
        mag = np.asarray([s["mag"] for s in stars])
        radius = 45
        fig.add_trace(go.Scatter3d(x=radius*np.cos(dec)*np.cos(ra), y=radius*np.cos(dec)*np.sin(ra), z=radius*np.sin(dec),
            mode="markers", name="Gaia DR3 stars", hoverinfo="skip",
            marker=dict(size=np.clip(4.5-mag*.45, 1, 3.5), color="#f5f2e8", opacity=.7)))

    day_offset = (day - date(2000, 1, 1)).days
    for body in PLANETS:
        name, body_id, color, size = body[0], body[1], body[8], body[9]
        x, y, z = positions.get(name, _kepler_position(body, day_offset)) if use_horizons else _kepler_position(body, day_offset)
        trail_offsets = np.linspace(-trail_days // 2, trail_days // 2, 50)
        trail = np.array([_kepler_position(body, day_offset + float(offset)) for offset in trail_offsets])
        fig.add_trace(go.Scatter3d(x=trail[:, 0], y=trail[:, 1], z=trail[:, 2], mode="lines",
            line=dict(color=color, width=2), name=f"{name} orbit", showlegend=False, hoverinfo="skip"))
        distance = float(np.sqrt(x*x+y*y+z*z))
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers+text", name=name,
            text=[name], textposition="top center", marker=dict(size=size+(5 if name == selected else 0), color=color),
            customdata=[[body_id, distance]], hovertemplate="<b>%{text}</b><br>Horizons ID %{customdata[0]}<br>Heliocentric distance %{customdata[1]:.4f} AU<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", name="Sun", text=["Sun"],
        textposition="top center", marker=dict(size=17, color="#ffd54f", line=dict(color="white", width=1)),
        hovertemplate="Sun · heliocentric origin<extra></extra>"))
    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(title=f"Solar System · {day.isoformat()}", paper_bgcolor="#02030a", plot_bgcolor="#02030a",
        font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data", camera=dict(eye=dict(x=1.5,y=1.5,z=1.1))))
    return fig


def main():
    st.title("Solar System & Infrared Archive")
    st.caption("The page renders immediately. Network requests run only after you press their buttons.")
    with st.sidebar:
        st.header("View controls")
        chosen_day = st.date_input("Date (UTC)", value=date.today())
        planet = st.selectbox("Highlight planet", ["None"] + [p[0] for p in PLANETS])
        trail_days = st.select_slider("Orbit trail (days)", options=[30, 90, 180], value=90)
        fetch_live = st.button("Fetch precise positions from JPL Horizons", type="primary")
        fetch_stars = st.button("Load real Gaia DR3 star background")
        st.caption("Initial positions are fast local Kepler estimates. Horizons and Gaia are opt-in so a slow archive cannot block page startup.")

    date_text = chosen_day.isoformat()
    if fetch_live:
        with st.spinner("Requesting JPL Horizons (bounded network timeout)…"):
            live_positions, errors = fetch_horizons_positions(date_text)
        st.session_state["horizons_date"] = date_text
        st.session_state["horizons_positions"] = live_positions
        st.session_state["horizons_errors"] = errors
    positions = st.session_state.get("horizons_positions", {})
    use_horizons = st.session_state.get("horizons_date") == date_text and bool(positions)
    if st.session_state.get("horizons_date") == date_text and st.session_state.get("horizons_errors"):
        st.warning("Some Horizons calls failed: " + "; ".join(st.session_state["horizons_errors"]))

    if fetch_stars:
        with st.spinner("Loading Gaia DR3 stars (bounded network timeout)…"):
            stars, error = fetch_gaia_stars(1500)
        st.session_state["gaia_stars"] = stars
        st.session_state["gaia_error"] = error
    stars = st.session_state.get("gaia_stars", [])
    if st.session_state.get("gaia_error"):
        st.warning(f"Gaia is temporarily unavailable: {st.session_state['gaia_error']}")

    st.plotly_chart(make_figure(chosen_day, positions, use_horizons, planet, int(trail_days), stars),
                    use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    if use_horizons:
        st.success(f"Showing JPL Horizons heliocentric vectors for {date_text}.")
    else:
        st.info("Showing approximate orbital estimates, not live ephemerides. Press the JPL Horizons button for real positions.")
    st.caption("Planet positions are heliocentric AU coordinates; planet sizes and orbit trails are not to scale. Gaia background stars are actual catalog directions on a compressed shell.")

    st.subheader("Actual SPHEREx infrared observations")
    st.write("Infrared measurements are archive products, not a single live value available for every planet. Search IRSA MOST for SPHEREx observations of moving Solar System objects; a result is shown only when the archive has matching data.")
    target = st.text_input("Planet or Solar System object", value=planet if planet != "None" else "Mars")
    st.link_button("Open NASA/IPAC IRSA MOST", "https://irsa.ipac.caltech.edu/applications/MOST/")
    st.link_button("Open SPHEREx archive and data access", "https://irsa.ipac.caltech.edu/Missions/spherex.html")
    st.caption("These official archive links are non-blocking. Use the target name above in MOST; no infrared value is invented when a matching observation is absent.")


if __name__ == "__main__":
    main()
