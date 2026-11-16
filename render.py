from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="REAL SOLAR SYSTEM SKY MAP", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
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
PLANET_NAMES = [entry[0] for entry in PLANETS]


def _horizons_vectors(target_id: str, start: str, stop: str, step_days: int) -> list[dict]:
    """Fetch heliocentric ICRF vectors from JPL Horizons and parse its CSV ephemeris."""
    params = {
        "format": "json",
        "COMMAND": f"'{target_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",
        "REF_PLANE": "'FRAME'",
        "VEC_TABLE": "'2'",
        "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step_days} d'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(3, 10))
    response.raise_for_status()
    payload = response.json()
    text = payload.get("result", "")
    if not text or "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError(payload.get("error", "JPL Horizons returned no vector data."))

    records = []
    table = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
    for row in csv.reader(io.StringIO(table)):
        if len(row) < 5:
            continue
        try:
            records.append({"date": row[1].strip(), "x": float(row[2]), "y": float(row[3]), "z": float(row[4])})
        except (TypeError, ValueError):
            continue
    if not records:
        raise ValueError("Could not parse vector rows in the JPL Horizons response.")
    return records


def _nearest_record(records: list[dict], wanted: date) -> dict:
    def record_day(record):
        # Horizons date format: A.D. 2026-Jan-01 00:00:00.0000
        text = record["date"].replace("A.D. ", "").split()[0]
        return datetime.strptime(text, "%Y-%b-%d").date()

    return min(records, key=lambda row: abs((record_day(row) - wanted).days))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_solar_system(ephemeris_date: str, trail_days: int) -> tuple[list[dict], list[str]]:
    """Fetch all eight planets with one cached query per body; each query also supplies its trail."""
    center = datetime.strptime(ephemeris_date, "%Y-%m-%d").date()
    half = max(1, trail_days // 2)
    start_day, stop_day = center - timedelta(days=half), center + timedelta(days=half)
    step_days = max(1, trail_days // 60)
    start, stop = f"{start_day.isoformat()} 00:00", f"{stop_day.isoformat()} 00:00"
    bodies = [{"name": "Sun", "id": "10", "x": 0.0, "y": 0.0, "z": 0.0,
               "color": "#ffd54f", "size": 18, "trail": []}]
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {
            pool.submit(_horizons_vectors, body_id, start, stop, step_days): (name, body_id, color, size)
            for name, body_id, color, size in PLANETS
        }
        for task in as_completed(tasks):
            name, body_id, color, size = tasks[task]
            try:
                rows = task.result()
                current = _nearest_record(rows, center)
                bodies.append({"name": name, "id": body_id, "x": current["x"], "y": current["y"],
                               "z": current["z"], "color": color, "size": size, "trail": rows})
            except Exception as exc:
                errors.append(f"{name} (Horizons ID {body_id}): {exc}")
    return bodies, errors


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia_background(limit: int = 2500) -> tuple[list[dict], str | None]:
    """Get actual bright-star sky positions from Gaia DR3 for the background layer."""
    query = (
        f"SELECT TOP {int(limit)} ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source "
        "WHERE phot_g_mean_mag < 6.5 AND ra IS NOT NULL AND dec IS NOT NULL "
        "ORDER BY phot_g_mean_mag ASC"
    )
    try:
        response = requests.get(GAIA_TAP_URL, params={"REQUEST": "doQuery", "LANG": "ADQL",
                               "FORMAT": "csv", "QUERY": query}, timeout=(3, 8))
        response.raise_for_status()
        rows = csv.DictReader(io.StringIO(response.text))
        stars = [{"ra": float(row["ra"]), "dec": float(row["dec"]), "mag": float(row["phot_g_mean_mag"])}
                 for row in rows if row.get("ra") and row.get("dec") and row.get("phot_g_mean_mag")]
        return stars, None
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia_star(source_id: str) -> dict:
    """Resolve a numeric Gaia DR3 source_id to its measured catalog position."""
    if not source_id.isdigit():
        raise ValueError("Use the numeric Gaia DR3 source_id for the star.")
    query = ("SELECT TOP 1 source_id,ra,dec,parallax,phot_g_mean_mag FROM gaiadr3.gaia_source WHERE source_id="
             + source_id)
    response = requests.get(GAIA_TAP_URL, params={"REQUEST": "doQuery", "LANG": "ADQL",
                           "FORMAT": "csv", "QUERY": query}, timeout=(3, 8))
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows:
        raise ValueError("No Gaia DR3 source was found with that ID.")
    row = rows[0]
    return {"id": row["source_id"], "ra": float(row["ra"]), "dec": float(row["dec"]),
            "parallax": float(row["parallax"]) if row.get("parallax") else float("nan"),
            "mag": float(row["phot_g_mean_mag"]) if row.get("phot_g_mean_mag") else float("nan")}


def add_star_sky(fig, stars: list[dict], radius: float = 80.0):
    if not stars:
        return
    ra, dec = np.radians([star["ra"] for star in stars]), np.radians([star["dec"] for star in stars])
    mags = np.asarray([star["mag"] for star in stars])
    fig.add_trace(go.Scatter3d(
        x=radius * np.cos(dec) * np.cos(ra), y=radius * np.cos(dec) * np.sin(ra), z=radius * np.sin(dec),
        mode="markers", name="Gaia DR3 night sky", text=[f"Gaia DR3 · G={mag:.2f}" for mag in mags],
        hovertemplate="%{text}<extra></extra>", marker=dict(size=np.clip(5.2 - mags * .55, 1.2, 4),
        color="#f4f3e8", opacity=.82)))


def make_figure(bodies: list[dict], stars: list[dict], highlight: str, selected_star: dict | None):
    fig = go.Figure()
    add_star_sky(fig, stars)
    for body in bodies:
        if body["trail"]:
            trail = body["trail"]
            fig.add_trace(go.Scatter3d(x=[p["x"] for p in trail], y=[p["y"] for p in trail],
                z=[p["z"] for p in trail], mode="lines", name=f"{body['name']} orbit trail",
                line=dict(color=body["color"], width=2), hoverinfo="skip", showlegend=False))
    if selected_star:
        ra, dec = np.radians(selected_star["ra"]), np.radians(selected_star["dec"])
        fig.add_trace(go.Scatter3d(x=[80*np.cos(dec)*np.cos(ra)], y=[80*np.cos(dec)*np.sin(ra)],
            z=[80*np.sin(dec)], mode="markers+text", name="Selected Gaia star",
            text=[f"Gaia {selected_star['id']}"], textposition="top center",
            marker=dict(size=10, color="#ff66cc", symbol="diamond"),
            customdata=[[selected_star["ra"], selected_star["dec"], selected_star["parallax"], selected_star["mag"]]],
            hovertemplate="Gaia DR3 %{text}<br>RA %{customdata[0]:.5f}° · Dec %{customdata[1]:.5f}°<br>"
                          "Parallax %{customdata[2]:.3f} mas · G %{customdata[3]:.2f}<extra></extra>"))

    for body in bodies:
        emphasized = body["name"] == highlight
        distance = float(np.sqrt(body["x"]**2 + body["y"]**2 + body["z"]**2))
        fig.add_trace(go.Scatter3d(x=[body["x"]], y=[body["y"]], z=[body["z"]],
            mode="markers+text", name=body["name"], text=[body["name"]], textposition="top center",
            marker=dict(size=body["size"] + (5 if emphasized else 0), color=body["color"],
                        line=dict(color="white", width=1 if emphasized else .4)),
            customdata=[[body["id"], distance]],
            hovertemplate="<b>%{text}</b><br>Horizons ID %{customdata[0]}<br>Heliocentric distance "
                          "%{customdata[1]:.4f} AU<extra></extra>"))

    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(title="Real Solar System · JPL Horizons + Gaia DR3", paper_bgcolor="#02030a",
        plot_bgcolor="#02030a", font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(bgcolor="rgba(0,0,0,0.35)"), scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
        aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))))
    return fig


def query_spherex_most(target_name: str):
    """Use IRSA's Moving Object Search Tool to find actual SPHEREx observation records."""
    from astroquery.ipac.irsa.most import Most

    result = Most.query_object(target_name, catalog="spherex", output_mode="Regular")
    if result is None or len(result) == 0:
        return None
    return result.to_pandas() if hasattr(result, "to_pandas") else result


def render_infrared_search(default_target: str):
    st.subheader("Actual SPHEREx infrared observations · NASA/IPAC IRSA")
    st.markdown("Search IRSA's **Moving Object Search Tool (MOST)** for SPHEREx exposures intersecting a Solar System object's predicted path. This returns real archive observation records when available; it does not fabricate spectra or claim every planet has an observation.")
    with st.form("spherex_most_search"):
        target = st.text_input("Solar System object name or designation", value=default_target,
                               help="For example: Mars, Jupiter, or a small-body designation supported by IRSA MOST.")
        search = st.form_submit_button("Search real SPHEREx observations")
    if search:
        if not target.strip():
            st.warning("Enter a target name first.")
            return
        try:
            with st.spinner(f"Searching IRSA MOST for SPHEREx observations of {target.strip()}…"):
                observations = query_spherex_most(target.strip())
            if observations is None or len(observations) == 0:
                st.info("IRSA MOST returned no matching SPHEREx observation records for this target.")
            else:
                st.success(f"IRSA MOST returned {len(observations)} real observation record(s).")
                st.dataframe(observations, use_container_width=True, hide_index=True)
                url_columns = [column for column in observations.columns if "url" in column.lower() or "access" in column.lower()]
                for column in url_columns:
                    for url in observations[column].dropna().astype(str).head(10):
                        if url.startswith("http"):
                            st.markdown(f"[Open/download archive product]({url})")
        except Exception as exc:
            st.error(f"IRSA MOST search did not complete: {exc}")
            st.markdown("You can also search directly in the [IRSA MOST web tool](https://irsa.ipac.caltech.edu/applications/MOST/).")


def main():
    st.title("Real Solar System & Infrared Night Sky")
    st.caption("Planet positions and orbit tracks: NASA/JPL Horizons · Background stars: ESA Gaia DR3 · Infrared archive: NASA/IPAC IRSA SPHEREx MOST")
    st.sidebar.header("Explore")
    selected_planet = st.sidebar.selectbox("Highlight planet by name", ["None"] + PLANET_NAMES, index=0)
    chosen_date = st.sidebar.date_input("Ephemeris date (UTC)", value=date.today())
    trail_days = st.sidebar.select_slider("Orbit-trail span (days)", options=[30, 90, 180], value=90)
    st.sidebar.caption("All planets remain in view. Planet markers and orbital display are not to scale.")

    st.sidebar.subheader("Find a star")
    with st.sidebar.form("gaia_star_lookup"):
        source_id = st.text_input("Gaia DR3 source_id", placeholder="Numeric Gaia source ID")
        lookup_star = st.form_submit_button("Show star with Solar System")
    if lookup_star:
        st.session_state["selected_gaia_id"] = source_id.strip()

    date_text = chosen_date.isoformat()
    with st.spinner("Fetching real JPL Horizons positions and the Gaia DR3 star background…"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            planet_task = pool.submit(fetch_solar_system, date_text, int(trail_days))
            star_task = pool.submit(fetch_gaia_background)
            bodies, body_errors = planet_task.result()
            stars, star_error = star_task.result()

    selected_star = None
    star_id = st.session_state.get("selected_gaia_id", "")
    if star_id:
        try:
            with st.spinner("Looking up the Gaia DR3 source…"):
                selected_star = fetch_gaia_star(star_id)
        except Exception as exc:
            st.error(f"Could not load Gaia star: {exc}")

    if bodies:
        st.success(f"Loaded {len(bodies)} Solar System body/bodies from JPL Horizons for {date_text}.")
    else:
        st.error("JPL Horizons could not be reached. Current real planet positions are unavailable; check network access and retry.")
    if body_errors:
        st.warning("Some Horizons requests failed: " + "; ".join(body_errors))
    if star_error:
        st.warning(f"Gaia's real star catalog is temporarily unavailable: {star_error}. Planet positions remain available.")
    elif stars:
        st.caption(f"Background: {len(stars):,} actual bright Gaia DR3 stars, displayed as sky directions on a compressed shell.")

    st.plotly_chart(make_figure(bodies, stars, selected_planet, selected_star), use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})
    st.caption("The Sun is at the origin. Planet coordinates are real heliocentric AU vectors; star directions are from Gaia DR3. Distances and marker sizes are compressed/enlarged for legibility, so this is not a literal scale model.")

    render_infrared_search(selected_planet if selected_planet != "None" else "Mars")
    st.markdown("Data services: [JPL Horizons](https://ssd.jpl.nasa.gov/api/horizons.api.html) · [Gaia Archive](https://gea.esac.esa.int/archive/) · [IRSA SPHEREx](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [MOST web tool](https://irsa.ipac.caltech.edu/applications/MOST/)")


if __name__ == "__main__":
    main()
