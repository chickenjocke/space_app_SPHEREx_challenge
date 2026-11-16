from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SPHEREx Infrared Solar System Explorer", layout="wide")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/sia2query/SPHEREx"
PLANET_IDS = {
    "Mercury": "199", "Venus": "299", "Earth": "399", "Mars": "499",
    "Jupiter": "599", "Saturn": "699", "Uranus": "799", "Neptune": "899",
    "Sun": "10",
}


def _sexagesimal_to_degrees(value: str, is_ra: bool) -> float:
    """Convert Horizons sexagesimal RA/Dec (or decimal degrees) to degrees."""
    value = value.strip()
    parts = value.split()
    if len(parts) == 1:
        return float(parts[0])
    sign = -1.0 if parts[0].startswith("-") else 1.0
    first = abs(float(parts[0]))
    result = first + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
    if is_ra:
        result *= 15.0
    return sign * result


def _horizons_apparent_radec(target: str, observed_on: date) -> tuple[float, float]:
    """Resolve any Horizons target to its apparent geocentric RA/Dec for the selected date."""
    start = datetime.combine(observed_on, datetime.min.time())
    stop = start + timedelta(minutes=2)
    params = {
        "format": "json", "COMMAND": f"'{target}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'500@399'",
        "QUANTITIES": "'1'", "CSV_FORMAT": "'YES'",
        "START_TIME": f"'{start:%Y-%m-%d %H:%M}'", "STOP_TIME": f"'{stop:%Y-%m-%d %H:%M}'",
        "STEP_SIZE": "'1 m'",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=(3, 12))
    response.raise_for_status()
    result = response.json().get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise ValueError(response.json().get("error", "Horizons returned no position for this target/date."))
    rows = list(csv.reader(io.StringIO(result.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip())))
    for row in rows:
        # With QUANTITIES=1 Horizons emits date, blank, RA, Dec, followed by other columns.
        for index in range(1, max(2, len(row) - 1)):
            if index + 1 >= len(row):
                break
            try:
                ra = _sexagesimal_to_degrees(row[index], is_ra=True)
                dec = _sexagesimal_to_degrees(row[index + 1], is_ra=False)
                if 0 <= ra < 360 and -90 <= dec <= 90:
                    return ra, dec
            except (ValueError, IndexError):
                continue
    raise ValueError("Could not read RA/Dec from the Horizons response. Try a planet name or Horizons ID.")


def _parse_sia_votable(xml_bytes: bytes) -> list[dict]:
    """Parse an IRSA SIA VOTable without adding a heavyweight astronomy dependency."""
    root = ET.fromstring(xml_bytes)
    fields = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "FIELD":
            fields.append(element.attrib.get("name") or element.attrib.get("ID") or f"column_{len(fields)}")
    records = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "TR":
            continue
        values = [child.text or "" for child in element if child.tag.rsplit("}", 1)[-1] == "TD"]
        if values:
            records.append({fields[i] if i < len(fields) else f"column_{i}": value
                            for i, value in enumerate(values)})
    return records


def _column(record: dict, *names: str) -> str:
    lookup = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return ""


def _float_field(record: dict, *names: str) -> float | None:
    try:
        value = float(_column(record, *names))
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def query_spherex_at_position(ra: float, dec: float, radius_deg: float) -> list[dict]:
    """Search actual SPHEREx spectral-image coverage around an apparent target position."""
    response = requests.get(
        IRSA_SIA_URL,
        params={
            "POS": f"CIRCLE {ra:.8f} {dec:.8f} {radius_deg}",
            "BAND": "0.75e-6 5.0e-6",
        },
        headers={"Accept": "application/x-votable+xml"},
        timeout=(4, 18),
    )
    response.raise_for_status()
    if not response.content.strip():
        return []
    return _parse_sia_votable(response.content)


def _local_preview_positions(day: date) -> dict:
    """Fast approximate local orbit preview; never presented as archive/infrared data."""
    # Compact circular-orbit preview keeps initial page rendering independent of network services.
    periods = {"Mercury": 87.969, "Venus": 224.701, "Earth": 365.256,
               "Mars": 686.980, "Jupiter": 4332.59, "Saturn": 10759.2,
               "Uranus": 30688.5, "Neptune": 60182.0}
    radii = {"Mercury": .387, "Venus": .723, "Earth": 1., "Mars": 1.524,
             "Jupiter": 5.203, "Saturn": 9.555, "Uranus": 19.218, "Neptune": 30.11}
    colors = {"Mercury": "#b8b8b8", "Venus": "#f5c46b", "Earth": "#4da6ff", "Mars": "#ff7043",
              "Jupiter": "#e8b17a", "Saturn": "#f2dc8c", "Uranus": "#83e3e3", "Neptune": "#577dff"}
    elapsed = (day - date(2000, 1, 1)).days
    positions = {"Sun": (0., 0., 0.)}
    for index, name in enumerate(periods):
        angle = 2 * np.pi * elapsed / periods[name] + index * .71
        positions[name] = (radii[name] * np.cos(angle), radii[name] * np.sin(angle), 0.)
    return positions


def solar_system_figure(day: date, highlighted: str):
    positions = _local_preview_positions(day)
    fig = go.Figure()
    orbit_radii = {"Mercury": .387, "Venus": .723, "Earth": 1., "Mars": 1.524,
                   "Jupiter": 5.203, "Saturn": 9.555, "Uranus": 19.218, "Neptune": 30.11}
    colors = {"Mercury": "#b8b8b8", "Venus": "#f5c46b", "Earth": "#4da6ff", "Mars": "#ff7043",
              "Jupiter": "#e8b17a", "Saturn": "#f2dc8c", "Uranus": "#83e3e3", "Neptune": "#577dff"}
    for name, radius in orbit_radii.items():
        angle = np.linspace(0, 2 * np.pi, 120)
        fig.add_trace(go.Scatter3d(x=radius*np.cos(angle), y=radius*np.sin(angle), z=np.zeros_like(angle),
                                   mode="lines", line=dict(color=colors[name], width=1),
                                   name=f"{name} orbit (preview)", showlegend=False, hoverinfo="skip"))
        x, y, z = positions[name]
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers+text", name=name, text=[name],
                                   textposition="top center", marker=dict(size=8+(4 if name == highlighted else 0), color=colors[name])))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", name="Sun", text=["Sun"],
                               textposition="top center", marker=dict(size=15, color="#ffd54f")))
    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(title=f"Solar System preview · {day.isoformat()}", paper_bgcolor="#02030a",
                      plot_bgcolor="#02030a", font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=40),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data"))
    return fig


def show_observations(records: list[dict]):
    st.success(f"IRSA returned {len(records)} actual SPHEREx archive product(s) in this sky area.")
    wavelength_rows = []
    for index, record in enumerate(records):
        lo = _float_field(record, "em_min")
        hi = _float_field(record, "em_max")
        # Standard SIA spectral bounds are in meters; show microns.
        if lo is not None and hi is not None and lo > 0 and hi > lo:
            wavelength_rows.append((index + 1, lo * 1e6, hi * 1e6))
    if wavelength_rows:
        chart = go.Figure()
        chart.add_trace(go.Bar(
            x=[hi - lo for _, lo, hi in wavelength_rows],
            base=[lo for _, lo, hi in wavelength_rows],
            y=[f"Product {number}" for number, _, _ in wavelength_rows],
            orientation="h", marker_color="#ff9f43",
            hovertemplate="%{y}<br>Wavelength: %{base:.3f}–%{x:.3f} μm<extra></extra>",
        ))
        chart.update_layout(title="Observed SPHEREx product wavelength coverage (archive metadata)",
                            xaxis_title="Wavelength (μm)", yaxis_title="Archive product",
                            paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="white",
                            height=max(300, min(800, 28 * len(wavelength_rows) + 120)))
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Products were found, but IRSA did not return em_min/em_max wavelength bounds for plotting.")

    display_rows = []
    for index, record in enumerate(records):
        display_rows.append({
            "#": index + 1,
            "Product": _column(record, "obs_id", "obs_publisher_did", "obs_title", "productID", "filename") or "SPHEREx data product",
            "Type": _column(record, "dataproduct_type", "productType"),
            "Wavelength min (μm)": (lambda x: round(x * 1e6, 4) if x and x > 0 else "")(_float_field(record, "em_min")),
            "Wavelength max (μm)": (lambda x: round(x * 1e6, 4) if x and x > 0 else "")(_float_field(record, "em_max")),
            "Observation time": _column(record, " t_min", "t_min", "obs_collection"),
        })
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    st.markdown("**Archive product files**")
    shown = 0
    for index, record in enumerate(records):
        product_url = _column(record, "access_url", "access_reference", "accessURL")
        if product_url.startswith("http"):
            title = _column(record, "obs_id", "obs_title", "filename") or f"SPHEREx product {index + 1}"
            st.link_button(f"Open/download: {title[:90]}", product_url)
            shown += 1
            if shown >= 20:
                break
    if not shown:
        st.caption("IRSA returned no direct access_url in the response. Open the IRSA archive result details for product access.")
    st.warning("These are real SPHEREx archive products and measured wavelength coverage, not fabricated planet fluxes. A product may be a spectral image/cube; extracting a calibrated per-planet spectrum requires processing the FITS product and checking that the target is actually in the field.")


def main():
    st.title("SPHEREx Infrared Solar System Explorer")
    st.caption("The app renders immediately. It queries JPL Horizons and NASA/IPAC IRSA only when you submit a search.")
    with st.sidebar:
        st.header("Solar System")
        selected = st.selectbox("Highlight planet by name", ["None", *[n for n in PLANET_IDS if n != "Sun"]])
        observed_on = st.date_input("Observation date (UTC)", value=date.today())
        st.caption("The chart is a quick illustrative preview. Archive search below uses the selected date and a real Horizons apparent position.")

    st.plotly_chart(solar_system_figure(observed_on, selected), use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})
    st.caption("Illustrative orbit preview only — not precise ephemerides and not infrared imagery.")

    st.header("Actual infrared archive search")
    st.write("Resolve a Solar System object with NASA/JPL Horizons, then search the official SPHEREx SIA archive for real spectral-image products around its apparent sky position. Any Horizons-recognized name, NAIF ID, asteroid, or comet may be tried; results depend on actual survey coverage.")
    default_target = selected if selected != "None" else "Mars"
    with st.form("infrared_search"):
        target = st.text_input("Object name or Horizons ID", value=default_target,
                               help="Examples: Mars, Jupiter, 499, Ceres, or another Horizons target.")
        radius = st.select_slider("Search radius around target", options=[0.02, 0.05, 0.1, 0.25], value=0.1,
                                  format_func=lambda value: f"{value}°")
        submit = st.form_submit_button("Find real SPHEREx infrared observations", type="primary")

    if submit:
        if not target.strip():
            st.error("Enter a planet/object name or Horizons ID.")
        else:
            try:
                with st.spinner("Resolving apparent target position with JPL Horizons…"):
                    ra, dec = _horizons_apparent_radec(target.strip(), observed_on)
                st.write(f"**{target.strip()}** on {observed_on.isoformat()}: RA {ra:.5f}°, Dec {dec:.5f}°")
                with st.spinner("Searching NASA/IPAC IRSA SPHEREx data (request has a timeout)…"):
                    products = query_spherex_at_position(ra, dec, float(radius))
                if products:
                    show_observations(products)
                else:
                    st.info("No SPHEREx products were returned at this position/radius. Try a wider radius or another observation date. No fake infrared points are substituted.")
            except requests.Timeout:
                st.error("The archive timed out. The page remains available; retry in a moment or use the IRSA links below.")
            except requests.RequestException as exc:
                st.error(f"A NASA service could not be reached: {exc}")
            except Exception as exc:
                st.error(f"Could not complete the target/archive search: {exc}")

    st.markdown("Official resources: [IRSA SPHEREx archive](https://irsa.ipac.caltech.edu/Missions/spherex.html) · [IRSA MOST moving-object search](https://irsa.ipac.caltech.edu/applications/MOST/) · [JPL Horizons API](https://ssd.jpl.nasa.gov/api/horizons.api.html)")


if __name__ == "__main__":
    main()
