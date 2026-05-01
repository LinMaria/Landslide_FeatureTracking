from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
from dash import Dash, Input, Output, State, dcc, html
from rasterio.features import rasterize

from src.io_utils import load_image


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "data" / "output" / "aligned"
RESULTS_DIR = BASE_DIR / "data" / "output" / "results"
INPUT_DIR = BASE_DIR / "data" / "input"
ANALYSIS_AREAS_PATH = INPUT_DIR / "analysis_areas.shp"

PIXEL_SIZE_M = 0.100008
VMAX_M = 2.5
DIRECTION_THRESHOLD_PX = 0.25
AREA_LINE_COLORS = {
    "flank": "#e76f51",
    "flux": "#2a9d8f",
}
OVERLAY_LABELS = {
    "magnitude": "Displacement Magnitude",
    "direction": "Movement Direction",
    "change": "Change Mask",
}

_cache = {}


def get_dense_field_pairs():
    files = sorted(RESULTS_DIR.glob("dense_field_*.npy"))
    pairs = []
    for path in files:
        label = path.stem.replace("dense_field_", "")
        d1, d2 = label.split("_to_")
        pairs.append((d1, d2))
    return pairs


PAIR_SEQUENCE = get_dense_field_pairs()
PAIR_LABELS = [f"{d1} -> {d2}" for d1, d2 in PAIR_SEQUENCE]


def _first_aligned_path() -> Path:
    try:
        return sorted(IMG_DIR.glob("clip_*_aligned.tif"))[0]
    except IndexError as exc:
        raise FileNotFoundError(f"No aligned images found in {IMG_DIR}") from exc


def _reference_raster_info():
    key = ("reference_raster_info",)
    if key not in _cache:
        with rasterio.open(_first_aligned_path()) as src:
            _cache[key] = {
                "shape": (src.height, src.width),
                "transform": src.transform,
                "crs": src.crs,
            }
    return _cache[key]


def _load_rgb(date_str: str) -> np.ndarray:
    path = IMG_DIR / f"clip_{date_str}_aligned.tif"
    key = ("img", str(path))
    if key not in _cache:
        img_bgr, _ = load_image(path)
        if img_bgr.ndim == 2:
            img_rgb = np.dstack([img_bgr, img_bgr, img_bgr])
        else:
            img_rgb = img_bgr[..., ::-1]
        _cache[key] = img_rgb
    return _cache[key]


def _load_flow(date_a: str, date_b: str) -> np.ndarray:
    path = RESULTS_DIR / f"dense_field_{date_a}_to_{date_b}.npy"
    key = ("flow", str(path))
    if key not in _cache:
        _cache[key] = np.load(path).astype(np.float32)
    return _cache[key]


def _load_change(date_a: str, date_b: str) -> np.ndarray:
    path = RESULTS_DIR / f"change_mask_{date_a}_to_{date_b}.npy"
    key = ("change", str(path))
    if key not in _cache:
        _cache[key] = np.load(path)
    return _cache[key]


def _load_analysis_areas(shape_hw):
    key = ("areas", tuple(shape_hw))
    if key in _cache:
        return _cache[key]

    raster_info = _reference_raster_info()
    gdf = gpd.read_file(ANALYSIS_AREAS_PATH)
    if gdf.crs != raster_info["crs"]:
        gdf = gdf.to_crs(raster_info["crs"])

    areas = []
    for idx, row in gdf.iterrows():
        area_name = str(row.get("Type", row.get("area", f"area_{idx + 1}"))).strip() or f"area_{idx + 1}"
        mask = rasterize(
            [(row.geometry, 1)],
            out_shape=shape_hw,
            fill=0,
            transform=raster_info["transform"],
            dtype="uint8",
        ).astype(bool)
        ys, xs = np.where(mask)
        if xs.size:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            poly_x = [x0, x1, x1, x0, x0]
            poly_y = [y0, y0, y1, y1, y0]
        else:
            poly_x, poly_y = [], []
        areas.append(
            {
                "name": area_name,
                "mask": mask,
                "poly_x": poly_x,
                "poly_y": poly_y,
                "pixel_count": int(mask.sum()),
            }
        )
    _cache[key] = areas
    return areas


def _pair_products(date_a: str, date_b: str):
    key = ("products", date_a, date_b)
    if key in _cache:
        return _cache[key]

    img = _load_rgb(date_b)
    flow = _load_flow(date_a, date_b)
    change = _load_change(date_a, date_b)

    mag_px = np.linalg.norm(flow, axis=-1)
    mag_m = mag_px * PIXEL_SIZE_M
    direction = (np.degrees(np.arctan2(flow[..., 1], flow[..., 0])) + 360) % 360
    direction = np.ma.masked_where(mag_px <= DIRECTION_THRESHOLD_PX, direction)

    products = {
        "img": img,
        "flow": flow,
        "change": change,
        "mag_px": mag_px,
        "mag_m": mag_m,
        "direction": direction,
        "areas": _load_analysis_areas(img.shape[:2]),
    }
    _cache[key] = products
    return products


def _base_figure():
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=52, b=10),
        paper_bgcolor="#f3efe5",
        plot_bgcolor="#f3efe5",
        font=dict(family="Georgia, serif", color="#24323d"),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False, autorange="reversed", scaleanchor="x")
    return fig


def _overlay_trace(kind: str, products, opacity: float):
    if kind == "magnitude":
        return go.Heatmap(
            z=products["mag_m"],
            colorscale="Viridis",
            zmin=0,
            zmax=VMAX_M,
            opacity=opacity,
            colorbar=dict(title="Displacement [m]"),
            hovertemplate="disp=%{z:.2f} m<extra></extra>",
        )
    if kind == "direction":
        return go.Heatmap(
            z=products["direction"],
            colorscale="HSV",
            zmin=0,
            zmax=360,
            opacity=opacity,
            colorbar=dict(
                title="Direction",
                tickvals=[0, 90, 180, 270, 360],
                ticktext=["E", "N", "W", "S", "E"],
            ),
            hovertemplate="dir=%{z:.1f}°<extra></extra>",
        )
    if kind == "change":
        return go.Heatmap(
            z=(products["change"] > 0).astype(float),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(214,40,40,0.90)"]],
            showscale=False,
            opacity=opacity,
            hovertemplate="changed=%{z}<extra></extra>",
        )
    raise ValueError(f"Unknown overlay kind: {kind}")


def _area_traces(products, selected_areas):
    traces = []
    for area in products["areas"]:
        if selected_areas and area["name"] not in selected_areas:
            continue
        traces.append(
            go.Scatter(
                x=area["poly_x"],
                y=area["poly_y"],
                mode="lines",
                line=dict(color=AREA_LINE_COLORS.get(area["name"], "#ffffff"), width=3),
                name=area["name"].title(),
                hoverinfo="skip",
            )
        )
    return traces


def build_frame_figure(overlay_kind: str, pair_label: str, selected_areas, opacity: float):
    date_a, date_b = pair_label.split(" -> ")
    products = _pair_products(date_a, date_b)
    fig = _base_figure()
    fig.add_trace(go.Image(z=products["img"], hoverinfo="skip"))
    fig.add_trace(_overlay_trace(overlay_kind, products, opacity))
    for trace in _area_traces(products, selected_areas):
        fig.add_trace(trace)
    fig.update_layout(title=f"{OVERLAY_LABELS[overlay_kind]} | {pair_label}")
    return fig


def build_animation_figure(overlay_kind: str, selected_areas, opacity: float, frame_duration_ms: int):
    first_label = PAIR_LABELS[0]
    fig = build_frame_figure(overlay_kind, first_label, selected_areas, opacity)
    frames = []
    for label in PAIR_LABELS:
        date_a, date_b = label.split(" -> ")
        products = _pair_products(date_a, date_b)
        frame_data = [
            go.Image(z=products["img"], hoverinfo="skip"),
            _overlay_trace(overlay_kind, products, opacity),
            *_area_traces(products, selected_areas),
        ]
        frames.append(go.Frame(data=frame_data, name=label))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 200},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": -0.05,
                "len": 0.88,
                "pad": {"b": 10, "t": 10},
                "steps": [
                    {
                        "label": label,
                        "method": "animate",
                        "args": [[label], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for label in PAIR_LABELS
                ],
            }
        ],
    )
    return fig


def _area_stats(date_a: str, date_b: str, selected_areas):
    products = _pair_products(date_a, date_b)
    rows = []
    for area in products["areas"]:
        if selected_areas and area["name"] not in selected_areas:
            continue
        mask = area["mask"]
        mag = products["mag_m"][mask]
        direction_vals = np.ma.compressed(products["direction"][mask])
        changed = int((products["change"][mask] > 0).sum())
        rows.append(
            {
                "Area": area["name"],
                "Mean displacement [m]": f"{float(np.mean(mag)):.3f}" if mag.size else "nan",
                "Max displacement [m]": f"{float(np.max(mag)):.3f}" if mag.size else "nan",
                "Mean direction [deg]": f"{float(np.mean(direction_vals)):.1f}" if direction_vals.size else "nan",
                "Changed pixels": changed,
            }
        )
    return rows


def build_area_timeseries(selected_areas):
    records = []
    for date_a, date_b in PAIR_SEQUENCE:
        products = _pair_products(date_a, date_b)
        for area in products["areas"]:
            if selected_areas and area["name"] not in selected_areas:
                continue
            mask = area["mask"]
            mag = products["mag_m"][mask]
            records.append(
                {
                    "pair_label": f"{date_a} -> {date_b}",
                    "curr_date": pd.Timestamp(date_b),
                    "area_name": area["name"],
                    "mean_disp_m": float(np.mean(mag)) if mag.size else np.nan,
                    "p95_disp_m": float(np.percentile(mag, 95)) if mag.size else np.nan,
                    "changed_pixels": int((products["change"][mask] > 0).sum()),
                }
            )

    df = pd.DataFrame(records)
    fig = go.Figure()
    for area_name, area_df in df.groupby("area_name"):
        color = AREA_LINE_COLORS.get(area_name, "#264653")
        fig.add_trace(
            go.Scatter(
                x=area_df["curr_date"],
                y=area_df["mean_disp_m"],
                mode="lines+markers",
                name=f"{area_name.title()} mean",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate="%{x|%Y-%m-%d}<br>mean=%{y:.3f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=area_df["curr_date"],
                y=area_df["p95_disp_m"],
                mode="lines+markers",
                name=f"{area_name.title()} p95",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=7),
                hovertemplate="%{x|%Y-%m-%d}<br>p95=%{y:.3f} m<extra></extra>",
            )
        )

    fig.update_layout(
        margin=dict(l=50, r=20, t=45, b=40),
        paper_bgcolor="#f3efe5",
        plot_bgcolor="white",
        font=dict(family="Georgia, serif", color="#24323d"),
        title="Area Comparison Through Time",
        xaxis_title="Observation date",
        yaxis_title="Displacement [m]",
        legend_title="Series",
    )
    return fig


app = Dash(__name__)
app.title = "Landslide Dashboard"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Landslide Displacement Dashboard", style={"marginBottom": "0.2rem"}),
                html.P(
                    "Animate full-image displacement products, inspect single pairs, and compare flank and flux behavior through time."
                ),
            ],
            style={"padding": "1rem 1.5rem 0.25rem 1.5rem"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Overlay"),
                        dcc.Dropdown(
                            id="overlay-kind",
                            options=[{"label": label, "value": value} for value, label in OVERLAY_LABELS.items()],
                            value="magnitude",
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Pair"),
                        dcc.Dropdown(
                            id="pair-label",
                            options=[{"label": label, "value": label} for label in PAIR_LABELS],
                            value=PAIR_LABELS[0],
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1.35"},
                ),
                html.Div(
                    [
                        html.Label("Areas"),
                        dcc.Checklist(
                            id="selected-areas",
                            options=[{"label": "Flank", "value": "flank"}, {"label": "Flux", "value": "flux"}],
                            value=["flank", "flux"],
                            inline=True,
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "1rem", "padding": "0.5rem 1.5rem 0.5rem 1.5rem"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Overlay opacity"),
                        dcc.Slider(id="overlay-opacity", min=0.15, max=1.0, step=0.05, value=0.68),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Animation speed"),
                        dcc.Slider(
                            id="animation-speed",
                            min=200,
                            max=1500,
                            step=100,
                            value=700,
                            marks={200: "Fast", 700: "Default", 1500: "Slow"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "1rem", "padding": "0.25rem 1.5rem 1rem 1.5rem"},
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Animated Overlay",
                    children=[dcc.Graph(id="animated-graph", style={"height": "76vh"})],
                ),
                dcc.Tab(
                    label="Single Pair Inspector",
                    children=[
                        dcc.Graph(id="pair-graph", style={"height": "68vh"}),
                        html.Pre(id="pair-readout", style={"padding": "0.75rem 1rem", "fontSize": "0.95rem"}),
                        html.Div(id="area-table-wrapper", style={"padding": "0 1rem 1rem 1rem"}),
                    ],
                ),
                dcc.Tab(
                    label="Area Trends",
                    children=[dcc.Graph(id="area-timeseries", style={"height": "72vh"})],
                ),
            ]
        ),
    ],
    style={"backgroundColor": "#f3efe5", "minHeight": "100vh"},
)


@app.callback(
    Output("animated-graph", "figure"),
    Input("overlay-kind", "value"),
    Input("selected-areas", "value"),
    Input("overlay-opacity", "value"),
    Input("animation-speed", "value"),
)
def update_animated_graph(overlay_kind, selected_areas, overlay_opacity, animation_speed):
    return build_animation_figure(
        overlay_kind,
        selected_areas or [],
        float(overlay_opacity),
        int(animation_speed),
    )


@app.callback(
    Output("pair-graph", "figure"),
    Input("overlay-kind", "value"),
    Input("pair-label", "value"),
    Input("selected-areas", "value"),
    Input("overlay-opacity", "value"),
)
def update_pair_graph(overlay_kind, pair_label, selected_areas, overlay_opacity):
    return build_frame_figure(overlay_kind, pair_label, selected_areas or [], float(overlay_opacity))


@app.callback(
    Output("pair-readout", "children"),
    Input("pair-graph", "clickData"),
    State("pair-label", "value"),
)
def update_readout(click_data, pair_label):
    if not click_data:
        return "Click on the image to inspect displacement, direction, and change state at one location."

    x = int(round(click_data["points"][0]["x"]))
    y = int(round(click_data["points"][0]["y"]))
    date_a, date_b = pair_label.split(" -> ")
    products = _pair_products(date_a, date_b)
    h, w = products["flow"].shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return f"Clicked outside bounds: x={x}, y={y}, bounds=({w}, {h})"

    u, v = products["flow"][y, x]
    mag_m = float(np.hypot(u, v) * PIXEL_SIZE_M)
    direction = float((np.degrees(np.arctan2(v, u)) + 360) % 360)
    changed = int(products["change"][y, x] > 0)

    labels = []
    for area in products["areas"]:
        if area["mask"][y, x]:
            labels.append(area["name"])
    area_label = ", ".join(labels) if labels else "outside flank/flux"

    return (
        f"Pair: {pair_label}\n"
        f"Pixel: ({x}, {y})\n"
        f"Area: {area_label}\n"
        f"u = {u:.3f} px\n"
        f"v = {v:.3f} px\n"
        f"Magnitude = {mag_m:.3f} m\n"
        f"Direction = {direction:.1f}°\n"
        f"Changed = {changed}"
    )


@app.callback(
    Output("area-table-wrapper", "children"),
    Input("pair-label", "value"),
    Input("selected-areas", "value"),
)
def update_area_table(pair_label, selected_areas):
    date_a, date_b = pair_label.split(" -> ")
    df = pd.DataFrame(_area_stats(date_a, date_b, selected_areas or []))
    if df.empty:
        return html.Div("No analysis area selected.", style={"padding": "0.5rem 0"})
    return dcc.Markdown(df.to_markdown(index=False))


@app.callback(
    Output("area-timeseries", "figure"),
    Input("selected-areas", "value"),
)
def update_area_timeseries(selected_areas):
    return build_area_timeseries(selected_areas or [])


if __name__ == "__main__":
    app.run(debug=True)
