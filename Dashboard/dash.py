import os
import numpy as np
import cv2
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

# -----------------------------
# CONFIG (edit to your paths)
# -----------------------------

# Dynamically find all available dates from aligned images and dense field files
IMG_DIR  = "data/output/aligned"
RESULTS_DIR = "data/output/results"

# Get all aligned image dates
def get_dates_from_images():
    files = [f for f in os.listdir(IMG_DIR) if f.startswith("clip_") and f.endswith("_aligned.tif")]
    dates = [f[len("clip_"):-len("_aligned.tif")] for f in files]
    return sorted(dates)

# Get all available dense field pairs
def get_dense_field_pairs():
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("dense_field_") and f.endswith(".npy")]
    pairs = []
    for f in files:
        s = f[len("dense_field_"):-len(".npy")]
        d1, d2 = s.split("_to_")
        pairs.append((d1, d2))
    pairs = sorted(pairs)
    return pairs

DENSE_FIELD_PAIRS = get_dense_field_pairs()
DATES = sorted(set([d for pair in DENSE_FIELD_PAIRS for d in pair]))

# For compatibility with the rest of the code, build a mapping from index to (d1, d2)
PAIR_IDX = [(d1, d2) for (d1, d2) in DENSE_FIELD_PAIRS]

DISPLAY_SCALE = 0.25   # downsample for speed; 1.0 for full-res
VMAX = 25              # pixels (for magnitude colorscale)
ARROW_STEP = 60        # subsample step (pixels in DISPLAY space)
ARROW_SCALE = 3.0      # arrow length multiplier for display

# -----------------------------
# CACHING (avoid reloading)
# -----------------------------
_cache = {}

def _load_rgb(date_str: str) -> np.ndarray:
    """
    Load an aligned image for display (RGB uint8).
    """
    path = os.path.join(IMG_DIR, f"clip_{date_str}_aligned.tif")
    if path not in _cache:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _cache[path] = img_rgb
    return _cache[path]

def _load_flow(date_a: str, date_b: str) -> np.ndarray:
    """
    Load dense field (flow) from date_a -> date_b as (H,W,2) float32 in pixels.
    """
    path = os.path.join(RESULTS_DIR, f"dense_field_{date_a}_to_{date_b}.npy")
    if path not in _cache:
        flow = np.load(path)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Flow must be (H,W,2). Got: {flow.shape} from {path}")
        _cache[path] = flow.astype(np.float32)
    return _cache[path]

def _downsample_for_display(img_rgb, flow):
    """
    If you display at smaller resolution, also downsample flow AND scale vectors.
    """
    if DISPLAY_SCALE == 1.0:
        return img_rgb, flow

    h, w = img_rgb.shape[:2]
    new_w = int(w * DISPLAY_SCALE)
    new_h = int(h * DISPLAY_SCALE)

    img_ds = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # resize each component then scale magnitudes to match new pixel units
    u = cv2.resize(flow[..., 0], (new_w, new_h), interpolation=cv2.INTER_AREA) * DISPLAY_SCALE
    v = cv2.resize(flow[..., 1], (new_w, new_h), interpolation=cv2.INTER_AREA) * DISPLAY_SCALE
    flow_ds = np.dstack([u, v]).astype(np.float32)
    return img_ds, flow_ds

def _make_quiver_traces(flow):
    """
    Build a single Scatter trace with many small line segments (fast-ish).
    """
    h, w = flow.shape[:2]
    xs, ys = [], []
    for y in range(0, h, ARROW_STEP):
        for x in range(0, w, ARROW_STEP):
            u, v = flow[y, x]
            x2 = x + u * ARROW_SCALE
            y2 = y + v * ARROW_SCALE
            xs += [x, x2, None]
            ys += [y, y2, None]

    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=1),
        name="arrows",
        hoverinfo="skip"
    )

def _figure_for(viz, img_rgb, flow, title):
    mag = np.hypot(flow[..., 0], flow[..., 1])
    ang = np.arctan2(flow[..., 1], flow[..., 0])  # radians [-pi, pi]

    fig = go.Figure()
    fig.add_trace(go.Image(z=img_rgb))

    if viz == "magnitude":
        fig.add_trace(go.Heatmap(
            z=mag,
            colorscale="Hot",
            zmin=0, zmax=VMAX,
            opacity=0.6,
            colorbar=dict(title="|u| [px]")
        ))
    elif viz == "direction":
        fig.add_trace(go.Heatmap(
            z=ang,
            colorscale="hsv",
            zmin=-np.pi, zmax=np.pi,
            opacity=0.6,
            colorbar=dict(title="angle [rad]")
        ))
    elif viz == "arrows":
        fig.add_trace(_make_quiver_traces(flow))
    else:
        raise ValueError("viz must be one of: magnitude, direction, arrows")

    # image-like axes (y downwards)
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        clickmode="event+select",
        showlegend=False
    )
    return fig

# -----------------------------
# DASH APP
# -----------------------------
app = Dash(__name__)


# We show all available dense field pairs
MAX_I = len(PAIR_IDX) - 1
marks = {i: f"{PAIR_IDX[i][0]}→{PAIR_IDX[i][1]}" for i in range(MAX_I + 1)}

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label("Visualization"),
            dcc.Dropdown(
                id="viz",
                options=[
                    {"label": "Magnitude", "value": "magnitude"},
                    {"label": "Direction", "value": "direction"},
                    {"label": "Arrows", "value": "arrows"},
                ],
                value="magnitude",
                clearable=False,
            ),
        ], style={"width": "250px", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.Label("Timeline"),
            dcc.Slider(
                id="frame",
                min=0, max=MAX_I, step=1, value=0,
                marks=marks
            ),
        ], style={"marginLeft": "20px", "display": "inline-block", "width": "70%"}),
    ], style={"marginBottom": "10px"}),

    dcc.Graph(id="graph", style={"height": "85vh"}),

    html.Div(id="readout", style={"fontFamily": "monospace", "padding": "8px"})
])

@app.callback(
    Output("graph", "figure"),
    Input("frame", "value"),
    Input("viz", "value"),
)
def update_graph(i, viz):
    d1, d2 = PAIR_IDX[i]
    img = _load_rgb(d2)            # show the later date as background (common choice)
    flow = _load_flow(d1, d2)
    img, flow = _downsample_for_display(img, flow)

    title = f"{viz} | {d1} → {d2} (display scale={DISPLAY_SCALE})"
    return _figure_for(viz, img, flow, title)

@app.callback(
    Output("readout", "children"),
    Input("graph", "clickData"),
    State("frame", "value"),
)
def show_click_value(clickData, i):
    if not clickData:
        return "Click on the image to read u, v, magnitude, direction."

    x = int(round(clickData["points"][0]["x"]))
    y = int(round(clickData["points"][0]["y"]))

    d1, d2 = PAIR_IDX[i]
    flow = _load_flow(d1, d2)
    img = _load_rgb(d2)
    img_ds, flow_ds = _downsample_for_display(img, flow)

    h, w = flow_ds.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return f"Clicked outside bounds: x={x}, y={y}, bounds=({w},{h})"

    u, v = flow_ds[y, x]
    mag = float(np.hypot(u, v))
    ang = float(np.arctan2(v, u))  # radians

    return (
        f"date: {d1}→{d2} | (x,y)=({x},{y})\n"
        f"u={u:.3f} px, v={v:.3f} px, |u|={mag:.3f} px, angle={ang:.3f} rad"
    )

if __name__ == "__main__":
    app.run_server(debug=True)