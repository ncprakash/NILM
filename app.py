"""NILM v5 — Non-Intrusive Load Monitoring Web App"""
import os, io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NILM — Smart Energy Disaggregation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
APPLIANCES    = ["kettle", "fridge", "washing_machine", "dish_washer"]
APP_LABELS    = {"kettle": "Kettle", "fridge": "Fridge",
                 "washing_machine": "Washing Machine", "dish_washer": "Dishwasher"}
APP_ICONS     = {"kettle": "🫖", "fridge": "🧊",
                 "washing_machine": "🫧", "dish_washer": "🍽️"}
APP_COLORS    = {"kettle": "#E63946", "fridge": "#06A77D",
                 "washing_machine": "#1D7DBC", "dish_washer": "#8338EC"}
ON_THRESHOLDS = {"kettle": 2000, "fridge": 50, "washing_machine": 50, "dish_washer": 100}
MAX_POWER     = {"kettle": 3998, "fridge": 300, "washing_machine": 2500, "dish_washer": 2500}
SAMPLE_PERIOD = 6

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
CARD_PATH  = os.path.join(os.path.dirname(__file__), "model_card.json")
MODELS_OK  = all(os.path.exists(os.path.join(MODEL_DIR, f"nilm_{a}.pt"))
                 for a in APPLIANCES)

# ── load model card ────────────────────────────────────────────────────────────
@st.cache_data
def load_card():
    with open(CARD_PATH) as f:
        return json.load(f)

card = load_card()
CARD_METRICS = card["metrics"]          # real F1 / MAE from UK-DALE House 5

# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    from inference import load_model
    return {a: load_model(os.path.join(MODEL_DIR, f"nilm_{a}.pt")) for a in APPLIANCES}

def run_inference(models, aggregate):
    from inference import predict
    return {a: predict(models[a], aggregate, a) for a in APPLIANCES}

@st.cache_data(show_spinner=False)
def get_demo_data(seed=42):
    from demo_data import generate_demo_trace
    return generate_demo_trace(hours=6, seed=seed)

def parse_uploaded_csv(file):
    df = pd.read_csv(file, header=None)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        st.error("No numeric column found in CSV.")
        st.stop()
    return numeric.iloc[:, 0].to_numpy(dtype=np.float32)

def f1_color(f1):
    if f1 >= 0.70: return "#06A77D"
    if f1 >= 0.50: return "#F4A261"
    return "#E63946"

def f1_label(f1):
    if f1 >= 0.70: return "Good"
    if f1 >= 0.50: return "Fair"
    return "Low"

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NILM Dashboard")
    st.caption(f"{card['model_name']} · v{card['version']}")
    st.divider()

    data_mode = st.radio("Data source", ["Demo trace (6 h)", "Upload CSV"])
    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Aggregate power (W, 6-second intervals, 1 column)",
            type=["csv", "txt"],
        )

    st.divider()
    st.markdown("**Model weights**")
    if MODELS_OK:
        st.success("✅ Weights loaded — running real inference")
    else:
        st.warning("⚠️ No weights found — showing demo predictions")

    st.divider()
    window_minutes = st.slider("Display window (minutes)", 30, 360, 120, step=30)

    st.divider()
    st.markdown("**Architecture**")
    st.markdown(f"""
- Dilated Conv1D × 5 (d=1–16)
- Global context pooling
- `{card['parameters']:,}` parameters / model
- Trained: {card['training_data']}
- Tested: {card['test_data']}
""")


# ── main ───────────────────────────────────────────────────────────────────────
st.title("⚡ Smart Energy Disaggregation — NILM v5")
st.caption(
    f"**{card['architecture']}** · "
    f"Trained on {card['training_data']} · "
    f"Evaluated on {card['test_data']}"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Trained model performance (always visible, real UK-DALE results)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Model Performance on Unseen House (UK-DALE House 5)")
st.caption("Results from training on Houses 1 & 2, evaluated on House 5 — data the model never saw.")

# KPI row — average
avg_f1  = CARD_METRICS["average"]["F1"]
avg_mae = CARD_METRICS["average"]["MAE_W"]

c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1.5])
c0.metric("Avg F1 Score",  f"{avg_f1:.3f}",  help="Macro-average across 4 appliances")
c1.metric("Avg MAE",       f"{avg_mae:.1f} W")
c2.metric("Parameters",    f"{card['parameters']:,}")
c3.metric("Test Protocol", "Cross-house (unseen)", help="Trained on Houses 1+2, tested on House 5")

st.markdown("")

# Per-appliance metric cards
cols = st.columns(4)
for col, app in zip(cols, APPLIANCES):
    m   = CARD_METRICS[app]
    f1  = m["F1"]
    mae = m["MAE_W"]
    col.markdown(
        f"""
        <div style="
            background: #F8F9FA;
            border-left: 5px solid {APP_COLORS[app]};
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 6px;
        ">
            <div style="font-size:1.5rem">{APP_ICONS[app]}</div>
            <div style="font-weight:700; font-size:1rem; color:#2B2D42">
                {APP_LABELS[app]}
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:{f1_color(f1)}">
                {f1:.3f}
            </div>
            <div style="font-size:0.78rem; color:#666; margin-top:2px">
                F1 · <b>{f1_label(f1)}</b>
            </div>
            <div style="font-size:0.85rem; color:#555; margin-top:6px">
                MAE: <b>{mae:.1f} W</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# F1 bar chart vs benchmarks
st.markdown("")
with st.expander("📈 F1 comparison vs published benchmarks", expanded=True):
    apps   = APPLIANCES
    labels = [APP_LABELS[a] for a in apps]
    our_f1 = [CARD_METRICS[a]["F1"] for a in apps]

    fig_bar = go.Figure()
    # benchmark bars
    bench_f1 = [0.73, 0.73, 0.73, 0.73]   # Seq2Point Zhang 2018 (reported avg)
    fig_bar.add_trace(go.Bar(
        name="Seq2Point (Zhang 2018)",
        x=labels, y=bench_f1,
        marker_color="rgba(180,180,180,0.6)",
        text=[f"{v:.2f}" for v in bench_f1],
        textposition="outside",
    ))
    # our bars
    fig_bar.add_trace(go.Bar(
        name="Our FastNILM-v1",
        x=labels, y=our_f1,
        marker_color=[APP_COLORS[a] for a in apps],
        text=[f"{v:.3f}" for v in our_f1],
        textposition="outside",
    ))
    fig_bar.add_hline(y=avg_f1, line_dash="dash", line_color="#333",
                      annotation_text=f"Our avg {avg_f1:.3f}",
                      annotation_position="top left")
    fig_bar.update_layout(
        barmode="group",
        height=340,
        yaxis=dict(range=[0, 1.0], title="F1 Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "Fridge achieves the best F1 (0.727 — always cycling, easy to detect). "
        "Washing machine is hardest (0.356 — variable load profile). "
        "Quick training (15 epochs, 150k samples) vs full training (~0.73 avg) explains the gap."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Live disaggregation demo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔌 Live Disaggregation Demo")

with st.spinner("Preparing data…"):
    if data_mode == "Demo trace (6 h)":
        aggregate, ground_truth = get_demo_data()
        st.info("Showing a synthetic 6-hour household trace. "
                "Upload your own CSV (Watts, 6-second intervals) to disaggregate real data.", icon="ℹ️")
    else:
        if uploaded_file is None:
            st.info("Upload a CSV file to get started.", icon="📂")
            st.stop()
        aggregate   = parse_uploaded_csv(uploaded_file)
        ground_truth = {}

with st.spinner("Running inference…"):
    if MODELS_OK:
        models      = load_all_models()
        predictions = run_inference(models, aggregate)
    else:
        predictions = {a: ground_truth.get(a, np.zeros_like(aggregate)) for a in APPLIANCES}

# clip to display window
n_show   = min(len(aggregate), int(window_minutes * 60 / SAMPLE_PERIOD))
agg_show = aggregate[:n_show]
t_min    = np.arange(n_show) * SAMPLE_PERIOD / 60.0

# quick energy KPIs
total_kwh    = aggregate.sum() * SAMPLE_PERIOD / 3600 / 1000
app_kwh      = {a: predictions[a].sum() * SAMPLE_PERIOD / 3600 / 1000 for a in APPLIANCES}
identified   = sum(app_kwh.values())
unaccounted  = max(0.0, total_kwh - identified)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total consumption",   f"{total_kwh:.2f} kWh")
k2.metric("Identified",          f"{identified:.2f} kWh",
          f"{100*identified/max(total_kwh,0.001):.0f}% of total")
k3.metric("Peak aggregate",      f"{aggregate.max():.0f} W")
k4.metric("Trace duration",      f"{len(aggregate)*SAMPLE_PERIOD/3600:.1f} h")

# disaggregation chart
n_rows = len(APPLIANCES) + 1
fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[2.0] + [1.0] * len(APPLIANCES),
    subplot_titles=["Aggregate"] + [
        f"{APP_ICONS[a]} {APP_LABELS[a]}  (F1 = {CARD_METRICS[a]['F1']:.3f})"
        for a in APPLIANCES
    ],
)

fig.add_trace(go.Scatter(
    x=t_min, y=agg_show,
    fill="tozeroy", fillcolor="rgba(43,45,66,0.12)",
    line=dict(color="#2B2D42", width=1.5), name="Aggregate",
), row=1, col=1)

for i, app in enumerate(APPLIANCES):
    pred_show = predictions[app][:n_show]
    c = APP_COLORS[app]
    if ground_truth:
        true_show = ground_truth.get(app, np.zeros(n_show))[:n_show]
        r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=t_min, y=true_show,
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(color=c, width=0),
            name=f"{APP_LABELS[app]} true", showlegend=False,
        ), row=i+2, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=pred_show,
        line=dict(color=c, width=2),
        name=APP_LABELS[app],
    ), row=i+2, col=1)
    fig.add_hline(y=ON_THRESHOLDS[app], line_dash="dot",
                  line_color="black", line_width=0.7, opacity=0.4,
                  row=i+2, col=1)

fig.update_layout(
    height=190 * n_rows,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=60, b=40),
    hovermode="x unified",
    plot_bgcolor="white", paper_bgcolor="white",
)
fig.update_xaxes(title_text="Time (minutes)", row=n_rows, col=1)
for row in range(1, n_rows+1):
    fig.update_yaxes(title_text="Watts", row=row, col=1, gridcolor="#eeeeee")

st.plotly_chart(fig, use_container_width=True)

# energy pie + download
st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Energy breakdown")
    fig_pie = go.Figure(go.Pie(
        labels=[APP_LABELS[a] for a in APPLIANCES] + ["Other"],
        values=[app_kwh[a] for a in APPLIANCES] + [unaccounted],
        marker=dict(colors=[APP_COLORS[a] for a in APPLIANCES] + ["#CCCCCC"]),
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.3f} kWh<extra></extra>",
    ))
    fig_pie.update_layout(
        height=320, showlegend=False,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader("Per-appliance summary")
    rows = []
    for a in APPLIANCES:
        on_pct = (predictions[a] > ON_THRESHOLDS[a]).mean() * 100
        rows.append({
            "Appliance":    APP_LABELS[a],
            "Energy (kWh)": round(app_kwh[a], 3),
            "Peak (W)":     int(predictions[a].max()),
            "ON time (%)":  round(on_pct, 1),
            "F1 (test)":    CARD_METRICS[a]["F1"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("Appliance"), use_container_width=True)

    st.subheader("Download predictions")
    df_out = pd.DataFrame({"time_min": t_min, "aggregate_W": agg_show})
    for a in APPLIANCES:
        df_out[f"{a}_pred_W"] = predictions[a][:n_show]
    st.download_button(
        "⬇ Download CSV",
        data=df_out.to_csv(index=False).encode(),
        file_name="nilm_predictions.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model card
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📋 Full Model Card"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
**Model name:** {card['model_name']}
**Version:** {card['version']}
**Date trained:** {card['date_trained']}
**Architecture:** {card['architecture']}
**Parameters:** {card['parameters']:,} per appliance
**Window size:** {card['window_size']} timesteps × 6 s = {card['window_size']*6//60} min
""")
    with col_b:
        st.markdown(f"""
**Training data:** {card['training_data']}
**Test data:** {card['test_data']}
**Evaluation:** Cross-house (unseen house protocol)
**Avg F1:** {avg_f1:.3f} &nbsp;&nbsp; **Avg MAE:** {avg_mae:.1f} W
""")

    st.markdown("**Per-appliance results (UK-DALE House 5 unseen):**")
    df_card = pd.DataFrame([
        {
            "Appliance": APP_LABELS[a],
            "F1 Score":  CARD_METRICS[a]["F1"],
            "MAE (W)":   CARD_METRICS[a]["MAE_W"],
            "Rating":    f1_label(CARD_METRICS[a]["F1"]),
        }
        for a in APPLIANCES
    ] + [{
        "Appliance": "**Average**",
        "F1 Score":  avg_f1,
        "MAE (W)":   avg_mae,
        "Rating":    f1_label(avg_f1),
    }]).set_index("Appliance")
    st.dataframe(df_card, use_container_width=True)

    st.markdown("**References:**")
    for ref in card["references"]:
        st.markdown(f"- {ref}")
