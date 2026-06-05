import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Stop Strategy AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), "F1_Model.pkl")
    bundle = joblib.load(path)
    return bundle["Model"], bundle["Categorical_Encoder"], bundle["Numerical_Encoder"]

model, cat_enc, num_enc = load_bundle()

FEATURE_NAMES = list(model.feature_names_in_)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}
COMPOUND_ICONS = {
    "SOFT": "🔴",
    "MEDIUM": "🟡",
    "HARD": "⚪",
    "INTERMEDIATE": "🟢",
    "WET": "🔵",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
    background-color: #0d0d0d;
    color: #ffffff;
  }
  .stApp { background-color: #0d0d0d; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 2px solid #e10600;
  }
  section[data-testid="stSidebar"] * { color: #ffffff !important; }
  section[data-testid="stSidebar"] .stSlider > label,
  section[data-testid="stSidebar"] .stNumberInput > label,
  section[data-testid="stSidebar"] .stSelectbox > label { color: #aaaaaa !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 1px; }

  /* Slider accent */
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: #e10600 !important; }
  .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: #e10600 !important; }

  /* Metric cards */
  .metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-label { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; }
  .metric-sub   { font-size: 0.8rem; color: #666; }

  /* Result banner */
  .result-pit      { color: #e10600; font-size: 2.2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; }
  .result-continue { color: #00d060; font-size: 2.2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; }
  .result-sub      { font-size: 1rem; font-weight: 600; color: #aaaaaa; text-transform: uppercase; letter-spacing: 2px; }
  .confidence      { font-size: 1.8rem; font-weight: 700; }

  /* Insight cards */
  .insight-card {
    background: #1a1a1a;
    border-left: 3px solid #e10600;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
  .insight-title { font-size: 0.85rem; font-weight: 700; color: #ffffff; }
  .insight-desc  { font-size: 0.75rem; color: #888; margin-top: 2px; }

  /* Predict button */
  .stButton > button {
    background: #e10600 !important;
    color: white !important;
    border: none !important;
    font-family: 'Titillium Web', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    padding: 14px !important;
    border-radius: 6px !important;
    width: 100% !important;
    margin-top: 8px;
  }
  .stButton > button:hover { background: #b00500 !important; }

  /* Section headers */
  .section-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: #e10600;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 6px;
    margin-bottom: 14px;
  }

  /* Top header bar */
  .top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 16px 0;
    border-bottom: 2px solid #e10600;
    margin-bottom: 18px;
  }
  .top-title { font-size: 1.9rem; font-weight: 900; text-transform: uppercase; letter-spacing: 3px; }
  .top-title span { color: #e10600; font-style: italic; }
  .top-sub { font-size: 0.7rem; color: #666; letter-spacing: 2px; text-transform: uppercase; }
  .badge { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 4px 12px; font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .badge span { color: #e10600; font-weight: 700; }

  /* Probability pill */
  .prob-label { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1.5px; text-align:center; margin-top: 8px; }
  .prob-value-red   { font-size: 2rem; font-weight: 900; color: #e10600; text-align: center; }
  .prob-value-green { font-size: 2rem; font-weight: 900; color: #00d060; text-align: center; }
  .prob-tag { text-align:center; margin-top:6px; }
  .prob-tag span {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;
    border-radius: 4px; padding: 4px 12px;
  }
  .tag-high   { background: #3a0000; color: #e10600; border: 1px solid #e10600; }
  .tag-mid    { background: #2a2a00; color: #ffcc00; border: 1px solid #ffcc00; }
  .tag-low    { background: #003a10; color: #00d060; border: 1px solid #00d060; }

  div[data-testid="stNumberInput"] input { background: #1a1a1a !important; color: white !important; border: 1px solid #333 !important; border-radius: 6px !important; }
  div[data-testid="stSelectbox"] > div { background: #1a1a1a !important; color: white !important; }
  hr { border-color: #2a2a2a; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
      <div style="font-size:2rem; font-weight:900; letter-spacing:4px; color:#e10600;">F1</div>
      <div style="font-size:0.65rem; color:#666; letter-spacing:2px; text-transform:uppercase;">Race Telemetry Inputs</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎮 Race Telemetry Inputs</div>', unsafe_allow_html=True)

    lap_number    = st.slider("Lap Number",        1,  70,  20)
    stint         = st.slider("Stint",             1,   5,   1)
    tyre_life     = st.slider("Tyre Life (laps)",  1,  50,  10)
    position      = st.slider("Position",          1,  20,   5)
    race_progress = st.slider("Race Progress (%)", 0, 100,  40)

    st.markdown("---")
    lap_time      = st.number_input("Lap Time (s)",           value=90.00, step=0.01, format="%.2f")
    lap_time_delta= st.number_input("Lap Time Delta (s)",     value=0.20,  step=0.01, format="%.2f")
    cum_deg       = st.number_input("Cumulative Degradation", value=0.50,  step=0.01, format="%.2f")
    norm_tyre     = st.number_input("Normalized Tyre Life",   value=0.50,  step=0.01, format="%.2f")
    pos_change    = st.number_input("Position Change",        value=0,     step=1)

    st.markdown("---")
    compound = st.selectbox("Compound", COMPOUNDS, index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  PREDICT PIT STRATEGY")


# ── Top Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
  <div>
    <div class="top-title">F1 PIT STOP STRATEGY <span>AI</span></div>
    <div class="top-sub">ML Powered Pit Stop Prediction System</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
compound_color = COMPOUND_COLORS.get(compound, "#FFF200")
compound_icon  = COMPOUND_ICONS.get(compound, "🟡")

for col, icon, label, val, sub in [
    (c1, "🏁", "CURRENT LAP",    f"{lap_number}", f"/ 70"),
    (c2, "⏱", "RACE PROGRESS",  f"{race_progress}%", ""),
    (c3, "🔄", "TYRE LIFE",      f"{tyre_life}", "LAPS"),
    (c4, "🏆", "POSITION",       f"{position}", f"/ 20"),
    (c5, "🕐", "STINT",          f"{stint}", f"/ 5"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{icon} {label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c6.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{compound_icon} COMPOUND</div>
  <div class="metric-value" style="color:{compound_color};">{compound}</div>
  <div class="metric-sub"></div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict(lap_number, stint, tyre_life, position, lap_time, lap_time_delta,
            cum_deg, race_progress, norm_tyre, pos_change, compound):
    cat_df = pd.DataFrame({"Compound": [compound]})
    num_df = pd.DataFrame({
        "LapNumber": [lap_number], "Stint": [stint], "TyreLife": [tyre_life],
        "Position": [position], "LapTime": [lap_time], "LapTimeDelta": [lap_time_delta],
        "CumulativeDegradation": [cum_deg], "RaceProgress": [race_progress / 100.0],
        "NormalizedTyreLife": [norm_tyre], "PositionChange": [pos_change],
    })
    cat_t = cat_enc.transform(cat_df)
    num_t = num_enc.transform(num_df)
    X = pd.concat([cat_t, num_t], axis=1)[FEATURE_NAMES]
    proba = model.predict_proba(X)[0]

    # 🔥 threshold fix (handles class imbalance)
    threshold = 0.4
    pred = 1 if proba[1] > threshold else 0

    return pred, proba


# ── Run prediction on button or on first load ─────────────────────────────────
if predict_btn or "result" not in st.session_state:
    pred, proba = predict(lap_number, stint, tyre_life, position, lap_time,
                          lap_time_delta, cum_deg, race_progress, norm_tyre,
                          pos_change, compound)
    st.session_state.result = (pred, proba)

pred, proba = st.session_state.result
pit_prob    = proba[1] * 100
conf        = max(proba) * 100
is_pit      = pred == 1


# ── Main body ─────────────────────────────────────────────────────────────────
left_col, mid_col, right_col = st.columns([1.15, 1.05, 1])

# ── LEFT – Prediction result ──────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-header">PREDICTION RESULT</div>', unsafe_allow_html=True)

    result_class = "result-pit" if is_pit else "result-continue"
    result_text  = "🔴 PIT THIS LAP"   if is_pit else "✅ CONTINUE RACE"
    sub_text     = "⚠️ PIT STOP NEXT LAP" if is_pit else "NO PIT STOP NEXT LAP"

    st.markdown(f"""
    <div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px; padding:20px 24px; margin-top:8px;">
      <div class="{result_class}">{result_text}</div>
      <div class="result-sub">{sub_text}</div>
      <div style="margin-top:14px; font-size:0.65rem; color:#666; text-transform:uppercase; letter-spacing:1.5px;">Model Confidence</div>
      <div class="confidence" style="color:{'#e10600' if is_pit else '#00d060'};">{conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)


# ── MID – Gauge + Feature Importance ─────────────────────────────────────────
with mid_col:
    st.markdown('<div class="section-header">PREDICTION PROBABILITY</div>', unsafe_allow_html=True)

    # Gauge chart
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    theta_start, theta_end = np.pi, 0  # 180° → 0°
    n_seg = 100
    thetas = np.linspace(theta_start, theta_end, n_seg + 1)
    r_inner, r_outer = 0.55, 0.85

    for i in range(n_seg):
        t0, t1 = thetas[i], thetas[i + 1]
        frac = i / n_seg
        r = frac  # 0=green, 1=red
        g = 1 - frac
        b = 0
        xs = [r_inner * np.cos(t0), r_outer * np.cos(t0),
              r_outer * np.cos(t1), r_inner * np.cos(t1)]
        ys = [r_inner * np.sin(t0), r_outer * np.sin(t0),
              r_outer * np.sin(t1), r_inner * np.sin(t1)]
        ax.fill(xs, ys, color=(r, g, b), zorder=1)

    needle_angle = np.pi - (pit_prob / 100) * np.pi
    needle_len = 0.70
    ax.plot([0, needle_len * np.cos(needle_angle)],
            [0, needle_len * np.sin(needle_angle)],
            color="white", linewidth=3, zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=6))

    ax.text(-0.88, -0.18, "0%",   color="#aaa", fontsize=8, ha="center", va="center")
    ax.text( 0.88, -0.18, "100%", color="#aaa", fontsize=8, ha="center", va="center")
    ax.text( 0,    0.62,  "50%",  color="#aaa", fontsize=8, ha="center", va="center")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.25, 1.05)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    tag_class = "tag-high" if pit_prob >= 60 else ("tag-mid" if pit_prob >= 35 else "tag-low")
    tag_text  = "HIGH PROBABILITY" if pit_prob >= 60 else ("MEDIUM PROBABILITY" if pit_prob >= 35 else "LOW PROBABILITY")
    prob_class= "prob-value-red" if pit_prob >= 50 else "prob-value-green"

    st.markdown(f"""
    <div class="{prob_class}">{pit_prob:.2f}%</div>
    <div class="prob-label">PIT STOP PROBABILITY</div>
    <div class="prob-tag"><span class="{tag_class}">{tag_text}</span></div>
    """, unsafe_allow_html=True)




# ── RIGHT – Race Insights ─────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-header">RACE INSIGHTS</div>', unsafe_allow_html=True)

    insights = []

    # Tyre condition
    if tyre_life <= 15:
        insights.append(("🔄", "#00d060", "Tyre condition is still optimal.", f"Tyre life is {tyre_life} laps."))
    elif tyre_life <= 30:
        insights.append(("🔄", "#ffcc00", "Tyre wear is increasing.", f"Consider pitting within {50 - tyre_life} laps."))
    else:
        insights.append(("🔄", "#e10600", "Tyres critically worn!", f"Tyre life at {tyre_life} laps — pit soon."))

    # Degradation
    if abs(cum_deg) < 50:
        insights.append(("📈", "#00d060", "Degradation is within acceptable range.", "Keep pushing!"))
    elif abs(cum_deg) < 150:
        insights.append(("📈", "#ffcc00", "Degradation climbing.", "Monitor tyre temperature."))
    else:
        insights.append(("📈", "#e10600", "High cumulative degradation!", "Pit stop strongly advised."))

    # Position
    if position <= 5:
        insights.append(("🏆", "#00d060", "You are in a good position.", "Maintain track position."))
    elif position <= 10:
        insights.append(("🏆", "#ffcc00", "In the points.", "Fight for top 5."))
    else:
        insights.append(("🏆", "#aaaaaa", "Outside top 10.", "An undercut could help."))

    # Race progress
    rp = race_progress
    if rp < 30:
        insights.append(("⏱", "#aaaaaa", f"Race progress is at {rp}%.", "Plenty of laps remaining."))
    elif rp < 65:
        insights.append(("⏱", "#ffcc00", f"Race progress at {rp}%.", "Mid-race strategy window open."))
    else:
        insights.append(("⏱", "#e10600", f"Race progress at {rp}%.", "Final stint — protect position."))

    # Compound info
    comp_tip = {
        "SOFT": "Maximum grip, short stint.",
        "MEDIUM": "Balanced performance.",
        "HARD": "High durability, low grip.",
        "INTERMEDIATE": "Light wet conditions.",
        "WET": "Heavy rain compound.",
    }
    insights.append((compound_icon, compound_color, f"{compound} compound active.", comp_tip.get(compound, "")))

    for icon, color, title, desc in insights:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color};">
          <div class="insight-title">{icon} {title}</div>
          <div class="insight-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Lap time delta insight
    if lap_time_delta > 0.5:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:#e10600;">
          <div class="insight-title">⚠️ Lap time deteriorating.</div>
          <div class="insight-desc">Delta: +{lap_time_delta:.2f}s — tyre drop-off suspected.</div>
        </div>""", unsafe_allow_html=True)
    elif lap_time_delta < -0.2:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:#00d060;">
          <div class="insight-title">⚡ Lap time improving.</div>
          <div class="insight-desc">Delta: {lap_time_delta:.2f}s — push harder!</div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #2a2a2a; padding-top:12px; display:flex; justify-content:space-between; align-items:center;">
  <div style="font-size:0.7rem; color:#555;">
    <span style="color:#e10600; font-weight:700;">F1</span> F1 PIT STOP STRATEGY AI &nbsp;|&nbsp; BUILT WITH PYTHON, SCIKIT-LEARN &amp; STREAMLIT
  </div>
  <div style="font-size:0.7rem; color:#e10600; font-weight:700; letter-spacing:2px;">DATA DRIVEN. STRATEGY PERFECTED.</div>
</div>
""", unsafe_allow_html=True)



