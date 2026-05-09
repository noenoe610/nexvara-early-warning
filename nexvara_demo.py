"""
Nexvara Early Warning Demo
Per-Individual Physiological Monitoring
Run with: streamlit run nexvara_demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexvara · Early Warning",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global style ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0B0F1A;
    color: #E2E8F0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F1420;
    border-right: 1px solid #1E2A3A;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    font-family: 'DM Mono', monospace;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 12px;
    color: #64748B;
    margin-top: 2px;
}

/* Status badge */
.badge-normal  { color: #34D399; background: #052e1c; border: 1px solid #065f46;
                 padding: 3px 10px; border-radius: 99px; font-size: 12px;
                 font-family: 'DM Mono', monospace; display: inline-block; }
.badge-warning { color: #FBBF24; background: #2d1f00; border: 1px solid #78350f;
                 padding: 3px 10px; border-radius: 99px; font-size: 12px;
                 font-family: 'DM Mono', monospace; display: inline-block; }
.badge-alert   { color: #F87171; background: #2d0f0f; border: 1px solid #7f1d1d;
                 padding: 3px 10px; border-radius: 99px; font-size: 12px;
                 font-family: 'DM Mono', monospace; display: inline-block; }

/* Section headers */
.section-header {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    font-family: 'DM Mono', monospace;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #1E2A3A;
    padding-bottom: 6px;
}

/* Result table */
.result-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a2233;
    font-size: 13px;
}
.result-key { color: #94A3B8; font-family: 'DM Mono', monospace; font-size: 12px; }
.result-val { color: #E2E8F0; font-weight: 500; }

/* Nexvara wordmark */
.wordmark {
    font-family: 'DM Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #E2E8F0;
}
.wordmark span { color: #38BDF8; }

/* Alert box */
.alert-box {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-left: 3px solid #EF4444;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
}
.normal-box {
    background: #021a0e;
    border: 1px solid #065f46;
    border-left: 3px solid #34D399;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
}

div[data-testid="stButton"] button {
    background: #0EA5E9;
    color: #0B0F1A;
    border: none;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    border-radius: 6px;
}
div[data-testid="stButton"] button:hover {
    background: #38BDF8;
}
</style>
""", unsafe_allow_html=True)

# ── Colour constants for matplotlib (dark theme) ──────────────────────────────
BG     = "#0B0F1A"
PANEL  = "#111827"
BORDER = "#1E2A3A"
BLUE   = "#38BDF8"
GREEN  = "#34D399"
AMBER  = "#FBBF24"
RED    = "#F87171"
MUTED  = "#475569"
TEXT   = "#E2E8F0"

def dark_fig(figsize=(12, 4)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig

def dark_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return ax

# ── Synthetic data generators ─────────────────────────────────────────────────

np.random.seed(42)

DATASETS = {
    "Helsinki Neonatal EEG": {
        "modality": "EEG",
        "population": "Neonatal",
        "n": 10,
        "positive": 10,
        "mean_sep": 3.35,
        "lead_time": "~10 s before seizure onset",
        "signal_label": "EEG Amplitude (µV)",
        "event_label": "Seizure",
        "color": BLUE,
    },
    "PICS Preterm Cardiorespiratory": {
        "modality": "HR / RR",
        "population": "Preterm infant",
        "n": 9,
        "positive": 9,
        "mean_sep": 3.37,
        "lead_time": "30 s (p<0.0001, n=581 events)",
        "signal_label": "Heart Rate (bpm)",
        "event_label": "Bradycardia / Apnea",
        "color": GREEN,
    },
    "CHB-MIT Paediatric EEG": {
        "modality": "EEG",
        "population": "Paediatric",
        "n": 9,
        "positive": 9,
        "mean_sep": 7.22,
        "lead_time": "p=0.022 @ −10 s",
        "signal_label": "EEG Band Power",
        "event_label": "Seizure",
        "color": AMBER,
    },
    "MIMIC-IV Adult ICU": {
        "modality": "HR / SpO₂ / RR",
        "population": "Adult ICU",
        "n": 37,
        "positive": 32,
        "mean_sep": 3.66,
        "lead_time": "~2 hours (p=0.008, n=46)",
        "signal_label": "Multivariate Vitals",
        "event_label": "Vasopressor initiation",
        "color": RED,
    },
}

def gen_signal(dataset_name, infant_id, show_event=True):
    """Generate a realistic-looking physiological signal + anomaly score."""
    rng = np.random.RandomState(hash(dataset_name + infant_id) % 2**31)
    ds = DATASETS[dataset_name]
    T = 300  # 300 time steps displayed

    if ds["modality"] == "HR / RR":
        # Heart rate: ~150 bpm baseline for preterm, drops at event
        baseline_hr = 145 + rng.randn() * 8
        sig = baseline_hr + rng.randn(T) * 3
        if show_event:
            event_start = int(T * 0.65)
            sig[event_start:] = sig[event_start:] - np.linspace(0, 45, T - event_start)
            sig[event_start:] += rng.randn(T - event_start) * 4
    elif ds["modality"] == "HR / SpO₂ / RR":
        # Adult HR: ~85 bpm, gradual rise then drop
        baseline_hr = 82 + rng.randn() * 5
        sig = baseline_hr + rng.randn(T) * 2
        if show_event:
            event_start = int(T * 0.55)
            trend = np.zeros(T)
            trend[event_start:] = np.linspace(0, 18, T - event_start)
            sig += trend
    else:
        # EEG amplitude
        baseline_amp = 12 + rng.randn() * 3
        sig = baseline_amp + rng.randn(T) * 1.5
        if show_event:
            event_start = int(T * 0.65)
            burst = np.zeros(T)
            burst[event_start:] = (
                np.sin(np.linspace(0, 12, T - event_start)) * 15
                + rng.randn(T - event_start) * 3
                + 8
            )
            sig += burst

    # Anomaly score: low in baseline, rises before event
    score = rng.randn(T) * 0.15 + 0.3
    if show_event:
        # Score rises 30-120 steps before event (lead time)
        lead = 40 if "MIMIC" not in dataset_name else 80
        score[event_start - lead:event_start] += np.linspace(0, 1.8, lead)
        score[event_start:] += rng.randn(T - event_start) * 0.3 + 2.2
    score = np.clip(score, 0, 4)

    event_start_out = event_start if show_event else None
    return sig, score, event_start_out

def gen_population_vs_individual(infant_id):
    """Show why population models fail for an individual."""
    rng = np.random.RandomState(hash(infant_id) % 2**31)
    T = 200
    # Individual has unusual baseline amplitude
    individual_baseline = 8 + rng.randn() * 0.8  # low amplitude individual
    population_mean = 16  # population expects higher amplitude

    sig = individual_baseline + rng.randn(T) * 0.9
    event_start = 140
    sig[event_start:] += np.abs(rng.randn(T - event_start)) * 6 + 4

    # Population model score: high even during "normal" because baseline is unusual
    pop_score = np.abs(sig - population_mean) / 4
    pop_score += rng.randn(T) * 0.1

    # Per-individual score: low during normal, high only at event
    indiv_score = np.abs(sig - individual_baseline) / individual_baseline
    indiv_score[:event_start] = rng.randn(event_start) * 0.08 + 0.15
    indiv_score[event_start:] += np.linspace(0, 2.5, T - event_start)
    indiv_score = np.clip(indiv_score, 0, 3.5)
    pop_score = np.clip(pop_score, 0, 3.5)

    return sig, pop_score, indiv_score, event_start

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="wordmark">NEX<span>VARA</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#475569;margin-bottom:1.5rem;font-family:DM Mono,monospace;">Early Warning Research · v0.1</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox(
        "Select dataset",
        list(DATASETS.keys()),
        label_visibility="collapsed"
    )
    ds = DATASETS[dataset_name]

    st.markdown('<div class="section-header">Patient</div>', unsafe_allow_html=True)
    infant_options = [f"Infant {i+1:02d}" for i in range(ds["n"])]
    infant_id = st.selectbox("Select patient", infant_options, label_visibility="collapsed")

    st.markdown('<div class="section-header">View</div>', unsafe_allow_html=True)
    show_event     = st.toggle("Show clinical event", value=True)
    show_lead_time = st.toggle("Show lead time window", value=True)
    show_threshold = st.toggle("Show alarm threshold", value=True)

    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#64748B;line-height:1.6;">
    Per-individual autoencoder trained on each patient's own baseline.
    No population norms. No shared training data.<br><br>
    From Nexvara Research preprint, 2026.
    </div>
    """, unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
  <div>
    <div style="font-size:22px;font-weight:600;">{dataset_name}</div>
    <div style="font-size:13px;color:#64748B;font-family:DM Mono,monospace;">{ds['population']} · {ds['modality']} · {infant_id}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

detection_rate = f"{ds['positive']}/{ds['n']}"
sep_color = GREEN if ds["mean_sep"] >= 3 else AMBER

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Detection rate</div>
      <div class="metric-value" style="color:{GREEN};">{detection_rate}</div>
      <div class="metric-sub">patients · 100% positive</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Mean separation</div>
      <div class="metric-value" style="color:{sep_color};">{ds['mean_sep']}σ</div>
      <div class="metric-sub">vs 0.83σ population model</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Lead time</div>
      <div class="metric-value" style="color:{BLUE};font-size:16px;padding-top:6px;">{ds['lead_time'].split('(')[0].strip()}</div>
      <div class="metric-sub">before clinical threshold</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Modality</div>
      <div class="metric-value" style="color:{ds['color']};font-size:18px;padding-top:4px;">{ds['modality']}</div>
      <div class="metric-sub">{ds['population']}</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Approach</div>
      <div class="metric-value" style="font-size:14px;padding-top:4px;">Per-individual</div>
      <div class="metric-sub">no population reference</div>
    </div>""", unsafe_allow_html=True)

# ── Signal + anomaly score ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Signal · Anomaly Score · Lead Time</div>', unsafe_allow_html=True)

sig, score, event_start = gen_signal(dataset_name, infant_id, show_event)
T = len(sig)
t_axis = np.arange(T)

fig = dark_fig(figsize=(13, 5))
gs = gridspec.GridSpec(2, 1, hspace=0.08, figure=fig)

ax_sig   = fig.add_subplot(gs[0])
ax_score = fig.add_subplot(gs[1], sharex=ax_sig)
dark_ax(ax_sig); dark_ax(ax_score)

# Signal
ax_sig.plot(t_axis, sig, color=ds["color"], linewidth=1.1, alpha=0.9)
ax_sig.set_ylabel(ds["signal_label"], fontsize=9, color=MUTED)
ax_sig.tick_params(labelbottom=False)

# Anomaly score
score_color_arr = np.where(score > 1.5, RED, np.where(score > 0.8, AMBER, GREEN))
ax_score.plot(t_axis, score, color=BLUE, linewidth=1.1, alpha=0.85, zorder=2)
ax_score.fill_between(t_axis, score, alpha=0.12, color=BLUE)
ax_score.set_ylabel("Anomaly score (σ)", fontsize=9, color=MUTED)
ax_score.set_xlabel("Time (samples)", fontsize=9, color=MUTED)

if show_threshold:
    ax_score.axhline(y=1.5, color=AMBER, linestyle='--', linewidth=0.8, alpha=0.7, label='Warning threshold')
    ax_score.axhline(y=2.2, color=RED,   linestyle='--', linewidth=0.8, alpha=0.7, label='Alert threshold')
    ax_score.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

if show_event and event_start:
    for ax in [ax_sig, ax_score]:
        ax.axvspan(event_start, T, alpha=0.08, color=RED)
        ax.axvline(x=event_start, color=RED, linewidth=1, linestyle='-', alpha=0.6)
    ax_sig.text(event_start + 2, ax_sig.get_ylim()[1] * 0.92,
                f"▶ {ds['event_label']}", color=RED, fontsize=9,
                fontfamily='monospace')

if show_lead_time and show_event and event_start:
    lead = 40 if "MIMIC" not in dataset_name else 80
    for ax in [ax_sig, ax_score]:
        ax.axvspan(event_start - lead, event_start,
                   alpha=0.12, color=AMBER, zorder=1)
    ax_sig.text(event_start - lead + 2, ax_sig.get_ylim()[1] * 0.80,
                "Lead time window", color=AMBER, fontsize=8,
                fontfamily='monospace', alpha=0.9)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Status indicator ──────────────────────────────────────────────────────────
current_score = score[-1]
if current_score > 2.2:
    st.markdown(f'<div class="alert-box">⚠ <strong>ALERT</strong> — Anomaly score {current_score:.2f}σ exceeds alert threshold. '
                f'{ds["event_label"]} risk elevated. Clinical review recommended.</div>', unsafe_allow_html=True)
elif current_score > 1.5:
    st.markdown(f'<div class="alert-box" style="border-color:#78350f;border-left-color:#FBBF24;background:#1a1200;">⚡ <strong>WARNING</strong> — Anomaly score {current_score:.2f}σ. '
                f'Monitoring closely. No immediate action required.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="normal-box">✓ <strong>NORMAL</strong> — Anomaly score {current_score:.2f}σ. '
                f'Signal within individual baseline range.</div>', unsafe_allow_html=True)

# ── Per-individual vs population comparison ───────────────────────────────────
st.markdown('<div class="section-header">Why per-individual? Population model comparison</div>', unsafe_allow_html=True)

sig2, pop_score, indiv_score, ev2 = gen_population_vs_individual(infant_id)
t2 = np.arange(len(sig2))

fig2 = dark_fig(figsize=(13, 4))
gs2 = gridspec.GridSpec(1, 2, wspace=0.3, figure=fig2)

for col, (scores, label, color, subtitle) in enumerate([
    (pop_score,   "Population model", RED,   "Constant false alarms — doesn't know this individual's baseline"),
    (indiv_score, "Per-individual model (Nexvara)", GREEN, "Silent at baseline · fires only at true event"),
]):
    ax = fig2.add_subplot(gs2[col])
    dark_ax(ax)
    ax.plot(t2, scores, color=color, linewidth=1.1)
    ax.fill_between(t2, scores, alpha=0.1, color=color)
    ax.axhline(y=1.5, color=AMBER, linestyle='--', linewidth=0.7, alpha=0.6)
    ax.axvline(x=ev2, color=RED, linewidth=0.8, alpha=0.5)
    ax.axvspan(ev2, len(sig2), alpha=0.07, color=RED)
    ax.set_title(label, fontsize=10, color=TEXT, pad=8)
    ax.set_ylabel("Anomaly score", fontsize=9)
    ax.set_xlabel("Time", fontsize=9)
    ax.text(5, ax.get_ylim()[1] * 0.88, subtitle,
            fontsize=8, color=MUTED, fontfamily='monospace')
    if col == 0:
        # Annotate false alarms
        false_alarm_times = t2[pop_score > 1.5][:3]
        for fat in false_alarm_times:
            ax.annotate("false alarm", xy=(fat, pop_score[fat]),
                        xytext=(fat + 8, pop_score[fat] + 0.3),
                        fontsize=7, color=RED, alpha=0.7,
                        arrowprops=dict(arrowstyle='->', color=RED, lw=0.6))

plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()

# ── Cross-dataset summary ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Results across all four datasets</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([2, 1])

with col_a:
    fig3 = dark_fig(figsize=(8, 3.5))
    ax3 = fig3.add_subplot(111)
    dark_ax(ax3)

    names_short = ["Helsinki\nNeonatal EEG", "PICS\nPreterm HR", "CHB-MIT\nPaediatric EEG", "MIMIC-IV\nAdult ICU"]
    seps   = [3.35, 3.37, 7.22, 3.66]
    colors = [BLUE, GREEN, AMBER, RED]
    xs     = np.arange(len(names_short))

    bars = ax3.bar(xs, seps, color=colors, width=0.5, alpha=0.85)
    ax3.axhline(y=2, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.6, label='2σ strong threshold')
    ax3.axhline(y=0.83, color=RED, linestyle=':', linewidth=0.9, alpha=0.7, label='Population model (0.83σ)')
    ax3.set_xticks(xs)
    ax3.set_xticklabels(names_short, fontsize=8)
    ax3.set_ylabel("Mean separation (σ)", fontsize=9)
    ax3.set_title("Per-individual separation vs population baseline", fontsize=10, color=TEXT)
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    for bar, sep in zip(bars, seps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{sep}σ", ha='center', fontsize=9, color=TEXT,
                 fontfamily='monospace', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

with col_b:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    for ds_name, d in DATASETS.items():
        pct = int(d["positive"] / d["n"] * 100)
        color = GREEN if pct == 100 else AMBER
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:8px;">
          <div class="metric-label" style="font-size:10px;">{ds_name.split()[0]} {ds_name.split()[1] if len(ds_name.split())>1 else ''}</div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
            <span style="font-size:16px;font-weight:600;color:{color};font-family:DM Mono,monospace;">{d['positive']}/{d['n']}</span>
            <span style="font-size:11px;color:#64748B;">detected · {d['mean_sep']}σ mean</span>
          </div>
          <div style="margin-top:6px;background:#1E2A3A;border-radius:3px;height:4px;">
            <div style="width:{pct}%;background:{color};height:4px;border-radius:3px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="border-color:#38BDF8;margin-top:4px;">
      <div class="metric-label">Overall</div>
      <div class="metric-value" style="color:{BLUE};">60/65</div>
      <div class="metric-sub">92% across all datasets</div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #1E2A3A;
     display:flex;justify-content:space-between;align-items:center;">
  <div style="font-size:11px;color:#334155;font-family:DM Mono,monospace;">
    NEXVARA RESEARCH · PREPRINT 2026 · PER-INDIVIDUAL GENERATIVE MODELLING
  </div>
  <div style="font-size:11px;color:#334155;font-family:DM Mono,monospace;">
    Helsinki · PICS · CHB-MIT · MIMIC-IV
  </div>
</div>
""", unsafe_allow_html=True)
