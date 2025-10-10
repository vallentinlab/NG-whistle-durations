# ---------------------------------
# PITCH ANALYSIS (adapted)
# ---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Polygon
from matplotlib.colors import to_hex
import matplotlib as mpl
import json
#------------------------------------------------------------------------------
# PATHS
#------------------------------------------------------------------------------
DATA_PKL = r"Control prev season/all_birds.pkl"

#------------------------------------------------------------------------------
# LOAD BIRDS DATA (from previous season)
#------------------------------------------------------------------------------
all_birds = pd.read_pickle(DATA_PKL)

#------------------------------------------------------------------------------
# CONTROL DATA (exclude playback) + export (optional)
#------------------------------------------------------------------------------
control = all_birds[all_birds.phase != 'playback'].copy()

# Build per-song summaries (control only)
whistle_songs_control = []
for bird in control.bird.unique():
    current_bird = control[control.bird == bird]
    for phase_id in current_bird.phase.unique():
        if phase_id == 'playback':
            continue
        current_phase = current_bird[current_bird.phase == phase_id]
        for set_id in current_phase.set.unique():
            current_set = current_phase[current_phase.set == set_id]
            for snippet in current_set.snippet_idx.unique():
                current_song = current_set[current_set.snippet_idx == snippet]
                range_duration = current_song['duration'].max() - current_song['duration'].min()
                average_duration = current_song['duration'].mean()
                average_freq = current_song['pitch_whistles'].mean()
                duration_last_whistle = current_song['duration'].iloc[-1]
                duration_first_whistle = current_song['duration'].iloc[0]
                median_duration = current_song['duration'].median()
                median_freq = current_song['pitch_whistles'].median()
                freq_last_whistle = current_song['pitch_whistles'].iloc[-1]
                freq_first_whistle = current_song['pitch_whistles'].iloc[0]
                d_first_to_last  = duration_first_whistle - duration_last_whistle
                f_first_to_last  = freq_first_whistle - freq_last_whistle
                n_syl = len(current_song['duration'])

                if n_syl > 1:
                    last_int = current_song['interval'].iloc[-2]
                    first_int = current_song['interval'].iloc[0]
                    interval_first_to_last  = first_int - last_int
                    last_gap = current_song['gap'].iloc[-2]
                    first_gap = current_song['gap'].iloc[0]
                    gap_first_to_last  = first_gap - last_gap
                else:
                    last_int = np.nan
                    first_int = np.nan
                    interval_first_to_last  = np.nan
                    last_gap = np.nan
                    first_gap = np.nan
                    gap_first_to_last  = np.nan

                median_gap = current_song['gap'].median()
                median_int = current_song['interval'].median()

                whistle_songs_control.append({
                    'bird': bird, 'song': snippet,
                    'last_d': duration_last_whistle, 'last_gap': last_gap, 'last_int': last_int,
                    'first_gap': first_gap, 'first_int': first_int,
                    'last_f': freq_last_whistle, 'f_average': average_freq, 'd_average': average_duration,
                    'd_median': median_duration, 'f_median': median_freq, 'n_syl': n_syl,
                    'gap_median': median_gap, 'int_median': median_int,
                    'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'd_range': range_duration,
                    'int_first_to_last': interval_first_to_last, 'gap_first_to_last': gap_first_to_last,
                    'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last
                })

whistle_songs_control = pd.DataFrame(whistle_songs_control)
# ---------- Prep data ----------
df = control[["bird", "pitch_whistles"]].copy()
df = df[np.isfinite(df["pitch_whistles"])].dropna()
birds = np.sort(df["bird"].unique())
assert len(birds) > 0, "No birds found in `control`."

# Global x-range and binning so all subplots are comparable
pmin = df["pitch_whistles"].min()
pmax = df["pitch_whistles"].max()
pad = 0.05 * (pmax - pmin) if pmax > pmin else 1.0
xmin, xmax = pmin - pad, pmax + pad

N_BINS = 40
BIN_EDGES = np.linspace(xmin, xmax, N_BINS + 1)
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

# Helper: set μ-axis ticks the way you like
def apply_pitch_ticks(ax, lo=None, hi=None):
    if lo is None or hi is None:
        cur_lo, cur_hi = ax.get_xlim()
        lo = cur_lo if lo is None else lo
        hi = cur_hi if hi is None else hi
    ax.set_xlim(lo, hi)
    major_ticks = [1000, 2000, 3000, 4000, 5000, 7000, 9000]
    major_ticks = [t for t in major_ticks if lo <= t <= hi]
    ax.set_xticks(major_ticks)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(500))
    ax.tick_params(axis='x', which='major', length=6, width=1.2, direction='out')
    ax.tick_params(axis='x', which='minor', length=3, width=0.8, direction='out')
    ax.set_xlabel("Component μ (Hz)")

# ============================================================
# GMM on per-song median pitch (f_median) from whistle_songs_control
# → global baseline weights W_GLOBAL (left→right) — no rug plot
# ============================================================
# ===============================
# FULL PLOTTING CODE (kHz units)
# ===============================
HZ_TO_KHZ = 1e-3  # convert Hz → kHz

assert 'whistle_songs_control' in globals(), "whistle_songs_control not found."

# --- data in kHz ---
fmed = whistle_songs_control['f_median'].to_numpy(float)
fmed = fmed[np.isfinite(fmed)]
assert fmed.size > 0, "No valid f_median values found."
fmed_khz = fmed * HZ_TO_KHZ  # now in kHz

# --- config ---
K_FIXED = 10
RANDOM_STATE = 0
N_INIT = 50
REG_COVAR = 1e-6
N_BINS = 40

# --- fit GMM on f_median in kHz ---
K_use = int(min(K_FIXED, max(1, fmed_khz.size)))
gm_f = GaussianMixture(
    n_components=K_use, covariance_type="full",
    random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR
).fit(fmed_khz.reshape(-1, 1))

means = gm_f.means_.ravel()                 # kHz
cov   = gm_f.covariances_
if cov.ndim == 3:
    sigmas = np.sqrt(cov[:, 0, 0])          # kHz
elif cov.ndim == 2:
    sigmas = np.sqrt(np.repeat(cov[0, 0], gm_f.n_components))
else:
    sigmas = np.sqrt(np.array(cov).reshape(-1))
weights = gm_f.weights_.ravel()

# sort left→right
order = np.argsort(means)
means, sigmas, weights = means[order], sigmas[order], weights[order]
W_GLOBAL = weights / weights.sum()

# keep original→sorted map for coloring points by sorted modes
means_orig = gm_f.means_.ravel()
order2 = np.argsort(means_orig)
inv_order = np.empty_like(order2)
inv_order[order2] = np.arange(order2.size)

# x-range (kHz) and binning
span = fmed_khz.max() - fmed_khz.min()
pad = 0.05 * (span if span > 0 else 1.0)
xmin, xmax = fmed_khz.min() - pad, fmed_khz.max() + pad
xx = np.linspace(xmin, xmax, 1400)

BIN_WIDTH = (xmax - xmin) / N_BINS

def normal_pdf_1d(xv, m, s):
    return (1.0 / (np.sqrt(2*np.pi) * s)) * np.exp(-0.5 * ((xv - m)/s)**2)

# total mixture pdf (already in kHz)
mix_pdf  = np.exp(gm_f.score_samples(xx.reshape(-1, 1)))
mix_prob = mix_pdf * BIN_WIDTH

# colors per (sorted) component
K_use = gm_f.n_components
magma = cm.get_cmap("magma")
colors = magma(np.linspace(0, 1, K_use))

# ---------------------------------
# Figure: Overall bird pitch distribution (kHz)
# ---------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4, 4),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# --- top: filled components + mixture ---
for k in range(K_use):
    comp_pdf  = W_GLOBAL[k] * normal_pdf_1d(xx, means[k], sigmas[k])  # pdf in kHz
    comp_prob = comp_pdf * BIN_WIDTH
    ax_top.fill_between(xx, comp_prob, 0, color=colors[k], alpha=0.9)

ax_top.plot(xx, mix_prob, color="black", lw=2, label="Mixture")
ax_top.set_ylabel("Probability")
ax_top.set_ylim(0, np.nanmax(mix_prob) * 1.12 if np.isfinite(np.nanmax(mix_prob)) else 1)
#ax_top.set_xscale('log')

# --- bottom: jittered dots (kHz), colored by GMM mode (sorted) ---
labels_raw = gm_f.predict(fmed_khz.reshape(-1, 1))
labels_sorted = inv_order[labels_raw]
pt_colors = colors[labels_sorted]

rng = np.random.default_rng(RANDOM_STATE)
y = rng.uniform(0.06, 0.16, size=fmed_khz.size)  # tiny band

ax_bot.scatter(fmed_khz, y, s=5, c=pt_colors, alpha=1, edgecolors='none')
ax_bot.set_ylim(0, 0.22)
ax_bot.set_yticks([])
ax_bot.set_xlabel("Whistle pitch (kHz)")
#ax_bot.set_xscale('log')

# --- stars at component means (top), colored per mode ---
ymax = ax_top.get_ylim()[1]
y_star = ymax * 0.96
ax_top.scatter(
    means,
    np.full(means.shape, y_star, dtype=float),
    marker='v',
    s=90,
    c=colors,
    edgecolors='black',
    linewidths=0.6,
    zorder=6,
    clip_on=False
)

# --- clean log-kHz ticks (1–9 kHz) for both subplots ---
xlo, xhi = 1, 9  # kHz
ticks = [1, 2, 3, 4, 5, 7, 9]
for a in (ax_top, ax_bot):
    #a.set_xscale('log')
    a.set_xlim(xlo, xhi)
    a.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    a.xaxis.set_minor_locator(mticker.NullLocator())
    a.xaxis.set_minor_formatter(mticker.NullFormatter())
    # integer labels: 1, 2, 3, ...
    a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x)}"))

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("gmm_f_median_k10_with_dots_kHz.pdf",transparent=True, dpi=300, bbox_inches="tight")
plt.show()
print("Saved PDF: gmm_f_median_k10_with_dots_kHz.pdf")

# --- tidy table (kHz) ---
weights_df = pd.DataFrame({
    "mode_L2R": np.arange(1, K_use+1, dtype=int),
    "mu_khz": means,
    "sigma_khz": sigmas,
    "weight": W_GLOBAL
})
print("\nGlobal weights from f_median GMM (left→right), kHz:")
print(weights_df.round(3).to_string(index=False))

# ---------------------------------
# Figure 2: ALL birds — 3 columns per row, ordered by data amount (desc),
# same layout & styling as Figure 1, BUT bottom scatters show INDIVIDUAL WHISTLES
# ---------------------------------
# Order birds by how much data they have (number of pitch_whistles rows per bird)
bird_counts = (
    df.groupby('bird')['pitch_whistles']
      .size()
      .sort_values(ascending=False)
)
birds_sorted = bird_counts.index.to_list()

COLS = 3
N = len(birds_sorted)
ROWS = int(np.ceil(N / COLS))

fig, axes = plt.subplots(
    2*ROWS, COLS, figsize=(4.8*COLS, 4*ROWS),
    sharex=True, sharey='row',
    gridspec_kw={"height_ratios": [3,1]*ROWS}
)
axes = np.atleast_2d(axes)

xlo, xhi = 1000, 9000
ticks = [1000, 2000, 3000, 4000, 5000, 7000, 9000]

# reset and (re)build components for Figure 3
all_comps = []

top_peak_global = 0.0  # unify top-row y across ALL birds
rng = np.random.default_rng(RANDOM_STATE)

for i, bird in enumerate(birds_sorted):
    r = i // COLS  # grid row (of bird panels)
    c = i %  COLS  # grid col
    ax_top = axes[2*r,   c]
    ax_bot = axes[2*r+1, c]

    # --- data for this bird (INDIVIDUAL WHISTLES) ---
    x = df.loc[df["bird"] == bird, "pitch_whistles"].to_numpy()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        ax_top.text(0.5, 0.5, f"No data for {bird}", ha='center', va='center', transform=ax_top.transAxes)
        ax_bot.set_visible(False)
        continue

    # --- fit GMM (fixed K up to 10) on individual whistles ---
    K_use_b = int(min(K_FIXED, max(1, n)))
    X = x.reshape(-1, 1)
    gm_b = GaussianMixture(
        n_components=K_use_b, covariance_type="full",
        random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR
    ).fit(X)

    # params + left→right ordering
    means_b = gm_b.means_.ravel()
    cov_b   = gm_b.covariances_
    if cov_b.ndim == 3:
        sigmas_b = np.sqrt(cov_b[:, 0, 0])
    elif cov_b.ndim == 2:
        sigmas_b = np.sqrt(np.repeat(cov_b[0, 0], gm_b.n_components))
    else:
        sigmas_b = np.sqrt(np.array(cov_b).reshape(-1))
    weights_b = gm_b.weights_.ravel()

    order = np.argsort(means_b)
    inv_order = np.empty_like(order); inv_order[order] = np.arange(order.size)

    means_b, sigmas_b, weights_b = means_b[order], sigmas_b[order], weights_b[order]
    W_b = weights_b / weights_b.sum()

    # --- store for Figure 3 (ALL birds) ---
    for rank, (mu_k, sg_k, w_k) in enumerate(zip(means_b, sigmas_b, weights_b), start=1):
        all_comps.append({
            "bird": bird,
            "mu": float(mu_k),
            "sigma": float(sg_k),
            "weight": float(w_k),
            "rank_L2R": int(rank),
            "K": int(gm_b.n_components)
        })

    # --- grid & mixture → prob per bin (use global BIN_WIDTH for comparability)
    xx = np.linspace(xlo, xhi, 1400)
    mix_pdf_b  = np.exp(gm_b.score_samples(xx.reshape(-1, 1)))
    mix_prob_b = mix_pdf_b * BIN_WIDTH

    colors_b = magma(np.linspace(0, 1, K_use_b))

    # --- TOP: filled components + mixture ---
    comp_peak = 0.0
    for k in range(K_use_b):
        comp_pdf_b  = W_b[k] * (1.0 / (np.sqrt(2*np.pi) * sigmas_b[k])) * np.exp(-0.5 * ((xx - means_b[k]) / sigmas_b[k])**2)
        comp_prob_b = comp_pdf_b * BIN_WIDTH
        ax_top.fill_between(xx, comp_prob_b, 0, color=colors_b[k], alpha=0.9)
        comp_peak = max(comp_peak, float(np.nanmax(comp_prob_b)))

    ax_top.plot(xx, mix_prob_b, color="black", lw=2)
    top_peak_global = max(top_peak_global, float(np.nanmax(mix_prob_b)), comp_peak)

    # stash mean markers/colors; draw triangles after setting shared y
    ax_top._mu_markers = means_b
    ax_top._mu_colors  = colors_b

    # --- BOTTOM: jittered INDIVIDUAL whistles, colored by per-whistle GMM label ---
    labels_raw_b = gm_b.predict(X)              # labels in original component indexing
    labels_sorted_b = inv_order[labels_raw_b]   # map to left→right order
    pt_colors_b = colors_b[labels_sorted_b]

    y_jit = rng.uniform(0.06, 0.16, size=n)
    ax_bot.scatter(x, y_jit, s=12, c=pt_colors_b, alpha=0.6, edgecolors='none')

    ax_bot.set_yticks([])
    ax_bot.set_xlabel("Whistle pitch (Hz)")

    # --- x ticks & formatting (plain numbers on log-x, no minors) ---
    for a in (ax_top, ax_bot):
        a.set_xscale('log')
        a.set_xlim(xlo, xhi)
        a.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        a.xaxis.set_minor_locator(mticker.NullLocator())
        a.xaxis.set_minor_formatter(mticker.NullFormatter())
        a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{int(v)}"))

    # small label with panel index (keeps same visual format)
    ax_top.text(0.02, 0.8, "Bird " + str(i+1), transform=ax_top.transAxes, fontsize=10, va='top', ha='left')

# Hide any unused grid slots (for 11 birds with 3 cols, 12 slots -> 1 hidden)
total_slots = ROWS * COLS
for j in range(N, total_slots):
    r = j // COLS
    c = j %  COLS
    axes[2*r,   c].axis('off')
    axes[2*r+1, c].axis('off')

# --- unify y across rows (sharey='row') ---
# TOP rows: use global peak
y_top = top_peak_global * 1.12 if top_peak_global > 0 else 1
for r in range(ROWS):
    for c in range(COLS):
        ax = axes[2*r, c]
        if hasattr(ax, "_mu_markers"):
            ax.set_ylim(0, y_top)

# Add triangles now that y is final (so they sit just under the top)
for r in range(ROWS):
    for c in range(COLS):
        ax = axes[2*r, c]
        if hasattr(ax, "_mu_markers"):
            y_star = ax.get_ylim()[1] * 0.96
            ax.scatter(
                ax._mu_markers,
                np.full_like(ax._mu_markers, y_star, dtype=float),
                marker='v', s=90, c=ax._mu_colors,
                edgecolors='black', linewidths=0.6, zorder=6, clip_on=False
            )

# BOTTOM rows: fixed shared band for jitter
for r in range(ROWS):
    for c in range(COLS):
        ax = axes[2*r+1, c]
        if ax.axes:  # skip hidden
            ax.set_ylim(0, 0.22)

# y-label (show only once in the top-left panel)
axes[0, 0].set_ylabel("Probability")

mpl.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig("pitch_gmm_k10_ALL_3cols_sorted_whistles_like_fig1_sharedY.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("Saved PDF: pitch_gmm_k10_ALL_3cols_sorted_whistles_like_fig1_sharedY.pdf")

# ---------------------------------
# Figure 3: μ–σ (Hz) scatter + DIAMOND "cones" per MODE (no clustering)
# ---------------------------------
# Build components DataFrame from the per-bird loop above
comp_df = pd.DataFrame(all_comps).copy()
assert {"mu", "sigma", "rank_L2R"} <= set(comp_df.columns), "comp_df must have columns: mu, sigma, rank_L2R"

# --- Config / outputs ---
PITCH_CONES_PDF   = "pitch_mu_sigma_cones_kHz.pdf"
CONE_STATS_CSV    = "cone_stats_kHz.csv"
CONE_VERTS_CSV    = "cone_vertices_kHz.csv"
CONE_JSON         = "cone_polygons_kHz.json"
MAX_MODES         = 10  # ranks 1..10

# --- Units: convert to kHz ---
HZ_TO_KHZ = 1e-3
comp_df["mu_khz"]    = comp_df["mu"].astype(float) * HZ_TO_KHZ
comp_df["sigma_khz"] = comp_df["sigma"].astype(float) * HZ_TO_KHZ

# --- Colors: consistent magma color per mode rank (1..MAX_MODES) ---
magma = cm.get_cmap("magma")
present_max_rank = int(min(MAX_MODES, comp_df["rank_L2R"].max()))
rank_palette = {r: magma(np.linspace(0, 1, MAX_MODES)[r-1]) for r in range(1, present_max_rank+1)}
rank_palette_hex = {r: to_hex(rank_palette[r]) for r in rank_palette}

# --- Scatter colored by mode rank ---
fig, ax = plt.subplots(figsize=(4, 4))
pt_colors = [rank_palette.get(int(r), magma(0.5)) for r in comp_df["rank_L2R"]]
ax.scatter(comp_df["mu_khz"], comp_df["sigma_khz"], s=40, c=pt_colors, alpha=0.95, edgecolors="gray")

# --- Per-mode quartiles and diamond polygons (in kHz) ---
rank_stats = (
    comp_df
    .groupby("rank_L2R", observed=True)
    .agg(
        mu_q25_khz=("mu_khz", lambda v: np.nanpercentile(v, 25)),
        mu_med_khz=("mu_khz", "median"),
        mu_q75_khz=("mu_khz", lambda v: np.nanpercentile(v, 75)),
        sg_q25_khz=("sigma_khz", lambda v: np.nanpercentile(v, 25)),
        sg_med_khz=("sigma_khz", "median"),
        sg_q75_khz=("sigma_khz", lambda v: np.nanpercentile(v, 75)),
        n=("mu_khz", "size"),
    )
    .reset_index()
)

# --- Figure 3: quick summary readouts (no backslashes anywhere) ---

# 1) How many birds?
has_bird = "bird" in comp_df.columns
if has_bird:
    n_birds = comp_df["bird"].nunique()
    print(f"[Fig3] Birds: {n_birds}")
else:
    print("[Fig3] Birds: N/A (no 'bird' column in comp_df)")

# 2) How many points (components)?
n_points = len(comp_df)
print(f"[Fig3] Total components (points): {n_points}")

# 3) Triangle positions (μ medians by rank, in kHz) + diamond centers
triangles_df = (
    rank_stats[["rank_L2R", "mu_med_khz", "sg_med_khz", "n"]]
    .sort_values("rank_L2R")
    .reset_index(drop=True)
)

print("\n[Fig3] Triangles — median μ per rank (kHz):")
for _, r in triangles_df.iterrows():
    print(f"  rank {int(r.rank_L2R)}: μ̃ = {r.mu_med_khz:.3f} kHz  (n={int(r.n)})")

print("\n[Fig3] Diamond centers (μ̃, σ̃) per rank (kHz):")
for _, r in triangles_df.iterrows():
    print(f"  rank {int(r.rank_L2R)}: (μ̃, σ̃) = ({r.mu_med_khz:.3f}, {r.sg_med_khz:.3f}) kHz")

# Optional CSV for the paper/supp
triangles_out = triangles_df.rename(
    columns={
        "rank_L2R": "rank",
        "mu_med_khz": "mu_median_kHz",
        "sg_med_khz": "sigma_median_kHz",
        "n": "count",
    }
)
triangles_out.to_csv("pitch_triangle_medians_kHz.csv", index=False)
print("Saved CSV: pitch_triangle_medians_kHz.csv")

rank_stats = rank_stats[rank_stats["rank_L2R"].between(1, MAX_MODES)].sort_values("rank_L2R")

# collect median μ positions and colors for the top triangles
mu_markers = []
marker_colors = []

for _, row in rank_stats.iterrows():
    r = int(row["rank_L2R"])
    mu_q25, mu_med, mu_q75 = float(row["mu_q25_khz"]), float(row["mu_med_khz"]), float(row["mu_q75_khz"])
    sg_q25, sg_med, sg_q75 = float(row["sg_q25_khz"]), float(row["sg_med_khz"]), float(row["sg_q75_khz"])
    color = rank_palette[r]

    # Diamond vertices in kHz: (μ_med, σ_q25) → (μ_q25, σ_med) → (μ_med, σ_q75) → (μ_q75, σ_med)
    verts = np.array([
        [mu_med, sg_q25],
        [mu_q25, sg_med],
        [mu_med, sg_q75],
        [mu_q75, sg_med]
    ], dtype=float)

    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor='k', alpha=0.9, lw=1))

    mu_markers.append(mu_med)
    marker_colors.append(color)

# place the downward triangles just under the top of the plot
ax.autoscale()  # make sure limits reflect patches too
ymax = ax.get_ylim()[1]
y_star = ymax * 0.98
ax.scatter(
    np.array(mu_markers, dtype=float),
    np.full(len(mu_markers), y_star, dtype=float),
    marker='v', s=90, c=marker_colors,
    edgecolors='black', linewidths=0.6, zorder=6, clip_on=False
)

# --- Axes, ticks, labels (plain numbers on log-x, 1–9 kHz) ---
ax.set_xscale('log')
xlo, xhi = 1, 9  # kHz
ticks = [1, 2, 3, 4, 5, 7, 9]
ax.set_xlim([xlo, xhi])
ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x)}"))

ax.set_xlabel("Component μ (kHz)")
ax.set_ylabel("Component σ (kHz)")

# (Legend omitted by design)

plt.tight_layout()
plt.savefig(PITCH_CONES_PDF, dpi=300, bbox_inches="tight")
mpl.rcParams['pdf.fonttype'] = 42
plt.show()
print(f"Saved PDF: {PITCH_CONES_PDF}")

# --- Exports: per-mode stats and diamond vertices (kHz) ---

# 1) Stats CSV (kHz)
rank_stats_out = rank_stats.copy()
rank_stats_out["color"] = rank_stats_out["rank_L2R"].map(rank_palette_hex)
rank_stats_out.to_csv(CONE_STATS_CSV, index=False)
print(f"Saved CSV: {CONE_STATS_CSV}")

# 2) Vertices CSV (kHz) — columns x1,y1,...,x4,y4 per rank
verts_rows = []
for _, row in rank_stats.iterrows():
    r = int(row["rank_L2R"])
    mu_q25, mu_med, mu_q75 = float(row["mu_q25_khz"]), float(row["mu_med_khz"]), float(row["mu_q75_khz"])
    sg_q25, sg_med, sg_q75 = float(row["sg_q25_khz"]), float(row["sg_med_khz"]), float(row["sg_q75_khz"])
    verts = [
        (mu_med, sg_q25),
        (mu_q25, sg_med),
        (mu_med, sg_q75),
        (mu_q75, sg_med),
    ]
    verts_rows.append({
        "rank_L2R": r,
        "x1": verts[0][0], "y1": verts[0][1],
        "x2": verts[1][0], "y2": verts[1][1],
        "x3": verts[2][0], "y3": verts[2][1],
        "x4": verts[3][0], "y4": verts[3][1],
        "color": rank_palette_hex[r],
        "n": int(row["n"]),
    })
verts_df = pd.DataFrame(verts_rows).sort_values("rank_L2R")
verts_df.to_csv(CONE_VERTS_CSV, index=False)
print(f"Saved CSV: {CONE_VERTS_CSV}")

# 3) JSON (kHz) — one entry per rank with verts and color
cone_json = {
    int(r["rank_L2R"]): {
        "verts": [
            [r["x1"], r["y1"]],
            [r["x2"], r["y2"]],
            [r["x3"], r["y3"]],
            [r["x4"], r["y4"]],
        ],
        "color": r["color"],
        "n": int(r["n"]),
    }
    for _, r in verts_df.iterrows()
}

with open(CONE_JSON, "w") as f:
    json.dump(cone_json, f, indent=2)
print(f"Saved JSON: {CONE_JSON}")

# ---------------------------------
# Simulate 7 birds: sample (μ, σ) per MODE from the diamonds and export
# ---------------------------------
SIM_XLSX = "simulated_bird_params_pitch.xlsx"

# Use the same RNG seed as above for reproducibility
rng = np.random.default_rng(RANDOM_STATE if 'RANDOM_STATE' in globals() else 0)

def _tri_area(tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    return abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)) * 0.5

def _sample_in_triangle(A, B, C):
    # Uniform sample in triangle via sqrt trick
    r1, r2 = rng.random(), rng.random()
    t = np.sqrt(r1)
    return (1 - t) * np.array(A) + t * ((1 - r2) * np.array(B) + r2 * np.array(C))

def _sample_in_diamond(row):
    """Uniform sample inside our diamond defined by quartiles/medians."""
    mu_q25, mu_med, mu_q75 = float(row["mu_q25"]), float(row["mu_med"]), float(row["mu_q75"])
    sg_q25, sg_med, sg_q75 = float(row["sg_q25"]), float(row["sg_med"]), float(row["sg_q75"])

    # Diamond vertices (order consistent with the plot)
    V0 = (mu_med, sg_q25)  # bottom
    V1 = (mu_q25, sg_med)  # left
    V2 = (mu_med, sg_q75)  # top
    V3 = (mu_q75, sg_med)  # right

    # Triangulate into two triangles: (V0,V1,V2) and (V0,V2,V3)
    T1 = (V0, V1, V2)
    T2 = (V0, V2, V3)
    A1 = _tri_area(T1)
    A2 = _tri_area(T2)
    Atot = A1 + A2

    if Atot <= 1e-18:
        # Degenerate diamond; fall back to center (μ_med, σ_med)
        return mu_med, sg_med

    if rng.random() < A1 / Atot:
        x, y = _sample_in_triangle(*T1)
    else:
        x, y = _sample_in_triangle(*T2)
    return float(x), float(y)

# Build once: the list of mode ranks (present in rank_stats)
if 'rank_stats' not in globals() or rank_stats.empty:
    raise RuntimeError("rank_stats not found or empty. Run the cones block before simulation.")

present_ranks = rank_stats["rank_L2R"].astype(int).sort_values().tolist()

# Simulate
rows = []
for b in range(1, 8):  # 7 birds
    bird_name = f"SimBird {b}"
    for r in present_ranks:
        row = rank_stats.loc[rank_stats["rank_L2R"] == r].iloc[0]
        mu, sigma = _sample_in_diamond(row)
        rows.append({
            "sim_bird": bird_name,
            "mode_rank": int(r),
            "mu_Hz": mu,
            "sigma_Hz": sigma,
        })

sim_df = pd.DataFrame(rows)
sim_df.to_excel(SIM_XLSX, index=False)
print(sim_df)
print(f"Saved simulated parameters: {SIM_XLSX}")

