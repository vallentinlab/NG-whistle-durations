#------------------------------------------------------------------------------
# IMPORT PACKAGES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import json
from scipy.stats import chi2 
from scipy.spatial import ConvexHull
import matplotlib.cm as cm
#------------------------------------------------------------------------------
# PATHS
#------------------------------------------------------------------------------
DATA_PKL = r"Control prev season/all_birds.pkl"
PDF_OUT_1 = "gmm_durations_per_bird.pdf"
PDF_OUT_2 = "scatter_mu_sigma_gmm_color_polygons.pdf"
COMPONENTS_CSV = "per_bird_components_seconds.csv"
COMPONENTS_PKL = "per_bird_components_seconds.pkl"
CLUSTERS_CSV = "gmm2d_clusters_params_seconds.csv"
CLUSTERS_PKL = "gmm2d_clusters_params_seconds.pkl"
BIRD_ANON_MAP_JSON = "bird_anonymization_map.json"

#------------------------------------------------------------------------------
# LOAD BIRDS DATA (from previous season)
#------------------------------------------------------------------------------
all_birds = pd.read_pickle(DATA_PKL)

#------------------------------------------------------------------------------
# CONTROL DATA (exclude playback) + export (optional)
#------------------------------------------------------------------------------
control = all_birds[all_birds.phase != 'playback'].copy()
control.to_excel('control_previous_season.xlsx', index=False)

#------------------------------------------------------------------------------
# Build per-song summaries (control only) – unchanged logic
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# COLORS
#------------------------------------------------------------------------------
FAST_RED   = "dimgrey"
MED_YELLOW = "darkgrey"
SLOW_BLUE  = "gainsboro"
MODE_COLORS = [FAST_RED, MED_YELLOW, SLOW_BLUE]
MODE_LABELS = ["Fast (short)", "Medium", "Slow (long)"]
# Demo categories and x values (same structure as your example)
COLORS = ["#E78652", "#DFA71F", "#C69C6D", "#7E4F25"]
CATEGORIES = [
    ("very-short",   [0.04, 0.06, 0.08]),
    ("medium-short", [0.13, 0.14, 0.15]),
    ("medium-long",  [0.30, 0.31, 0.32]),
    ("very-long",    [0.58, 0.72, 0.86]),
]
#------------------------------------------------------------------------------
# CONFIG
#------------------------------------------------------------------------------
FIG_COLS = 4
MAX_COMPONENTS = 3
RANDOM_STATE = 0
N_INIT = 100
REG_COVAR = 1e-6

#------------------------------------------------------------------------------
# PREP DATA + anonymization map
#------------------------------------------------------------------------------
birds = sorted(control["bird"].dropna().unique().tolist())
bird_label_map = {b: f"Bird {i+1}" for i, b in enumerate(birds)}  # anonymize
# Save anonymization map
with open(BIRD_ANON_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(bird_label_map, f, ensure_ascii=False, indent=2)

all_durations = control["duration"].dropna().values
dmin, dmax = float(np.min(all_durations)), float(np.max(all_durations))
# Choose a global bin width for "probability per bin"
NBINS = 30
BIN_WIDTH = (dmax - dmin) / NBINS
#------------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------------
def normal_pdf(x, mean, sigma):
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    z = (x - mean) / sigma
    return inv * np.exp(-0.5 * z * z)

#------------------------------------------------------------------------------
# GLOBAL GMM on per-song MEDIAN durations (control only)
#------------------------------------------------------------------------------
PDF_OUT_MEDIANS = "gmm_median_duration_all_birds.pdf"

# Collect median durations (seconds) from control summaries
med_vals = whistle_songs_control["d_median"].dropna().to_numpy()
med_vals = med_vals[med_vals > 0]
if med_vals.size == 0:
    raise ValueError("No positive median durations found in control summaries.")

# Bin width based on median-duration range (probability per bin)
mmin, mmax = float(np.min(med_vals)), float(np.max(med_vals))
BIN_WIDTH_MED = (mmax - mmin) / NBINS if mmax > mmin else 1.0

# Fit GMM on LOG of median durations, choose K by BIC (up to MAX_COMPONENTS)
logm_all = np.log(med_vals).reshape(-1, 1)
lowest_bic, best_gmm_med, best_k_med = np.inf, None, None
for k in range(1, MAX_COMPONENTS + 1):
    gmm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR
    ).fit(logm_all)
    bic = gmm.bic(logm_all)
    if bic < lowest_bic:
        lowest_bic, best_gmm_med, best_k_med = bic, gmm, k

# Order components short → long (by mean in log space)
mu_log_med = best_gmm_med.means_.ravel()
sd_log_med = np.sqrt(best_gmm_med.covariances_.ravel())
w_med = best_gmm_med.weights_.ravel()
order_med = np.argsort(mu_log_med)
mu_log_med, sd_log_med, w_med = mu_log_med[order_med], sd_log_med[order_med], w_med[order_med]

# x-grid in seconds for plotting
x_med = np.linspace(max(1e-9, np.quantile(med_vals, 0.005)*0.8),
                    np.quantile(med_vals, 0.995)*1.2, 1200)

# Component and mixture densities in seconds: w_k * N(log x; μ_k, σ_k) / x
comp_density_med = [
    w_med[k] * normal_pdf(np.log(x_med), mu_log_med[k], sd_log_med[k]) / x_med
    for k in range(best_k_med)
]
mix_density_med = np.sum(comp_density_med, axis=0)

# Convert to "probability per (median-duration) bin"
comp_prob_med = [cd * BIN_WIDTH_MED for cd in comp_density_med]
mix_prob_med  = mix_density_med * BIN_WIDTH_MED
# ---------------------------------
# Figure: Overall bird duration distribution (s)
# ---------------------------------
fig_m, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4,3), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# choose colors (reuse your MODE_COLORS if available and long enough)
if 'MODE_COLORS' in globals() and len(MODE_COLORS) >= best_k_med:
    comp_colors = [MODE_COLORS[k] for k in range(best_k_med)]
else:
    comp_colors = cm.get_cmap("magma")(np.linspace(0, 1, best_k_med))

# --- TOP: components + mixture ---
for k in range(best_k_med):
    color = comp_colors[k]
    lbl = MODE_LABELS[k] if ('MODE_LABELS' in globals() and best_k_med == 3 and k < len(MODE_LABELS)) else f"Component {k+1}"
    ax_top.fill_between(x_med, comp_prob_med[k], 0, color=color, alpha=1, linewidth=0)
    ax_top.plot(x_med, comp_prob_med[k], color=color, lw=1.8, label=lbl)

ax_top.plot(x_med, mix_prob_med, color="black", lw=2, label="GMM total")

# Nice axes (top)
ax_top.set_ylim(0, max(float(mix_prob_med.max()), 1e-12) * 1.1)
ax_top.set_ylabel("Probability")

# --- Triangles at the MEAN of each mode (lognormal mean in seconds) ---
# mean_k(sec) = exp(mu_k + 0.5 * sigma_k^2), where mu_k, sigma_k are in log-space
mean_sec = np.exp(mu_log_med + 0.5 * (sd_log_med ** 2))
print(mean_sec)
y_star = ax_top.get_ylim()[1] * 0.95
ax_top.scatter(
    mean_sec,
    np.full_like(mean_sec, y_star, dtype=float),
    marker='v', s=90, c=comp_colors,
    edgecolors='black', linewidths=0.6, zorder=6, clip_on=False
)

# --- BOTTOM: jittered scatter of per-song medians colored by GMM mode ---
# map predicted labels to the left→right (short→long) ordering
inv_order_med = np.empty_like(order_med)
inv_order_med[order_med] = np.arange(best_k_med)

labels_raw    = best_gmm_med.predict(logm_all).ravel()        # labels in original model order
labels_sorted = inv_order_med[labels_raw]                     # remapped to short→long
# --- How many points are scattered? (total + per-mode) ---
n_points = med_vals.size
counts_per_mode = np.bincount(labels_sorted, minlength=best_k_med)
print(f"\n[Scatter] Total points (per-song medians): {n_points}")
for k in range(best_k_med):
    print(f"[Scatter] Mode {k+1} count: {counts_per_mode[k]} "
          f"({counts_per_mode[k] / n_points:.1%})")

# --- Parameters for each mode ---
# mu_log_med, sd_log_med, w_med are already sorted short→long
summary_rows = []
print("\n[GMM modes on log(median duration)]")
for k in range(best_k_med):
    mu_log = float(mu_log_med[k])
    sigma_log = float(sd_log_med[k])
    weight = float(w_med[k])

    # Lognormal transforms to seconds
    median_sec = float(np.exp(mu_log))                          # exp(mu)
    mean_sec   = float(np.exp(mu_log + 0.5 * sigma_log**2))     # exp(mu + 0.5*sigma^2)
    mode_sec   = float(np.exp(mu_log - sigma_log**2))           # exp(mu - sigma^2)
    std_sec    = float(np.sqrt((np.exp(sigma_log**2) - 1.0) *
                               np.exp(2*mu_log + sigma_log**2)))
    q05_sec    = float(np.exp(mu_log - 1.96 * sigma_log))
    q95_sec    = float(np.exp(mu_log + 1.96 * sigma_log))

    print(f"Mode {k+1}:")
    print(f"  μ_log = {mu_log:.6f}, σ_log = {sigma_log:.6f}, weight = {weight:.4f}")
    print(f"  median(s) = {median_sec:.4f}, mean(s) = {mean_sec:.4f}, mode(s) = {mode_sec:.4f}, std(s) = {std_sec:.4f}")
    print(f"  ~95% range in seconds: [{q05_sec:.4f}, {q95_sec:.4f}]")

    summary_rows.append({
        "mode": k+1,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "weight": weight,
        "count": counts_per_mode[k],
        "fraction": counts_per_mode[k] / n_points,
        "median_sec": median_sec,
        "mean_sec": mean_sec,
        "mode_sec": mode_sec,
        "std_sec": std_sec,
        "q05_sec": q05_sec,
        "q95_sec": q95_sec,
    })

# Optional: save a CSV with the mode summaries
gmm_modes_csv = "gmm_median_duration_modes.csv"
pd.DataFrame(summary_rows).to_csv(gmm_modes_csv, index=False)
print(f"\nSaved mode summaries to: {gmm_modes_csv}")

pt_colors     = np.array(comp_colors, dtype=object)[labels_sorted]

rng = np.random.default_rng(RANDOM_STATE if 'RANDOM_STATE' in globals() else 0)
y_jit = rng.uniform(0.0, 0.16, size=med_vals.size)

ax_bot.scatter(
    med_vals, y_jit, s=5, c=pt_colors, alpha=0.8,
    edgecolors='gray', linewidths=0.3, rasterized=True
)

# === NEW: playback durations and median vline, like your earlier figures ===
# Expand y-limits so playback vlines can run from -0.08 to -0.2
ax_bot.set_ylim(-0.25, 0.22)

# Global median of control medians (black vline)
med_global = np.median(med_vals)
ax_bot.vlines(med_global, -0.08, -0.2, color='black', lw=2, zorder=7)

# Playback duration markers per category, using your COLORS and CATEGORIES
if 'CATEGORIES' in globals() and 'COLORS' in globals():
    for (cat_name, xs), color in zip(CATEGORIES, COLORS):
        for t in xs:
            ax_bot.vlines(t, -0.08, -0.2, color=color, lw=2, zorder=6)

# --------------------------------------------------------------------------

ax_bot.set_yticks([])
ax_bot.set_xlabel("Whistle duration (s)")

# tidy shared x
plt.setp(ax_top.get_xticklabels(), visible=False)

fig_m.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig_m.savefig(PDF_OUT_MEDIANS, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_MEDIANS}")

#------------------------------------------------------------------------------
# STORAGE for components (seconds units)
#------------------------------------------------------------------------------
all_components = []

#------------------------------------------------------------------------------
# ORDER BIRDS BY AMOUNT OF DATA (number of individual whistles with duration > 0)
#------------------------------------------------------------------------------
bird_counts = (
    control.loc[control["duration"] > 0]
           .groupby("bird")["duration"]
           .size()
           .sort_values(ascending=False)
)
birds_sorted = bird_counts.index.to_numpy()

#------------------------------------------------------------------------------
# LAYOUT for per-bird panels (Top: filled comps + mixture + triangles at means;
#                             Bottom: jittered INDIVIDUAL whistles colored by mode)
#------------------------------------------------------------------------------
n_birds = len(birds_sorted)
ncols   = min(FIG_COLS, n_birds) if 'FIG_COLS' in globals() else min(3, n_birds)
nrows   = int(np.ceil(n_birds / ncols))

fig, axes = plt.subplots(
    nrows=2*nrows, ncols=ncols, figsize=(4.4*ncols, 3.6*nrows),
    sharex=True, sharey='row',
    gridspec_kw={"height_ratios": [3, 1] * nrows}
)
axes = np.atleast_2d(axes)

top_peak_global = 0.0
rng = np.random.default_rng(RANDOM_STATE if 'RANDOM_STATE' in globals() else 0)

for i, bird in enumerate(birds_sorted):
    r = i // ncols
    c = i %  ncols
    ax_top = axes[2*r,   c]
    ax_bot = axes[2*r+1, c]

    vals = control.loc[(control["bird"] == bird) & (control["duration"] > 0), "duration"].to_numpy(float)
    label = bird_label_map.get(bird, str(bird)) if 'bird_label_map' in globals() else str(bird)

    if vals.size == 0:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_title(label)
        ax_bot.set_visible(False)
        continue

    # --- Fit GMM on LOG-durations, choose K via BIC ---
    logd = np.log(vals).reshape(-1, 1)
    lowest_bic, best_gmm, best_k = np.inf, None, None
    for k in range(1, MAX_COMPONENTS + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR
        ).fit(logd)
        bic = gmm.bic(logd)
        if bic < lowest_bic:
            lowest_bic, best_gmm, best_k = bic, gmm, k

    # --- Params ordered short→long by μ_log ---
    mu_log = best_gmm.means_.ravel()
    sd_log = np.sqrt(best_gmm.covariances_.ravel())
    w      = best_gmm.weights_.ravel()
    order  = np.argsort(mu_log)
    mu_log, sd_log, w = mu_log[order], sd_log[order], w[order]

    # Colors: use MODE_COLORS when best_k==3; otherwise magma gradient
    if 'MODE_COLORS' in globals() and best_k == 3 and len(MODE_COLORS) >= 3:
        comp_colors = [MODE_COLORS[k] for k in range(3)]
    else:
        comp_colors = cm.get_cmap("magma")(np.linspace(0, 1, best_k))

    # --- Densities in seconds: sum_k w_k * N(log x; μ_k, σ_k) / x ---
    x = np.linspace(max(1e-9, np.quantile(vals, 0.005)*0.8),
                    np.quantile(vals, 0.995)*1.2, 1200)
    comp_density = [w[k] * normal_pdf(np.log(x), mu_log[k], sd_log[k]) / x for k in range(best_k)]
    mix_density  = np.sum(comp_density, axis=0)

    # Convert to probability per (global) bin
    comp_prob = [cd * BIN_WIDTH for cd in comp_density]   # uses your global BIN_WIDTH
    mix_prob  =  mix_density * BIN_WIDTH

    # --- TOP: filled components + mixture ---
    comp_peak = 0.0
    for k in range(best_k):
        ax_top.fill_between(x, comp_prob[k], 0, color=comp_colors[k], alpha=0.9, linewidth=0)
        ax_top.plot(x, comp_prob[k], color=comp_colors[k], lw=1.5)
        comp_peak = max(comp_peak, float(np.nanmax(comp_prob[k])))

    ax_top.plot(x, mix_prob, color="black", lw=2)
    top_peak_global = max(top_peak_global, float(np.nanmax(mix_prob)), comp_peak)

    # Triangles at mode MEANS in seconds: E[X]=exp(μ+½σ²)
    mean_sec = np.exp(mu_log + 0.5 * (sd_log ** 2))
    ax_top._mu_markers = mean_sec
    ax_top._mu_colors  = comp_colors

    ax_top.set_title(label, fontsize=10)

    # --- BOTTOM: jittered INDIVIDUAL whistles colored by mode ---
    labels_raw = best_gmm.predict(logd).ravel()             # original component indices
    inv_order  = np.empty_like(order); inv_order[order] = np.arange(best_k)
    labels_sorted = inv_order[labels_raw]
    pt_colors     = np.array(comp_colors, dtype=object)[labels_sorted]

    y_jit = rng.uniform(0.06, 0.16, size=vals.size)
    ax_bot.scatter(vals, y_jit, s=12, c=pt_colors, alpha=0.6, edgecolors='none')

    ax_bot.set_ylim(0, 0.22)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel("Whistle duration (s)")

    # --- Store per-bird components in **seconds** for later use ---
    for idx in range(best_k):
        mu_sec    = np.exp(mu_log[idx])
        sigma_sec = np.exp(mu_log[idx]) * np.sqrt(np.exp(sd_log[idx]**2) - 1)
        all_components.append({
            "bird_label": label,
            "component": idx + 1,
            "mu_sec": float(mu_sec),
            "sigma_sec": float(sigma_sec),
            "weight": float(w[idx]),
            "k": int(best_k)
        })

# Remove any unused slots (both top & bottom of that column)
total_slots = nrows * ncols
for j in range(n_birds, total_slots):
    r = j // ncols
    c = j %  ncols
    axes[2*r,   c].axis('off')
    axes[2*r+1, c].axis('off')

# Axis limits: global (expects dmin/dmax to be defined upstream)
for r in range(nrows):
    for c in range(ncols):
        ax_top = axes[2*r, c]
        ax_bot = axes[2*r+1, c]
        if ax_top.axes:  # skip hidden
            ax_top.set_xlim(dmin, dmax)

# Unify y across ALL top panels, then add triangles just under the top
y_top = top_peak_global * 1.12 if top_peak_global > 0 else 1
for r in range(nrows):
    for c in range(ncols):
        ax_top = axes[2*r, c]
        if hasattr(ax_top, "_mu_markers"):
            ax_top.set_ylim(0, y_top)
            y_star = y_top * 0.94
            ax_top.scatter(
                ax_top._mu_markers,
                np.full_like(ax_top._mu_markers, y_star, dtype=float),
                marker='v', s=60, c=ax_top._mu_colors,
                edgecolors='black', linewidths=0.6, zorder=6, clip_on=False
            )

# Labels
fig.supxlabel("Whistle duration (s)")
fig.supylabel("Probability")

fig.tight_layout()
fig.savefig(PDF_OUT_1, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_1}")


# --- CREATE COMPONENTS DF ---
components_df = pd.DataFrame(all_components)
#------------------------------------------------------------------------------
# μ vs. σ (seconds): cluster directly in (μ, σ) and draw HDR polygons
#------------------------------------------------------------------------------
PDF_OUT_2 = "scatter_mu_sigma_gmm_color_polygons.pdf"   # overwrite if you want
POLYGONS_JSON = "mu_sigma_hdr_polygons.json"
USE_HULL_POLYGONS = True     # True => straight-edged polygons; False => smooth ellipse
OUTER_MASS = 0.90            # outer region mass
INNER_MASS = 0.68            # inner region mass (~1σ)
N_CONTOUR_PTS = 240          # resolution for smooth boundary before hull

# 1) Data for clustering in (μ, σ)
df = components_df.dropna(subset=["mu_sec", "sigma_sec"]).copy()

# 2) Fit 2D GMM directly in (μ, σ)
gmm2d = GaussianMixture(
    n_components=3, covariance_type="full",
    random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR
).fit(df[["mu_sec", "sigma_sec"]])

df["cluster"] = gmm2d.predict(df[["mu_sec", "sigma_sec"]])

# 3) Colors ordered by cluster mean μ (left → right)
cluster_palette = [FAST_RED, MED_YELLOW, SLOW_BLUE]
cluster_order = np.argsort(gmm2d.means_[:, 0])  # order by μ
cluster_color_map = {orig: cluster_palette[rank] for rank, orig in enumerate(cluster_order)}

# 4) HDR helper in (μ, σ). Constant-density contour of 2D Gaussian => ellipse.
def hdr_contour(mean, cov, mass=0.90, n=N_CONTOUR_PTS):
    """
    Return n points on the constant-Mahalanobis contour enclosing `mass`
    for a 2D Gaussian with mean (2,) and cov (2,2), in (μ, σ) space.
    """
    r = np.sqrt(chi2.ppf(mass, df=2))
    vals, vecs = np.linalg.eigh(cov)
    A = vecs @ np.diag(np.sqrt(vals))
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.vstack([np.cos(theta), np.sin(theta)])      # 2×n
    pts = (mean[:, None] + r * (A @ circle)).T              # n×2 in (μ, σ)
    # keep σ positive just in case numerical tails cross 0
    pts[:, 1] = np.clip(pts[:, 1], a_min=1e-12, a_max=None)
    return pts

def to_polygon(points):
    """Convert a smooth closed curve to a straight-edged polygon via convex hull."""
    if not USE_HULL_POLYGONS:
        return points
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]

# 5) Plot μ vs σ with polygons + top-aligned cluster-mean triangles
fig2, ax = plt.subplots(figsize=(4,3))

# --- scatter points by cluster (rasterized for Illustrator speed) ---
for k in range(gmm2d.n_components):
    mask = (df["cluster"] == k)
    if not mask.any():
        continue
    color = cluster_color_map[k]
    ax.scatter(
        df.loc[mask, "mu_sec"], df.loc[mask, "sigma_sec"],
        s=60, alpha=0.85, edgecolor="black", facecolor=color, linewidths=0.5,
        zorder=2, label=f"Cluster {int(k)} (n={int(mask.sum())})",
        rasterized=False  # points rasterized; polygons/triangles stay vector
    )

# --- draw polygons & collect vertices for saving ---
polygons_out = []
for k in range(gmm2d.n_components):
    color = cluster_color_map[k]
    mean = gmm2d.means_[k]         # expects (mu_sec, sigma_sec)
    cov  = gmm2d.covariances_[k]   # 2x2 covariance matrix

    # outer region (e.g., 90% HDR)
    boundary90 = hdr_contour(mean, cov, mass=OUTER_MASS, n=N_CONTOUR_PTS)
    poly90 = to_polygon(boundary90)
    ax.fill(
        poly90[:, 0], poly90[:, 1],
        facecolor=color, edgecolor=color, alpha=0.5, linewidth=1.6, zorder=1
    )
    polygons_out.append({
        "cluster": int(k),
        "mass": float(OUTER_MASS),
        "vertices": poly90.tolist(),
        "color": color
    })

    # inner region (e.g., ~68% HDR ~ 1σ)
    boundary68 = hdr_contour(mean, cov, mass=INNER_MASS, n=N_CONTOUR_PTS)
    poly68 = to_polygon(boundary68)
    ax.fill(
        poly68[:, 0], poly68[:, 1],
        facecolor=color, edgecolor=color, alpha=1.0, linewidth=1.4, zorder=1
    )
    polygons_out.append({
        "cluster": int(k),
        "mass": float(INNER_MASS),
        "vertices": poly68.tolist(),
        "color": color
    })

# --- axes labels before placing triangles (limits are now settled) ---
ax.set_xlabel("μ (s)")
ax.set_ylabel("σ (s)")

# --- cluster means as downward triangles at the SAME top y-level ---
# x = μ_k (cluster mean in seconds), y = constant near top of axis
mean_mu = [gmm2d.means_[k][0] for k in range(gmm2d.n_components)]
mean_colors = [cluster_color_map[k] for k in range(gmm2d.n_components)]

# choose a constant y slightly below the top bound so markers sit inside the frame
y_top = ax.get_ylim()[1]
y_star = y_top * 0.98  # adjust (0.96–0.99) if you want a bit more/less padding

ax.scatter(
    mean_mu, [y_star] * len(mean_mu),
    marker='v', s=120, facecolors=mean_colors,
    edgecolors='black', linewidths=0.8,
    zorder=5, clip_on=False  # clip_off keeps tips visible if very close to the frame
)

fig2.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig2.savefig(PDF_OUT_2, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_2}")


# 6) Save polygons for later use
with open(POLYGONS_JSON, "w", encoding="utf-8") as f:
    json.dump(polygons_out, f, ensure_ascii=False, indent=2)
print(f"Saved polygons JSON: {POLYGONS_JSON}")

#------------------------------------------------------------------------------
# Simulate 7 birds: sample (μ, σ) per mode from the polygon regions and export
#------------------------------------------------------------------------------

SIM_XLSX = "simulated_bird_params_duration.xlsx"

# --- helpers for uniform sampling inside a (convex) polygon -------------------
rng = np.random.default_rng(RANDOM_STATE)

def _tri_area(tri):
    (x1,y1),(x2,y2),(x3,y3) = tri
    return abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) * 0.5

def _tri_sample(tri, rng):
    A, B, C = tri
    r1, r2 = rng.random(), rng.random()
    t = np.sqrt(r1)          # area correction
    return (1-t)*A + t*(1-r2)*B + t*r2*C

def _sample_in_polygon(poly, rng):
    # fan triangulation from the first vertex (valid for convex polygons)
    tris = [np.vstack([poly[0], poly[i], poly[i+1]]) for i in range(1, len(poly)-1)]
    areas = np.array([_tri_area(tr) for tr in tris])
    probs = areas / areas.sum()
    idx = rng.choice(len(tris), p=probs)
    return _tri_sample(tris[idx], rng)

# build dict of OUTER_MASS polygons by cluster id
polygons_outer = {
    int(item["cluster"]): np.array(item["vertices"])
    for item in polygons_out
    if abs(float(item["mass"]) - float(OUTER_MASS)) < 1e-9
}

# simulate
rows = []
for b in range(7):
    sim_bird = f"SimBird {b+1}"
    # use cluster_order so modes are Fast/Medium/Slow left→right
    for rank, orig_cluster in enumerate(cluster_order):
        poly = polygons_outer[orig_cluster]
        mu, sigma = _sample_in_polygon(poly, rng)
        rows.append({
            "sim_bird": sim_bird,
            "mode": MODE_LABELS[rank],
            "mu_sec": float(mu),
            "sigma_sec": float(sigma),
        })

sim_df = pd.DataFrame(rows)
sim_df.to_excel(SIM_XLSX, index=False)
print(sim_df)
print(f"Saved simulated parameters: {SIM_XLSX}")
