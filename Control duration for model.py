import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from scipy.spatial import ConvexHull
import json

# --- Paths
DATA_PKL = r"Control prev season/all_birds.pkl"
PDF_OUT_MEDIANS = "gmm_median_duration_all_birds.pdf"
PDF_OUT_1 = "gmm_durations_per_bird.pdf"
PDF_OUT_2 = "scatter_mu_sigma_gmm_color_polygons.pdf"
BIRD_ANON_MAP_JSON = "bird_anonymization_map.json"
GMM_MODES_CSV = "gmm_median_duration_modes.csv"
POLYGONS_JSON = "mu_sigma_hdr_polygons.json"
SIM_XLSX = "simulated_bird_params_duration.xlsx"

# --- Config
FIG_COLS = 4
MAX_COMPONENTS = 3
RANDOM_STATE = 0
N_INIT = 100
REG_COVAR = 1e-6
NBINS = 30
OUTER_MASS = 0.90
INNER_MASS = 0.68
N_CONTOUR_PTS = 240
USE_HULL_POLYGONS = True

# --- Colors
FAST_RED   = "dimgrey"
MED_YELLOW = "darkgrey"
SLOW_BLUE  = "gainsboro"
MODE_COLORS = [FAST_RED, MED_YELLOW, SLOW_BLUE]
MODE_LABELS = ["Fast (short)", "Medium", "Slow (long)"]
COLORS = ["#E78652", "#DFA71F", "#C69C6D", "#7E4F25"]
CATEGORIES = [
    ("very-short",   [0.04, 0.06, 0.08]),
    ("medium-short", [0.13, 0.14, 0.15]),
    ("medium-long",  [0.30, 0.31, 0.32]),
    ("very-long",    [0.58, 0.72, 0.86]),
]

mpl.rcParams['pdf.fonttype'] = 42
rng = np.random.default_rng(RANDOM_STATE)

# --- Data
all_birds = pd.read_pickle(DATA_PKL)
control = all_birds[all_birds.phase != 'playback'].copy()
control.to_excel('control_previous_season.xlsx', index=False)

# --- Per-song summaries (control)
whistle_songs_control = []
for bird in control.bird.unique():
    bdf = control[control.bird == bird]
    for phase_id in bdf.phase.unique():
        if phase_id == 'playback':
            continue
        pdf = bdf[bdf.phase == phase_id]
        for set_id in pdf.set.unique():
            sdf = pdf[pdf.set == set_id]
            for snippet in sdf.snippet_idx.unique():
                s = sdf[sdf.snippet_idx == snippet]
                d = s['duration']
                f = s['pitch_whistles']

                n_syl = len(d)
                last_int = s['interval'].iloc[-2] if n_syl > 1 else np.nan
                first_int = s['interval'].iloc[0] if n_syl > 1 else np.nan
                interval_first_to_last = (first_int - last_int) if n_syl > 1 else np.nan
                last_gap = s['gap'].iloc[-2] if n_syl > 1 else np.nan
                first_gap = s['gap'].iloc[0] if n_syl > 1 else np.nan
                gap_first_to_last = (first_gap - last_gap) if n_syl > 1 else np.nan

                whistle_songs_control.append({
                    'bird': bird,
                    'song': snippet,
                    'last_d': d.iloc[-1],
                    'last_gap': last_gap,
                    'last_int': last_int,
                    'first_gap': first_gap,
                    'first_int': first_int,
                    'last_f': f.iloc[-1],
                    'f_average': f.mean(),
                    'd_average': d.mean(),
                    'd_median': d.median(),
                    'f_median': f.median(),
                    'n_syl': n_syl,
                    'gap_median': s['gap'].median(),
                    'int_median': s['interval'].median(),
                    'first_d': d.iloc[0],
                    'first_f': f.iloc[0],
                    'd_range': d.max() - d.min(),
                    'int_first_to_last': interval_first_to_last,
                    'gap_first_to_last': gap_first_to_last,
                    'd_first_to_last': d.iloc[0] - d.iloc[-1],
                    'f_first_to_last': f.iloc[0] - f.iloc[-1],
                })

whistle_songs_control = pd.DataFrame(whistle_songs_control)

# --- Anonymization
birds = sorted(control["bird"].dropna().unique().tolist())
bird_label_map = {b: f"Bird {i+1}" for i, b in enumerate(birds)}
with open(BIRD_ANON_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(bird_label_map, f, ensure_ascii=False, indent=2)

# --- Helpers

def normal_pdf(x, mean, sigma):
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    z = (x - mean) / sigma
    return inv * np.exp(-0.5 * z * z)

# --- Global GMM on per-song median durations
med_vals = whistle_songs_control["d_median"].dropna().to_numpy()
med_vals = med_vals[med_vals > 0]

mmin, mmax = float(np.min(med_vals)), float(np.max(med_vals))
BIN_WIDTH_MED = (mmax - mmin) / NBINS if mmax > mmin else 1.0

logm_all = np.log(med_vals).reshape(-1, 1)
lowest_bic, best_gmm_med, best_k_med = np.inf, None, None
for k in range(1, MAX_COMPONENTS + 1):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        reg_covar=REG_COVAR,
    ).fit(logm_all)
    bic = gmm.bic(logm_all)
    if bic < lowest_bic:
        lowest_bic, best_gmm_med, best_k_med = bic, gmm, k

mu_log_med = best_gmm_med.means_.ravel()
sd_log_med = np.sqrt(best_gmm_med.covariances_.ravel())
w_med = best_gmm_med.weights_.ravel()
order_med = np.argsort(mu_log_med)
mu_log_med, sd_log_med, w_med = (
    mu_log_med[order_med],
    sd_log_med[order_med],
    w_med[order_med],
)

x_med = np.linspace(max(1e-9, np.quantile(med_vals, 0.005) * 0.8),
                    np.quantile(med_vals, 0.995) * 1.2, 1200)
comp_density_med = [
    w_med[k] * normal_pdf(np.log(x_med), mu_log_med[k], sd_log_med[k]) / x_med
    for k in range(best_k_med)
]
mix_density_med = np.sum(comp_density_med, axis=0)

comp_prob_med = [cd * BIN_WIDTH_MED for cd in comp_density_med]
mix_prob_med = mix_density_med * BIN_WIDTH_MED

fig_m, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4, 3), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

comp_colors = (
    [MODE_COLORS[k] for k in range(best_k_med)]
    if best_k_med <= len(MODE_COLORS)
    else cm.get_cmap("magma")(np.linspace(0, 1, best_k_med))
)

for k in range(best_k_med):
    color = comp_colors[k]
    lbl = MODE_LABELS[k] if (best_k_med == 3 and k < len(MODE_LABELS)) else f"Component {k+1}"
    ax_top.fill_between(x_med, comp_prob_med[k], 0, color=color, alpha=1, linewidth=0)
    ax_top.plot(x_med, comp_prob_med[k], color=color, lw=1.8, label=lbl)

ax_top.plot(x_med, mix_prob_med, color="black", lw=2, label="GMM total")
ax_top.set_ylim(0, max(float(mix_prob_med.max()), 1e-12) * 1.1)
ax_top.set_ylabel("Probability")

mean_sec = np.exp(mu_log_med + 0.5 * (sd_log_med ** 2))
y_star = ax_top.get_ylim()[1] * 0.95
ax_top.scatter(
    mean_sec,
    np.full_like(mean_sec, y_star, dtype=float),
    marker='v', s=90, c=comp_colors,
    edgecolors='black', linewidths=0.6, zorder=6, clip_on=False,
)

inv_order_med = np.empty_like(order_med)
inv_order_med[order_med] = np.arange(best_k_med)
labels_sorted = inv_order_med[best_gmm_med.predict(logm_all).ravel()]
pt_colors = np.array(comp_colors, dtype=object)[labels_sorted]

y_jit = rng.uniform(0.0, 0.16, size=med_vals.size)
ax_bot.scatter(
    med_vals, y_jit, s=5, c=pt_colors, alpha=0.8,
    edgecolors='gray', linewidths=0.3, rasterized=True,
)
ax_bot.set_ylim(-0.25, 0.22)

med_global = np.median(med_vals)
ax_bot.vlines(med_global, -0.08, -0.2, color='black', lw=2, zorder=7)
for (_, xs), color in zip(CATEGORIES, COLORS):
    for t in xs:
        ax_bot.vlines(t, -0.08, -0.2, color=color, lw=2, zorder=6)

ax_bot.set_yticks([])
ax_bot.set_xlabel("Whistle duration (s)")
plt.setp(ax_top.get_xticklabels(), visible=False)

fig_m.tight_layout()
fig_m.savefig(PDF_OUT_MEDIANS, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_MEDIANS}")

# --- Per-bird GMMs
all_components = []
all_durations = control["duration"].dropna().values
dmin, dmax = float(np.min(all_durations)), float(np.max(all_durations))
BIN_WIDTH = (dmax - dmin) / NBINS

bird_counts = (
    control.loc[control["duration"] > 0]
           .groupby("bird")["duration"]
           .size()
           .sort_values(ascending=False)
)
birds_sorted = bird_counts.index.to_numpy()

n_birds = len(birds_sorted)
ncols = min(FIG_COLS, n_birds)
nrows = int(np.ceil(n_birds / ncols))

fig, axes = plt.subplots(
    nrows=2*nrows, ncols=ncols, figsize=(4.4*ncols, 3.6*nrows),
    sharex=True, sharey='row', gridspec_kw={"height_ratios": [3, 1] * nrows}
)
axes = np.atleast_2d(axes)

top_peak_global = 0.0

for i, bird in enumerate(birds_sorted):
    r = i // ncols
    c = i %  ncols
    ax_t = axes[2*r,   c]
    ax_b = axes[2*r+1, c]

    vals = control.loc[(control["bird"] == bird) & (control["duration"] > 0), "duration"].to_numpy(float)
    label = bird_label_map.get(bird, str(bird))

    if vals.size == 0:
        ax_t.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_t.transAxes)
        ax_t.set_title(label)
        ax_b.set_visible(False)
        continue

    logd = np.log(vals).reshape(-1, 1)
    lowest_bic, best_gmm, best_k = np.inf, None, None
    for k in range(1, MAX_COMPONENTS + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR,
        ).fit(logd)
        bic = gmm.bic(logd)
        if bic < lowest_bic:
            lowest_bic, best_gmm, best_k = bic, gmm, k

    mu_log = best_gmm.means_.ravel()
    sd_log = np.sqrt(best_gmm.covariances_.ravel())
    w = best_gmm.weights_.ravel()
    order = np.argsort(mu_log)
    mu_log, sd_log, w = mu_log[order], sd_log[order], w[order]

    comp_colors = (
        MODE_COLORS[:3] if best_k == 3 else cm.get_cmap("magma")(np.linspace(0, 1, best_k))
    )

    x = np.linspace(max(1e-9, np.quantile(vals, 0.005)*0.8),
                    np.quantile(vals, 0.995)*1.2, 1200)
    comp_density = [w[k] * normal_pdf(np.log(x), mu_log[k], sd_log[k]) / x for k in range(best_k)]
    mix_density  = np.sum(comp_density, axis=0)

    comp_prob = [cd * BIN_WIDTH for cd in comp_density]
    mix_prob  = mix_density * BIN_WIDTH

    comp_peak = 0.0
    for k in range(best_k):
        ax_t.fill_between(x, comp_prob[k], 0, color=comp_colors[k], alpha=0.9, linewidth=0)
        ax_t.plot(x, comp_prob[k], color=comp_colors[k], lw=1.5)
        comp_peak = max(comp_peak, float(np.nanmax(comp_prob[k])))

    ax_t.plot(x, mix_prob, color="black", lw=2)
    top_peak_global = max(top_peak_global, float(np.nanmax(mix_prob)), comp_peak)

    mean_sec = np.exp(mu_log + 0.5 * (sd_log ** 2))
    ax_t._mu_markers = mean_sec
    ax_t._mu_colors  = comp_colors
    ax_t.set_title(label, fontsize=10)

    labels_raw = best_gmm.predict(logd).ravel()
    inv_order  = np.empty_like(order); inv_order[order] = np.arange(best_k)
    labels_sorted = inv_order[labels_raw]
    pt_colors = np.array(comp_colors, dtype=object)[labels_sorted]

    y_jit = rng.uniform(0.06, 0.16, size=vals.size)
    ax_b.scatter(vals, y_jit, s=12, c=pt_colors, alpha=0.6, edgecolors='none')

    ax_b.set_ylim(0, 0.22)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Whistle duration (s)")

    for idx in range(best_k):
        mu_sec    = np.exp(mu_log[idx])
        sigma_sec = np.exp(mu_log[idx]) * np.sqrt(np.exp(sd_log[idx]**2) - 1)
        all_components.append({
            "bird_label": label,
            "component": idx + 1,
            "mu_sec": float(mu_sec),
            "sigma_sec": float(sigma_sec),
            "weight": float(w[idx]),
            "k": int(best_k),
        })

# hide empty slots
total_slots = nrows * ncols
for j in range(n_birds, total_slots):
    r = j // ncols
    c = j %  ncols
    axes[2*r,   c].axis('off')
    axes[2*r+1, c].axis('off')

# global axes limits
for r in range(nrows):
    for c in range(ncols):
        axes[2*r, c].set_xlim(dmin, dmax)

# align y and place triangles
y_top = top_peak_global * 1.12 if top_peak_global > 0 else 1
for r in range(nrows):
    for c in range(ncols):
        ax_t = axes[2*r, c]
        if hasattr(ax_t, "_mu_markers"):
            ax_t.set_ylim(0, y_top)
            y_star = y_top * 0.94
            ax_t.scatter(
                ax_t._mu_markers,
                np.full_like(ax_t._mu_markers, y_star, dtype=float),
                marker='v', s=60, c=ax_t._mu_colors,
                edgecolors='black', linewidths=0.6, zorder=6, clip_on=False,
            )

fig.supxlabel("Whistle duration (s)")
fig.supylabel("Probability")
fig.tight_layout()
fig.savefig(PDF_OUT_1, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_1}")

components_df = pd.DataFrame(all_components)

# --- μ vs σ clustering and HDR polygons

df = components_df.dropna(subset=["mu_sec", "sigma_sec"]).copy()

gmm2d = GaussianMixture(
    n_components=3, covariance_type="full",
    random_state=RANDOM_STATE, n_init=N_INIT, reg_covar=REG_COVAR,
).fit(df[["mu_sec", "sigma_sec"]])

df["cluster"] = gmm2d.predict(df[["mu_sec", "sigma_sec"]])

cluster_palette = [FAST_RED, MED_YELLOW, SLOW_BLUE]
cluster_order = np.argsort(gmm2d.means_[:, 0])
cluster_color_map = {orig: cluster_palette[rank] for rank, orig in enumerate(cluster_order)}


def hdr_contour(mean, cov, mass=0.90, n=N_CONTOUR_PTS):
    r = np.sqrt(chi2.ppf(mass, df=2))
    vals, vecs = np.linalg.eigh(cov)
    A = vecs @ np.diag(np.sqrt(vals))
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    pts = (mean[:, None] + r * (A @ circle)).T
    pts[:, 1] = np.clip(pts[:, 1], a_min=1e-12, a_max=None)
    return pts


def to_polygon(points):
    if not USE_HULL_POLYGONS or len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]

fig2, ax = plt.subplots(figsize=(4, 3))

for k in range(gmm2d.n_components):
    mask = (df["cluster"] == k)
    if not mask.any():
        continue
    color = cluster_color_map[k]
    ax.scatter(
        df.loc[mask, "mu_sec"], df.loc[mask, "sigma_sec"],
        s=60, alpha=0.85, edgecolor="black", facecolor=color, linewidths=0.5,
        zorder=2, label=f"Cluster {int(k)} (n={int(mask.sum())})",
        rasterized=False,
    )

polygons_out = []
for k in range(gmm2d.n_components):
    color = cluster_color_map[k]
    mean = gmm2d.means_[k]
    cov = gmm2d.covariances_[k]

    boundary90 = hdr_contour(mean, cov, mass=OUTER_MASS, n=N_CONTOUR_PTS)
    poly90 = to_polygon(boundary90)
    ax.fill(poly90[:, 0], poly90[:, 1], facecolor=color, edgecolor=color, alpha=0.5, linewidth=1.6, zorder=1)
    polygons_out.append({"cluster": int(k), "mass": float(OUTER_MASS), "vertices": poly90.tolist(), "color": color})

    boundary68 = hdr_contour(mean, cov, mass=INNER_MASS, n=N_CONTOUR_PTS)
    poly68 = to_polygon(boundary68)
    ax.fill(poly68[:, 0], poly68[:, 1], facecolor=color, edgecolor=color, alpha=1.0, linewidth=1.4, zorder=1)
    polygons_out.append({"cluster": int(k), "mass": float(INNER_MASS), "vertices": poly68.tolist(), "color": color})

ax.set_xlabel("μ (s)")
ax.set_ylabel("σ (s)")

mean_mu = [gmm2d.means_[k][0] for k in range(gmm2d.n_components)]
mean_colors = [cluster_color_map[k] for k in range(gmm2d.n_components)]

y_top = ax.get_ylim()[1]
y_star = y_top * 0.98
ax.scatter(mean_mu, [y_star] * len(mean_mu), marker='v', s=120, facecolors=mean_colors, edgecolors='black', linewidths=0.8, zorder=5, clip_on=False)

fig2.tight_layout()
fig2.savefig(PDF_OUT_2, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved PDF: {PDF_OUT_2}")

with open(POLYGONS_JSON, "w", encoding="utf-8") as f:
    json.dump(polygons_out, f, ensure_ascii=False, indent=2)
print(f"Saved polygons JSON: {POLYGONS_JSON}")

# --- Simulate birds from outer polygons

def _tri_area(tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    return abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) * 0.5


def _tri_sample(tri, rng):
    A, B, C = tri
    r1, r2 = rng.random(), rng.random()
    t = np.sqrt(r1)
    return (1-t)*A + t*(1-r2)*B + t*r2*C


def _sample_in_polygon(poly, rng):
    tris = [np.vstack([poly[0], poly[i], poly[i+1]]) for i in range(1, len(poly)-1)]
    areas = np.array([_tri_area(tr) for tr in tris])
    probs = areas / areas.sum()
    idx = rng.choice(len(tris), p=probs)
    return _tri_sample(tris[idx], rng)

polygons_outer = {int(item["cluster"]): np.array(item["vertices"]) for item in polygons_out if abs(float(item["mass"]) - float(OUTER_MASS)) < 1e-9}

rows = []
for b in range(7):
    sim_bird = f"SimBird {b+1}"
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
print(f"Saved simulated parameters: {SIM_XLSX}")

# --- Save GMM mode table
summary_rows = []
counts_per_mode = np.bincount(labels_sorted, minlength=best_k_med)
for k in range(best_k_med):
    mu_log = float(mu_log_med[k])
    sigma_log = float(sd_log_med[k])
    weight = float(w_med[k])
    median_sec = float(np.exp(mu_log))
    mean_sec_k = float(np.exp(mu_log + 0.5 * sigma_log**2))
    mode_sec = float(np.exp(mu_log - sigma_log**2))
    std_sec = float(np.sqrt((np.exp(sigma_log**2) - 1.0) * np.exp(2*mu_log + sigma_log**2)))
    q05_sec = float(np.exp(mu_log - 1.96 * sigma_log))
    q95_sec = float(np.exp(mu_log + 1.96 * sigma_log))

    summary_rows.append({
        "mode": k+1,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "weight": weight,
        "count": int(counts_per_mode[k]),
        "fraction": float(counts_per_mode[k] / med_vals.size),
        "median_sec": median_sec,
        "mean_sec": mean_sec_k,
        "mode_sec": mode_sec,
        "std_sec": std_sec,
        "q05_sec": q05_sec,
        "q95_sec": q95_sec,
    })

pd.DataFrame(summary_rows).to_csv(GMM_MODES_CSV, index=False)
print(f"Saved CSV: {GMM_MODES_CSV}")