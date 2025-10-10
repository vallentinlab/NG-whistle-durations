# ============================================================
# Frequency response simulation USING EXCEL per-bird parameters
# Distance-decay attractor (favors nearest mode; smooth decay)
# + Attractor Landscape plot (component probs vs playback pitch)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
# ------------------------------------------------------------
# 0) CONFIG
# ------------------------------------------------------------
EXCEL_PATH = "simulated_bird_params_pitch.xlsx"  # <-- set your Excel path
PDF_OUT_2 = "pitch_modulation_results_model_comparison.pdf"
LANDSCAPE_PDF = "attractor_landscape_bird.pdf"

N_SAMPLES_PER_PLAYBACK_PER_BIRD = 200
PLAYBACKS = np.array([1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600], float)

# RNG seeds (one generator for sim; one for plotting jitter)
RNG_SEED_SIM = 12
RNG_SEED_VIS = 202

# Visualization
JITTER_WIDTH = 60.0
MAKE_AXES_SQUARE = True  # identity comparisons are easier to read

# KDE bandwidth (None -> data-driven)
KDE_BW = None
DELTA = 0.5

# ------------------------------------------------------------
# GLOBAL weights for every bird/component (left→right)
# (Used as a weak background prior and for alignment)
# ------------------------------------------------------------
W_GLOBAL = np.array([
    0.06823987, 0.11403142, 0.16720762, 0.20454148, 0.11624734,
    0.08214482, 0.04874001, 0.09543889, 0.07369576, 0.02971277
], dtype=float)

EPS = 1e-12  # small constant for numerical stability

# ------------------------------------------------------------
# 1) Load per-bird (μ, σ, w) from Excel  ✅ fixed column rename bug
# ------------------------------------------------------------
def load_bird_params_from_excel_freq(path):
    """
    Excel must have per-component rows with at least:
      - 'bird' or 'sim_bird'
      - μ column (e.g., 'mu_hz', 'mu (hz)', 'mu')
      - σ column (e.g., 'sigma_hz', 'sigma (hz)', 'sigma')
    Optional:
      - 'weight' (per component); if missing or invalid, use uniform within bird.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # pick bird column
    if "bird" in df.columns:
        bird_col = "bird"
    elif "sim_bird" in df.columns:
        bird_col = "sim_bird"
    else:
        raise ValueError("Excel must have a 'bird' or 'sim_bird' column.")

    # be flexible with mu/sigma headers
    mu_candidates = ["mu_hz", "mu (hz)", "mu"]
    sg_candidates = ["sigma_hz", "sigma (hz)", "sigma"]
    try:
        mu_col = next(c for c in mu_candidates if c in df.columns)
        sg_col = next(c for c in sg_candidates if c in df.columns)
    except StopIteration:
        raise ValueError("Excel must include μ and σ columns (e.g., 'mu_hz' and 'sigma_hz').")

    has_weight = "weight" in df.columns

    # ✅ select first, then rename (avoids the ['sim_bird'] not in index error)
    cols_in = [bird_col, mu_col, sg_col] + (["weight"] if has_weight else [])
    df_use = df[cols_in].rename(columns={bird_col: "bird", mu_col: "mu_hz", sg_col: "sigma_hz"}).copy()

    # cleaning
    df_use = df_use.replace([np.inf, -np.inf], np.nan).dropna(subset=["bird", "mu_hz", "sigma_hz"])
    df_use["sigma_hz"] = df_use["sigma_hz"].clip(lower=1e-6)

    bird_params = {}
    for b, sub in df_use.groupby("bird", sort=False):
        sub = sub.sort_values("mu_hz")  # left → right
        mu = sub["mu_hz"].to_numpy(float)
        sd = sub["sigma_hz"].to_numpy(float)
        if has_weight:
            w = sub["weight"].to_numpy(float)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones_like(mu, float)
        else:
            w = np.ones_like(mu, float)
        w = w / w.sum()
        bird_params[str(b)] = dict(mu=mu, sigma=sd, w=w)

    birds = sorted(bird_params.keys())
    if not birds:
        raise ValueError("No birds found in Excel after filtering/cleaning.")
    return bird_params, birds

# ------------------------------------------------------------
# 2) Utilities
# ------------------------------------------------------------
def _softmax(logits):
    logits = np.asarray(logits, float)
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def _align_global_weights(mu, w_global=W_GLOBAL, eps=EPS):
    """Align global weights to number of components for this bird."""
    Kb = len(mu)
    if Kb <= len(w_global):
        w = np.array(w_global[:Kb], dtype=float)
    else:
        w = np.ones(Kb, dtype=float) / Kb
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(Kb, dtype=float) / Kb
    w = np.clip(w, eps, None)
    return w / w.sum()

def _coerce_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng  # assume Generator

def control_probs(params):
    w = np.asarray(params.get("w", None), float)
    if w is None or not np.all(np.isfinite(w)) or w.sum() <= 0:
        w = _align_global_weights(params["mu"])
    w = np.clip(w, EPS, None)
    return w / w.sum()

def blended_attractor_probs(x_hz, params, *, delta=1.0, beta=1.0, **gate_kwargs):
    mu, w = params["mu"], params.get("w", None)
    p_eng = attractor_probs_local(x_hz, mu, w, beta=beta, **gate_kwargs)
    p_no  = control_probs(params)
    delta = float(np.clip(delta, 0.0, 1.0))
    return delta * p_eng + (1.0 - delta) * p_no

# ------------------------------------------------------------
# 3) Distance-decay attractor (component selection prior)
# ------------------------------------------------------------
def _distance_kernel(d, tau=400.0, p=2.0, kind="gaussian", nu=1.0):
    d = np.asarray(d, float)
    if kind == "gaussian":
        return np.exp(- (d / tau)**p)
    elif kind == "laplace":
        return np.exp(- (d / tau))
    elif kind == "cauchy":
        return 1.0 / (1.0 + (d / tau)**2)
    elif kind == "rq":
        return (1.0 + (d / tau)**2 / (2.0 * nu))**(-nu)
    else:
        raise ValueError(f"unknown kernel kind={kind}")

def attractor_probs_local(x_hz, mu, w_unused=None, *,
                          tau=400.0, p=2.0, kind="gaussian",
                          alpha_bg=0.05, nearest_boost=0.0, rq_nu=1.0,
                          beta=1.0):
    mu = np.asarray(mu, float)
    w_prior = _align_global_weights(mu)
    d = np.abs(float(x_hz) - mu)
    K = _distance_kernel(d, tau=tau, p=p, kind=kind, nu=rq_nu)
    if beta != 1.0:
        K = np.clip(K, EPS, None) ** float(beta)

    scores = np.log(np.clip(w_prior, EPS, None)) + np.log(np.clip(K, EPS, None))
    if nearest_boost:
        scores[np.argmin(d)] += float(nearest_boost)
    loc = _softmax(scores).ravel()

    if alpha_bg > 0.0:
        bg = w_prior / w_prior.sum()
        loc = (1.0 - alpha_bg) * loc + alpha_bg * bg
        loc = loc / loc.sum()

    return loc

def sample_response_bird_local(x_hz, params, n=1, rng=None, delta=1.0, beta=1.0, **gate_kwargs):
    rng = _coerce_rng(rng)
    mu, sd = params["mu"], params["sigma"]
    pis = blended_attractor_probs(x_hz, params, delta=delta, beta=beta, **gate_kwargs)
    k   = rng.choice(len(mu), size=n, p=pis)
    y   = rng.normal(loc=mu[k], scale=sd[k], size=n)
    return y

# ------------------------------------------------------------
# 4) KDE helper
# ------------------------------------------------------------
def kde_density_and_mode(samples, grid_pts=400, bandwidth=KDE_BW):
    y = np.asarray(samples, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None, None, None, None
    lo, hi = y.min(), y.max()
    if hi == lo:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_pts).reshape(-1, 1)
    if bandwidth is None:
        s = np.std(y, ddof=1) if y.size > 1 else 1.0
        bandwidth = max(1e-6, 1.06 * s * y.size ** (-1/5))
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(y.reshape(-1, 1))
    dens_grid = np.exp(kde.score_samples(grid))
    mode_val = float(grid[np.argmax(dens_grid), 0])
    return dens_grid, grid.ravel(), mode_val, kde

# ------------------------------------------------------------
# 5) Load data
# ------------------------------------------------------------
bird_params, birds = load_bird_params_from_excel_freq(EXCEL_PATH)

# ------------------------------------------------------------
# 6) Gate hyperparameters (define BEFORE function)
# ------------------------------------------------------------
TAU_HZ = 100       # range of influence (~0.4–0.6 × median μ spacing)
KIND   = "cauchy"  # heavy tails ('rq' or 'cauchy' recommended)
P_EXP  = 2.0       # only used by gaussian/laplace
ALPHA_BG = 0.0     # background prior
NEAREST_BONUS = 0  # extra push to the closest mode
RQ_NU = 1.0        # shape for rational quadratic

# ------------------------------------------------------------
# 7) Simulation + Visualization for a given beta
# (kept as in your version; not the target figure, but retained)
# ------------------------------------------------------------
def run_simulation_and_plot(beta, ax, title, *, single_column=False, column_x=None):
    rng_sim = np.random.default_rng(RNG_SEED_SIM)

    if single_column:
        x_single = float(np.median(PLAYBACKS)) if column_x is None else float(column_x)
        effective_playbacks = np.array([x_single], dtype=float)
    else:
        effective_playbacks = PLAYBACKS

    rows = []
    for b in birds:
        params = bird_params[b]
        for x0 in effective_playbacks:
            y = sample_response_bird_local(
                x0, params,
                n=N_SAMPLES_PER_PLAYBACK_PER_BIRD,
                rng=rng_sim,
                tau=TAU_HZ, p=P_EXP, kind=KIND,
                alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BONUS,
                rq_nu=RQ_NU,
                delta=DELTA,
                beta=beta,
            )
            rows.extend({"bird": b, "playback_hz": x0, "response_hz": yy} for yy in y)

    sim_df = pd.DataFrame(rows)

    rng_vis = np.random.default_rng(RNG_SEED_VIS)
    cmap = cm.get_cmap("viridis")

    empirical_modes = {}
    first_scatter = True
    first_modebar = True

    for x0 in effective_playbacks:
        mask = (sim_df["playback_hz"] == x0).values
        y    = sim_df.loc[mask, "response_hz"].to_numpy()

        _, _, mode_val, kde = kde_density_and_mode(y, bandwidth=KDE_BW)
        empirical_modes[x0] = mode_val

        dens_y = np.exp(kde.score_samples(y.reshape(-1, 1)))
        vmin, vmax = dens_y.min(), dens_y.max()
        vmax = vmin + 1e-9 if vmax <= vmin else vmax
        colors_i = cmap(Normalize(vmin=vmin, vmax=vmax, clip=True)(dens_y))

        xj = x0 + rng_vis.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=mask.sum())
        ax.scatter(xj, y, c=colors_i, s=5, alpha=0.9, edgecolors="none",
                   label="Responses" if first_scatter else None)
        first_scatter = False

        ax.hlines(y=mode_val, xmin=x0 - JITTER_WIDTH, xmax=x0 + JITTER_WIDTH,
                  colors="crimson", linewidth=3, zorder=2,
                  label="Mode (KDE)" if first_modebar else None)

        if not single_column:
            ax.hlines(y=x0, xmin=x0 - JITTER_WIDTH, xmax=x0 + JITTER_WIDTH,
                      colors="black", linewidth=3, zorder=1,
                      label="Playback" if first_modebar else None)

        first_modebar = False

    xmin, xmax = PLAYBACKS.min(), PLAYBACKS.max()
    pad = 0.05 * (xmax - xmin)
    mn, mx = xmin - pad, xmax + pad
    ax.set_xlim(mn, mx)

    ax.set_ylim([1000, 9000])
    ax.set_yscale("log")

    if single_column:
        ax.set_xticks([effective_playbacks[0]])
        ax.set_xticklabels(["Responses"])
        ax.set_xlabel("No playback")
        ax.set_ylabel("Whistle syllable pitch (Hz)")
    else:
        ax.set_xticks(PLAYBACKS)
        ax.set_xlabel("Pitch of whistle playback (Hz)")
        ax.set_ylabel("Whistle syllable pitch of matching bird (Hz)")

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set_title(title)
    return empirical_modes, sim_df

# Example retained (not required for the landscape figure)
BETA = 3
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
_ , _ = run_simulation_and_plot(BETA, axes[0], f"Attractor (β={BETA}, δ={DELTA})", single_column=False)
_ , _ = run_simulation_and_plot(0, axes[1], "No attractor (β=0)", single_column=True)
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig(PDF_OUT_2, dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 8) Attractor landscape (one bird): 1000–9000 Hz, magma, NO scatter/dashes/legend
#    Triangles at y=1.0 sized 90, face color by mode, black edge.
#    Log x-axis with fixed ticks.
# ------------------------------------------------------------

def compute_K_matrix_for_bird(
    bird_id,
    x_grid,
    *,
    tau=TAU_HZ,
    p=P_EXP,
    kind=KIND,
    rq_nu=RQ_NU,
    apply_beta=False,
    beta=1.0,
):
    """Return K(x,k) for a bird across x_grid (optionally ^beta)."""
    params = bird_params[str(bird_id)]
    mu = np.asarray(params["mu"], float)
    Kmat = np.empty((x_grid.size, mu.size), float)
    for i, x in enumerate(x_grid):
        d = np.abs(float(x) - mu)
        Krow = _distance_kernel(d, tau=tau, p=p, kind=kind, nu=rq_nu)
        if apply_beta and beta != 1.0:
            Krow = np.clip(Krow, EPS, None) ** float(beta)
        Kmat[i, :] = Krow
    return Kmat, mu

def plot_K_landscape(
    bird_id=None,
    *,
    x_min=1000.0,
    x_max=9000.0,
    n_grid=1200,
    tau=TAU_HZ,
    p=P_EXP,
    kind=KIND,
    rq_nu=RQ_NU,
    apply_beta=False,   # set True if you want to see K^beta instead of K
    beta=1.0,
    triangle_size=10,
    save_pdf_path="K_landscape_bird.pdf",
    show=True
):
    if bird_id is None:
        bird_id = birds[0]

    x = np.linspace(float(x_min), float(x_max), int(n_grid))
    Kmat, mu = compute_K_matrix_for_bird(
        bird_id, x, tau=tau, p=p, kind=kind, rq_nu=rq_nu,
        apply_beta=apply_beta, beta=beta
    )

    fig, ax = plt.subplots(figsize=(5, 2))

    # magma colors per component
    cmap = cm.get_cmap("magma")
    colors = cmap(np.linspace(0, 1, Kmat.shape[1]))

    # curves (no legend)
    for k, col in enumerate(colors):
        ax.plot(x, Kmat[:, k], lw=2, color=col)

    # triangles at the top (y=1.05) at each μ, colored by mode with black edge
    for k, m in enumerate(mu):
        if x_min <= m <= x_max:
            ax.plot([m], [1.05],
                    marker="v", ls="none",
                    ms=triangle_size,
                    markerfacecolor=colors[k],
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                    zorder=5, clip_on=False)

    # Labels and axes
    label_core = r"$K_k(x)$" if not (apply_beta and beta != 1.0) else r"$K_k(x)^{\beta}$"
    ax.set_xlabel("Playback pitch (kHz)")
    ax.set_ylabel(label_core)

    # Log x-axis with ONLY the specified major ticks, labeled in kHz
    major_ticks = np.array([1000, 2000, 3000, 4000, 5000, 7000, 9000], float)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/1000:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())  # no minor ticks (no surprise 6000s)

    # Kernel range
    ax.set_ylim(0.0, 1.1)

    plt.tight_layout()
    mpl.rcParams['pdf.fonttype'] = 42
    if save_pdf_path:
        plt.savefig(save_pdf_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# --- Example: RAW K(x) (no beta sharpening) ---
_ = plot_K_landscape(
    bird_id=birds[0],
    x_min=1000, x_max=9000,
    apply_beta=False, beta=1.0,
    save_pdf_path=f"K_landscape_raw_bird_{birds[0]}.pdf"
)
