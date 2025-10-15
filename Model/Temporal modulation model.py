# --- Playback response simulation from Excel duration parameters ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import warnings
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

# Config
EXCEL_PATH = "simulated_bird_params_duration.xlsx"
MODE_LABELS = np.array(["Fast (short)", "Medium", "Slow (long)"])
MODE_TO_IDX = {m: i for i, m in enumerate(MODE_LABELS)}
PDF_OUT_COMBINED = "temporal_modulation_model_comparison.pdf"
W_CTRL = np.array([0.157, 0.365, 0.478])
COLORS = ["#E78652", "#DFA71F", "#C69C6D", "#7E4F25"]
CATEGORIES = [
    ("very-short",   [0.04, 0.06, 0.08]),
    ("medium-short", [0.13, 0.14, 0.15]),
    ("medium-long",  [0.30, 0.31, 0.32]),
    ("very-long",    [0.58, 0.72, 0.86]),
]
PANEL_CONFIGS = [
    dict(title="Attractor (β=1, δ=0.5)", beta=1, delta=0.5, n_renditions=50, n_runs=50, base_seed=1200),
    dict(title="No Attractor (β=0)",      beta=0, delta=1.0, n_renditions=50, n_runs=50, base_seed=431),
]

# Utils
def mad(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def _softmax(logits):
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def linear_to_lognormal(mu_sec, sigma_sec):
    mu_sec = float(mu_sec); sigma_sec = float(sigma_sec)
    if mu_sec <= 0 or sigma_sec <= 0:
        raise ValueError("mu_sec and sigma_sec must be positive.")
    sigma2_log = np.log(1.0 + (sigma_sec**2) / (mu_sec**2))
    sd_log = np.sqrt(sigma2_log)
    mu_log = np.log(mu_sec) - 0.5 * sigma2_log
    return mu_log, sd_log

def attractor_probs_no_alpha(x_dur, mu_log, sd_log, w_ctrl=W_CTRL, beta=1.5):
    z = float(np.log(x_dur))
    r = -0.5 * ((z - mu_log) / sd_log) ** 2
    logits = np.log(w_ctrl) + beta * r
    return _softmax(logits).ravel()

def blended_mode_probs(x_dur, mu_log, sd_log, w_ctrl=W_CTRL, beta=1.5, delta=1.0):
    p_eng = attractor_probs_no_alpha(x_dur, mu_log, sd_log, w_ctrl=w_ctrl, beta=beta)
    p_no  = attractor_probs_no_alpha(x_dur, mu_log, sd_log, w_ctrl=w_ctrl, beta=0.0)
    return delta * p_eng + (1.0 - delta) * p_no

def load_bird_params_from_excel(path, expected_modes=MODE_LABELS):
    df = pd.read_excel(path)
    required = {"sim_bird", "mode", "mu_sec", "sigma_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel '{path}' missing columns: {missing}")
    mask_modes = df["mode"].isin(expected_modes)
    if (~mask_modes).any():
        warnings.warn(f"Ignoring unexpected modes: {df.loc[~mask_modes, 'mode'].unique().tolist()}")
        df = df.loc[mask_modes].copy()

    birds = df["sim_bird"].unique().tolist()
    K = len(expected_modes)
    mu_log_birds = np.full((len(birds), K), np.nan, float)
    sd_log_birds = np.full((len(birds), K), np.nan, float)

    for i, b in enumerate(birds):
        sub = df[df["sim_bird"] == b]
        for _, row in sub.iterrows():
            k = MODE_TO_IDX[row["mode"]]
            mu_log, sd_log = linear_to_lognormal(row["mu_sec"], row["sigma_sec"])
            mu_log_birds[i, k] = mu_log
            sd_log_birds[i, k] = sd_log
        if np.isnan(mu_log_birds[i]).any() or np.isnan(sd_log_birds[i]).any():
            miss = [expected_modes[j] for j in np.where(np.isnan(mu_log_birds[i]))[0]]
            raise ValueError(f"Bird '{b}' missing modes: {miss}")

    return birds, mu_log_birds, sd_log_birds

def simulate_population_from_params(x_dur, mu_log_birds, sd_log_birds, w_ctrl=W_CTRL,
                                    beta=1.5, n_renditions=100, rng=123, bird_ids=None, delta=1.0):
    rng = np.random.default_rng(rng)
    n_birds, K = mu_log_birds.shape
    bird_ids = list(range(n_birds)) if bird_ids is None else bird_ids
    rows = []
    for bi in range(n_birds):
        mu_log_k, sd_log_k = mu_log_birds[bi], sd_log_birds[bi]
        pi = blended_mode_probs(x_dur, mu_log_k, sd_log_k, w_ctrl=w_ctrl, beta=beta, delta=delta)
        ks = rng.choice(K, size=n_renditions, p=pi)
        y_log = rng.normal(loc=mu_log_k[ks], scale=sd_log_k[ks], size=n_renditions)
        y_sec = np.exp(y_log)
        for r in range(n_renditions):
            rows.append({"bird": bird_ids[bi], "rendition": r, "x_dur": x_dur,
                         "mode": int(ks[r]), "label": MODE_LABELS[ks[r]],
                         "y_log": float(y_log[r]), "y_sec": float(y_sec[r])})
    return pd.DataFrame(rows)

def kde_1d(samples, grid, bandwidth=None):
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if bandwidth is None:
        s = np.std(x, ddof=1) if n > 1 else 1e-6
        bandwidth = max(1e-6, 1.06 * s * n ** (-1/5))
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(x.reshape(-1, 1))
    dens = np.exp(kde.score_samples(grid.reshape(-1, 1)))
    return dens, bandwidth

def run_panel(fig, outer_spec, title, *, bird_ids, mu_log_birds, sd_log_birds,
              beta, delta, n_renditions, n_runs, base_seed,
              categories=None, colors=None, total_rows=None,
              show_row_titles=True, show_playback_markers=True):
    rows = CATEGORIES if categories is None else categories
    row_colors = (COLORS[:len(rows)] if colors is None else colors)
    assert len(row_colors) == len(rows)

    cat_data = {name: [] for name, _ in rows}
    seed_counter = 0
    for name, xs in rows:
        for x in xs:
            for _ in range(n_runs):
                seed_counter += 1
                df_sim = simulate_population_from_params(
                    x_dur=x, mu_log_birds=mu_log_birds, sd_log_birds=sd_log_birds,
                    w_ctrl=W_CTRL, beta=beta, n_renditions=n_renditions,
                    rng=base_seed + seed_counter, bird_ids=bird_ids, delta=delta
                )
                cat_data[name].append(df_sim["y_sec"].to_numpy())
    for name in cat_data:
        cat_data[name] = np.concatenate(cat_data[name], axis=0)

    xmin = min(arr.min() for arr in cat_data.values())
    xmax = max(arr.max() for arr in cat_data.values())
    xgrid = np.linspace(xmin, xmax, 800)
    N_BINS = 40
    BIN_WIDTH = (xmax - xmin) / N_BINS

    title_ax = fig.add_subplot(outer_spec, frameon=False)
    title_ax.set_title(title, fontsize=13, pad=20)
    title_ax.set_xticks([]); title_ax.set_yticks([])
    for sp in title_ax.spines.values():
        sp.set_visible(False)

    n_rows = len(rows)
    total_rows = n_rows if total_rows is None else total_rows
    sub_rows = outer_spec.subgridspec(nrows=total_rows, ncols=1, hspace=0.45)
    main_axes, last_ax_jit = [], None

    for i, ((name, xs), color) in enumerate(zip(rows, row_colors)):
        row_spec = sub_rows[i].subgridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
        ax_main = fig.add_subplot(row_spec[0])
        ax_jit  = fig.add_subplot(row_spec[1], sharex=ax_main)
        main_axes.append(ax_main)

        arr = cat_data[name]
        dens, _ = kde_1d(arr, xgrid)
        prob_curve = dens * BIN_WIDTH
        ax_main.plot(xgrid, prob_curve, lw=2, color='k')
        ax_main.fill_between(xgrid, 0, prob_curve, color=color)
        ax_main.axvline(np.median(arr), color="black", lw=2)
        if show_row_titles:
            ax_main.set_title(name.replace("-", " "))
        ax_main.set_ylabel("Probability")

        rng_vis = np.random.default_rng(123 + i)
        n_plot = min(5000, len(arr))
        idx = rng_vis.choice(len(arr), size=n_plot, replace=False)
        ax_jit.scatter(arr[idx], rng_vis.uniform(0.0, 1.0, size=n_plot),
                       s=6, alpha=0.35, color=color, edgecolors="none")
        if show_playback_markers:
            for xv in xs:
                ax_jit.scatter([xv], [1.02], s=28, color="black",
                               edgecolors="white", linewidths=0.6, zorder=6)
        ax_jit.set_ylim(0, 1.10); ax_jit.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax_jit.spines[sp].set_visible(False)
        ax_main.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_jit.tick_params(axis="x", labelbottom=True, bottom=True)
        last_ax_jit = ax_jit

    for j in range(n_rows, total_rows):
        row_spec = sub_rows[j].subgridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
        fig.add_subplot(row_spec[0]).axis("off")
        fig.add_subplot(row_spec[1]).axis("off")

    for ax in main_axes:
        ax.set_xlim(0, 0.9)
    for ax in main_axes[1:]:
        ax.sharey(main_axes[0])
    if last_ax_jit is not None:
        last_ax_jit.set_xlabel("Whistle duration (s)")
    return main_axes

# Main figure (control vs engagement)
if __name__ == "__main__":
    birds, mu_log_birds, sd_log_birds = load_bird_params_from_excel(EXCEL_PATH)

    control_chunks, _seed = [], 0
    for _, xs in CATEGORIES:
        for x in xs:
            for _ in range(PANEL_CONFIGS[1]["n_runs"]):
                _seed += 1
                df_c = simulate_population_from_params(
                    x_dur=x, mu_log_birds=mu_log_birds, sd_log_birds=sd_log_birds,
                    w_ctrl=W_CTRL, beta=PANEL_CONFIGS[1]["beta"],
                    n_renditions=PANEL_CONFIGS[1]["n_renditions"],
                    rng=PANEL_CONFIGS[1]["base_seed"] + _seed,
                    delta=PANEL_CONFIGS[1]["delta"]
                )
                control_chunks.append(df_c["y_sec"].to_numpy())
    control_data = np.concatenate(control_chunks, axis=0)

    cat_data, _seed = {name: [] for name, _ in CATEGORIES}, 0
    for name, xs in CATEGORIES:
        for x in xs:
            for _ in range(PANEL_CONFIGS[0]["n_runs"]):
                _seed += 1
                df_e = simulate_population_from_params(
                    x_dur=x, mu_log_birds=mu_log_birds, sd_log_birds=sd_log_birds,
                    w_ctrl=W_CTRL, beta=PANEL_CONFIGS[0]["beta"],
                    n_renditions=PANEL_CONFIGS[0]["n_renditions"],
                    rng=PANEL_CONFIGS[0]["base_seed"] + _seed,
                    delta=PANEL_CONFIGS[0]["delta"]
                )
                cat_data[name].append(df_e["y_sec"].to_numpy())
    for k in list(cat_data.keys()):
        cat_data[k] = np.concatenate(cat_data[k], axis=0)

    for cat_name, _ in CATEGORIES:
        arr = cat_data[cat_name]
        print(f"[Region] {cat_name:>12s} | n={arr.size:,} | median={np.median(arr):.4f} s | MAD={mad(arr):.4f} s")
    print(f"[Control] {'pooled':>12s} | n={control_data.size:,} | median={np.median(control_data):.4f} s | MAD={mad(control_data):.4f} s")

    plt.style.use('default')
    fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(5, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5]})
    for i in range(0, 10, 2):
        axes[i].sharey(axes[4])

    xmin = min([control_data.min()] + [v.min() for v in cat_data.values()])
    xmax = max([control_data.max()] + [v.max() for v in cat_data.values()])
    xgrid = np.linspace(xmin, xmax, 800)
    dx_ref = float(np.mean(np.diff(xgrid)))

    def kde_prob(samples, grid):
        dens, _ = kde_1d(samples, grid)
        return dens * dx_ref

    ymax_list = []
    prob_ctrl = kde_prob(control_data, xgrid)
    axes[0].plot(xgrid, prob_ctrl, color='black', lw=2, zorder=2)
    axes[0].fill_between(xgrid, prob_ctrl, color='gray', alpha=0.2, zorder=1)
    axes[0].set_xlim([0, 0.9]); axes[0].set_ylabel("Probability")
    ymax_list.append(np.nanmax(prob_ctrl))

    rng_vis = np.random.default_rng(7)
    target_n = min(len(control_data), min(len(v) for v in cat_data.values()))
    idx = rng_vis.choice(len(control_data), size=target_n, replace=False)
    control_plot = control_data[idx]
    yjit = rng_vis.uniform(low=-0.075, high=0.075, size=len(control_plot))
    axes[1].scatter(control_plot, yjit, color='black', alpha=0.01, s=2, rasterized=True)
    axes[1].vlines(np.median(control_data), -0.08, -0.2, color='black', lw=2, zorder=6)
    for (cat_name, xs), color in zip(CATEGORIES, COLORS):
        for t in xs:
            axes[1].vlines(t, -0.08, -0.2, color=color, lw=2, zorder=5)
    axes[1].set_yticks([]); axes[1].set_ylabel("Control")

    for idx_cat, ((cat_name, xs), color) in enumerate(zip(CATEGORIES, COLORS)):
        arr = cat_data[cat_name]
        prob_exp = kde_prob(arr, xgrid)
        kde_ax = axes[2 * (idx_cat + 1)]
        kde_ax.plot(xgrid, prob_exp, color='black', lw=2, zorder=2)
        kde_ax.fill_between(xgrid, prob_exp, color=color, zorder=1)
        kde_ax.set_ylabel("Probability")
        ymax_list.append(np.nanmax(prob_exp))
        jit_ax = axes[2 * (idx_cat + 1) + 1]
        yjit = np.random.default_rng(123 + idx_cat).uniform(low=-0.075, high=0.075, size=len(arr))
        jit_ax.scatter(arr, yjit, color=color, alpha=0.1, s=2, edgecolor='none', rasterized=True)
        for t in xs:
            jit_ax.vlines(t, -0.08, -0.2, color=color, lw=2, zorder=5)
        jit_ax.vlines(np.median(arr), -0.08, -0.2, color='black', lw=2, zorder=6)
        jit_ax.set_yticks([]); jit_ax.set_ylabel(cat_name.replace("-", " "))

    ymax = float(max(ymax_list)) * 1.05
    for ax in axes[::2]:
        ax.set_ylim(0, ymax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    mpl.rcParams['pdf.fonttype'] = 42
    fig.savefig(PDF_OUT_COMBINED, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

# --- Attractor landscape (duration) ---
EPS = 1e-12

def _r_k(x, mu_log, sd_log):
    z = np.log(x)
    return -0.5 * ((z - mu_log) / sd_log) ** 2

def compute_attractor_matrix_duration(x_grid, mu_log, sd_log, *, apply_beta=False, beta=1.0, normalize_each=True):
    x_grid = np.asarray(x_grid, float)
    mu_log = np.asarray(mu_log, float)
    sd_log = np.asarray(sd_log, float)
    K = mu_log.size
    A = np.empty((x_grid.size, K), float)
    for i, x in enumerate(x_grid):
        r = _r_k(float(x), mu_log, sd_log)
        a = np.exp((beta if apply_beta else 1.0) * r)
        A[i, :] = a
    if normalize_each:
        A = A / np.maximum(A.max(axis=0), EPS)
    return A

def plot_attractor_landscape_duration(mu_log, sd_log, *, x_min=0.04, x_max=0.9, n_grid=1200,
                                      apply_beta=False, beta=1.0, triangle_size=10, use_log_x=False,
                                      save_pdf_path="attractor_landscape_duration.pdf", show=True,
                                      gray_levels=(0.25, 0.45, 0.65)):
    mu_log = np.asarray(mu_log, float)
    sd_log = np.asarray(sd_log, float)
    K = mu_log.size
    x = np.exp(np.linspace(np.log(x_min), np.log(x_max), int(n_grid))) if use_log_x else np.linspace(float(x_min), float(x_max), int(n_grid))
    A = compute_attractor_matrix_duration(x, mu_log, sd_log, apply_beta=apply_beta, beta=beta, normalize_each=True)
    if isinstance(gray_levels, (list, tuple)) and len(gray_levels) >= K:
        colors = [str(g) for g in gray_levels[:K]]
    else:
        colors = [str(g) for g in np.linspace(0.25, 0.75, K)]

    fig, ax = plt.subplots(figsize=(5, 2))
    for k, col in enumerate(colors):
        ax.plot(x, A[:, k], lw=2, color=col)

    m = np.exp(mu_log)
    for k, mk in enumerate(m):
        if x_min <= mk <= x_max:
            ax.plot([mk], [1.05], marker="v", ls="none", ms=triangle_size,
                    markerfacecolor=colors[k], markeredgecolor="black",
                    markeredgewidth=1.2, zorder=5, clip_on=False)

    label_core = r"$A_k(x)=\exp(\beta\,r_k(x))$" if (apply_beta and beta != 1.0) else r"$A_k(x)=\exp(r_k(x))$"
    ax.set_xlabel("Playback duration (s)")
    ax.set_ylabel(label_core)
    if use_log_x:
        ax.set_xscale("log")
        major_ticks = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
        major_ticks = major_ticks[(major_ticks >= x_min) & (major_ticks <= x_max)]
        if major_ticks.size:
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
            ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.1)

    plt.tight_layout()
    mpl.rcParams['pdf.fonttype'] = 42
    if save_pdf_path:
        plt.savefig(save_pdf_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

#Attractor
fig, ax = plot_attractor_landscape_duration(
    mu_log=mu_log_birds[0], sd_log=sd_log_birds[0],
    x_min=0.04, x_max=0.9, apply_beta=True, beta=1.0,
    save_pdf_path="attractor_landscape_duration_bird0.pdf",
    gray_levels=(0.20, 0.45, 0.70)
)
