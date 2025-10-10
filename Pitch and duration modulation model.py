# ============================================================
# Nightingale: Control KDE → Duration-Conditioned Support → Joint (d,f) Simulation
# With control-KDE-based blended probabilities (δ=0.3, β=2) for duration & pitch
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.special import erf, erfinv

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
DATA_PKL       = r"Control prev season/all_birds.pkl"                 # <- input data
EXCEL_DURATION = "simulated_bird_params_duration.xlsx"               # <- per-bird duration params
EXCEL_PITCH    = "simulated_bird_params_pitch.xlsx"                  # <- per-bird pitch params
PLOT_OUT       = r"Plots/control_meanvspitch_with_clusters.pdf"      # <- output figure

os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)

# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------
# Control KDE / support
KDE_BW_ADJUST = 0.7
SUPPORT_THRESH = 0.1     # outer contour = SUPPORT_THRESH * max density
GRIDSIZE = 50
CUT_PAD = 0.20
BIN_WIDTH_S = 0.020        # 20 ms bins

# Duration model (3 lognormal modes)
MODE_LABELS = np.array(["Fast (short)", "Medium", "Slow (long)"])
MODE_TO_IDX = {m: i for i, m in enumerate(MODE_LABELS)}
W_CTRL = np.array([0.157, 0.365, 0.478])  # baseline mixture weights (still used as prior)

# Pitch model (component prior template)
W_GLOBAL = np.array([
    0.06823987, 0.11403142, 0.16720762, 0.20454148, 0.11624734,
    0.08214482, 0.04874001, 0.09543889, 0.07369576, 0.02971277
], dtype=float)

# Blending hyperparams (as requested)
DELTA_BLEND = 0.5

# Engagement sharpness (separate betas)
BETA_DUR   = 1    # duration gate sharpness
BETA_PITCH = 3    # pitch gate sharpness


EPS = 1e-12

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def _softmax(logits):
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def linear_to_lognormal(mu_sec, sigma_sec):
    mu_sec = float(mu_sec); sigma_sec = float(sigma_sec)
    if mu_sec <= 0 or sigma_sec <= 0:
        raise ValueError("mu_sec and sigma_sec must be positive.")
    sigma2_log = np.log(1.0 + (sigma_sec**2)/(mu_sec**2))
    sd_log = np.sqrt(sigma2_log)
    mu_log = np.log(mu_sec) - 0.5 * sigma2_log
    return mu_log, sd_log

def attractor_probs_no_alpha(x_dur, mu_log, sd_log, w_ctrl=W_CTRL, beta=BETA_DUR):
    """Duration gate: engaged probs given playback duration (lognormal modes)."""
    z = float(np.log(x_dur))
    r = -0.5 * ((z - mu_log) / sd_log) ** 2
    logits = np.log(w_ctrl) + beta * r
    return _softmax(logits).ravel()

def _align_global_weights(mu, w_global=W_GLOBAL, eps=EPS):
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

def _distance_kernel(d, tau=100.0, p=2.0, kind="cauchy", nu=1.0):
    d = np.asarray(d, float)
    if kind == "gaussian":
        return np.exp(-(d / tau) ** p)
    if kind == "laplace":
        return np.exp(-(d / tau))
    if kind == "cauchy":
        return 1.0 / (1.0 + (d / tau) ** 2)
    if kind == "rq":
        return (1.0 + (d / tau) ** 2 / (2.0 * nu)) ** (-nu)
    raise ValueError(f"unknown kernel kind={kind}")

def attractor_probs_local(x_hz, mu, *, tau=400.0, p=2.0, kind="cauchy",
                          alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0, beta=BETA_PITCH):
    """Pitch gate: engaged probs given playback pitch."""
    mu = np.asarray(mu, float)
    w_prior = _align_global_weights(mu)
    d = np.abs(float(x_hz) - mu)
    K = _distance_kernel(d, tau=tau, p=p, kind=kind, nu=rq_nu)
    # apply beta (sharpens/softens distance)
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

def normal_cdf(x, mu, sd):
    z = (x - mu) / (sd * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))

def normal_ppf(p, mu, sd):
    return mu + sd * np.sqrt(2.0) * erfinv(2.0 * p - 1.0)

# ------------------------------------------------------------
# BUILD CONTROL SUMMARIES (whistle_songs_control)
# ------------------------------------------------------------
all_birds = pd.read_pickle(DATA_PKL)
control = all_birds[all_birds.phase != 'playback'].copy()

whistle_songs_control = []
for bird in control.bird.unique():
    cb = control[control.bird == bird]
    for phase_id in cb.phase.unique():
        if phase_id == 'playback':
            continue
        cp = cb[cb.phase == phase_id]
        for set_id in cp.set.unique():
            cs = cp[cp.set == set_id]
            for snippet in cs.snippet_idx.unique():
                song = cs[cs.snippet_idx == snippet].copy()
                n_syl = len(song['duration'])
                last_int = song['interval'].iloc[-2] if n_syl > 1 else np.nan
                first_int = song['interval'].iloc[0] if n_syl > 0 else np.nan
                last_gap = song['gap'].iloc[-2] if n_syl > 1 else np.nan
                first_gap = song['gap'].iloc[0] if n_syl > 0 else np.nan
                whistle_songs_control.append({
                    'bird': bird, 'song': snippet,
                    'last_d': song['duration'].iloc[-1],
                    'first_d': song['duration'].iloc[0],
                    'd_median': song['duration'].median(),
                    'd_average': song['duration'].mean(),
                    'd_range': song['duration'].max() - song['duration'].min(),
                    'last_f': song['pitch_whistles'].iloc[-1],
                    'first_f': song['pitch_whistles'].iloc[0],
                    'f_median': song['pitch_whistles'].median(),
                    'f_average': song['pitch_whistles'].mean(),
                    'n_syl': n_syl,
                    'gap_median': song['gap'].median(),
                    'int_median': song['interval'].median(),
                    'last_int': last_int, 'first_int': first_int,
                    'last_gap': last_gap, 'first_gap': first_gap,
                    'int_first_to_last': (first_int - last_int) if n_syl > 1 else np.nan,
                    'gap_first_to_last': (first_gap - last_gap) if n_syl > 1 else np.nan,
                    'd_first_to_last': song['duration'].iloc[0] - song['duration'].iloc[-1],
                    'f_first_to_last': song['pitch_whistles'].iloc[0] - song['pitch_whistles'].iloc[-1],
                })

whistle_songs_control = pd.DataFrame(whistle_songs_control)

# ------------------------------------------------------------
# CONTROL KDE PLOT (duration vs pitch) + save
# ------------------------------------------------------------
plt.style.use('default')
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])
fig = plt.figure(figsize=(5, 5))

sns.kdeplot(
    x=whistle_songs_control.d_median,
    y=whistle_songs_control.f_median,
    cmap=custom_cmap, fill=True,
    bw_adjust=KDE_BW_ADJUST, levels=20, thresh=SUPPORT_THRESH, zorder=-1
)
sns.kdeplot(
    x=whistle_songs_control.d_median,
    y=whistle_songs_control.f_median,
    cmap="Reds_r", fill=False,
    bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH
)
#plt.scatter(
#    whistle_songs_control.d_median,
#    whistle_songs_control.f_median,
#    c='k', s=2, zorder=1
#)
plt.xlabel('Median of whistle durations per song (s)')
plt.ylabel('Median frequency (Hz)')
plt.gca().set_box_aspect(1)
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig(PLOT_OUT, transparent=True)
plt.show()

# ------------------------------------------------------------
# SUPPORT MAP CLASS (duration → union of valid frequency intervals)
# ------------------------------------------------------------
class SupportByDuration:
    def __init__(self, d_median, f_median,
                 bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
                 gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_S):
        d = np.asarray(d_median, float)
        f = np.asarray(f_median, float)
        m = np.isfinite(d) & np.isfinite(f)
        d, f = d[m], f[m]
        if d.size < 5:
            raise ValueError("Not enough points to build support.")

        x_min, x_max = d.min(), d.max()
        y_min, y_max = f.min(), f.max()
        x_pad = (x_max - x_min) * cut_pad
        y_pad = (y_max - y_min) * cut_pad
        self.xx = np.linspace(x_min - x_pad, x_max + x_pad, gridsize)
        self.yy = np.linspace(y_min - y_pad, y_max + y_pad, gridsize)
        X, Y = np.meshgrid(self.xx, self.yy)

        kde = gaussian_kde(np.vstack([d, f]))
        kde.covariance_factor = lambda: kde.scotts_factor() * bw_adjust
        kde._compute_covariance()
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(gridsize, gridsize)

        self.level = float(Z.max()) * float(thresh)
        self.mask = (Z >= self.level)
        self.bin_width = float(bin_width)

        # Precompute per-bin frequency unions
        rows = []
        edges = np.arange(self.xx.min(), self.xx.max() + bin_width, bin_width)
        for left, right in zip(edges[:-1], edges[1:]):
            x_cols = np.where((self.xx >= left) & (self.xx < right))[0]
            if x_cols.size == 0:
                rows.append((left, right, []))
                continue
            col_any = self.mask[:, x_cols].any(axis=1)
            intervals = self._intervals_from_bool(col_any, self.yy)
            rows.append((left, right, intervals))
        self.rows = rows  # list of (left, right, [(y0,y1), ...])

    @staticmethod
    def _intervals_from_bool(bool_vec, y_axis):
        idx = np.flatnonzero(bool_vec)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [(float(y_axis[g[0]]), float(y_axis[g[-1]])) for g in groups]

    def intervals_for_duration(self, x):
        x = float(x)
        for (l, r, ints) in self.rows:
            if l <= x < r:
                return ints
        return []

    def intervals_near_duration(self, x, max_neighbors=3):
        x = float(x)
        edges = np.array([r[0] for r in self.rows] + [self.rows[-1][1]])
        centers = 0.5 * (edges[:-1] + edges[1:])
        if centers.size == 0:
            return []
        idx = int(np.argmin(np.abs(centers - x)))
        for k in range(max_neighbors + 1):
            left = max(0, idx - k); right = min(len(self.rows) - 1, idx + k)
            cand = []
            for j in range(left, right + 1):
                cand += self.rows[j][2]
            if cand:
                cand = sorted(cand)
                merged = []
                for a, b in cand:
                    if not merged or a > merged[-1][1]:
                        merged.append([a, b])
                    else:
                        merged[-1][1] = max(merged[-1][1], b)
                return [(float(a), float(b)) for a, b in merged]
        return []

# Build support from control medians
support = SupportByDuration(
    d_median=whistle_songs_control["d_median"].to_numpy(),
    f_median=whistle_songs_control["f_median"].to_numpy(),
    bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
    gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_S
)

# ------------------------------------------------------------
# CONTROL (d,f) KDE for blended control probabilities
# ------------------------------------------------------------
class ControlDFKDE:
    """2D control KDE over control medians with marginals & component weight helpers."""
    def __init__(self, df_ctrl, bw_adjust=KDE_BW_ADJUST, gridsize=220, cut_pad=CUT_PAD):
        d = np.asarray(df_ctrl["d_median"], float)
        f = np.asarray(df_ctrl["f_median"], float)
        m = np.isfinite(d) & np.isfinite(f)
        d = d[m]; f = f[m]

        # grids
        x_min, x_max = d.min(), d.max()
        y_min, y_max = f.min(), f.max()
        x_pad = (x_max - x_min) * cut_pad
        y_pad = (y_max - y_min) * cut_pad
        self.d_grid = np.linspace(x_min - x_pad, x_max + x_pad, gridsize)
        self.f_grid = np.linspace(y_min - y_pad, y_max + y_pad, gridsize)
        X, Y = np.meshgrid(self.d_grid, self.f_grid, indexing="ij")

        # fit KDE like support
        kde = gaussian_kde(np.vstack([d, f]))
        kde.covariance_factor = lambda: kde.scotts_factor() * bw_adjust
        kde._compute_covariance()
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(gridsize, gridsize)
        self.Z = np.clip(Z, 0, None)
        self.kde = kde

        # numeric steps for Riemann sums
        self.dd = np.gradient(self.d_grid)
        self.df = np.gradient(self.f_grid)

        # marginals
        self.p_d = (self.Z * self.df[np.newaxis, :]).sum(axis=1)
        self.p_f = (self.Z * self.dd[:, np.newaxis]).sum(axis=0)
        self.p_d = self.p_d / (self.p_d.sum() if self.p_d.sum() > 0 else 1.0)
        self.p_f = self.p_f / (self.p_f.sum() if self.p_f.sum() > 0 else 1.0)

    def _eval_along_f(self, d_fixed, f_vals):
        pts = np.vstack([np.full_like(f_vals, float(d_fixed)), f_vals])
        return np.clip(self.kde(pts), 0, None)

    def ctrl_comp_weights_given_d(self, d_fixed, mu, w_prior=None):
        """p_ctrl(k|d) ∝ p_ctrl(d, f=mu_k) × w_prior[k]."""
        vals = self._eval_along_f(d_fixed, np.asarray(mu, float))
        if w_prior is None:
            w_prior = _align_global_weights(mu)
        comp = np.clip(vals, EPS, None) * np.asarray(w_prior, float)
        return comp / comp.sum()

    def ctrl_comp_weights_marginal_f(self, mu, w_prior=None):
        """p_ctrl(k) from marginal p_ctrl(f=mu_k) × w_prior[k]."""
        mu = np.asarray(mu, float)
        pf = np.interp(mu, self.f_grid, self.p_f, left=EPS, right=EPS)
        if w_prior is None:
            w_prior = _align_global_weights(mu)
        comp = np.clip(pf, EPS, None) * np.asarray(w_prior, float)
        return comp / comp.sum()

    def ctrl_duration_mode_weights(self, mu_log):
        """Control weights over duration modes using duration marginal at each mode median."""
        d_modes = np.exp(np.asarray(mu_log, float))  # median of each lognormal mode
        pd = np.interp(d_modes, self.d_grid, self.p_d, left=EPS, right=EPS)
        w = np.clip(pd, EPS, None)
        return w / w.sum()

ctrl_kde = ControlDFKDE(whistle_songs_control)

# ------------------------------------------------------------
# LOAD PER-BIRD PARAMS (duration & pitch)
# ------------------------------------------------------------
def load_bird_params_from_excel_duration(path, expected_modes=MODE_LABELS):
    df = pd.read_excel(path)
    required = {"sim_bird", "mode", "mu_sec", "sigma_sec"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Duration Excel missing columns: {miss}")
    df = df[df["mode"].isin(expected_modes)].copy()
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
            missing = [expected_modes[j] for j in np.where(np.isnan(mu_log_birds[i]))[0]]
            raise ValueError(f"Bird '{b}' missing duration modes: {missing}")
    return birds, mu_log_birds, sd_log_birds

def load_bird_params_from_excel_pitch(path):
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "bird" in df.columns:
        bird_col = "bird"
    elif "sim_bird" in df.columns:
        bird_col = "sim_bird"
    else:
        raise ValueError("Pitch Excel must have 'bird' or 'sim_bird'.")
    mu_candidates = ["mu_hz", "mu (hz)", "mu"]
    sg_candidates = ["sigma_hz", "sigma (hz)", "sigma"]
    try:
        mu_col = next(c for c in mu_candidates if c in df.columns)
        sg_col = next(c for c in sg_candidates if c in df.columns)
    except StopIteration:
        raise ValueError("Pitch Excel must include μ and σ columns.")
    cols = [bird_col, mu_col, sg_col] + (["weight"] if "weight" in df.columns else [])
    df_use = df[cols].rename(columns={bird_col: "bird", mu_col: "mu_hz", sg_col: "sigma_hz"}).copy()
    df_use = df_use.replace([np.inf, -np.inf], np.nan).dropna(subset=["bird", "mu_hz", "sigma_hz"])
    df_use["sigma_hz"] = df_use["sigma_hz"].clip(lower=1e-6)
    bird_params = {}
    for b, sub in df_use.groupby("bird", sort=False):
        sub = sub.sort_values("mu_hz")
        mu = sub["mu_hz"].to_numpy(float)
        sd = sub["sigma_hz"].to_numpy(float)
        if "weight" in sub.columns:
            w = sub["weight"].to_numpy(float)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones_like(mu, float)
        else:
            w = np.ones_like(mu, float)
        bird_params[str(b)] = dict(mu=mu, sigma=sd, w=w / w.sum())
    birds = sorted(bird_params.keys())
    if not birds:
        raise ValueError("No birds found in pitch Excel after cleaning.")
    return bird_params, birds

birds_dur, mu_log_birds, sd_log_birds = load_bird_params_from_excel_duration(EXCEL_DURATION)
bird_params_pitch, birds_pitch = load_bird_params_from_excel_pitch(EXCEL_PITCH)

# ------------------------------------------------------------
# TRUNCATED MIXTURE SAMPLING WITH BLENDING (pitch | duration support)
# ------------------------------------------------------------
def _blend(p_eng, p_ctrl, delta=DELTA_BLEND):
    p_eng = np.asarray(p_eng, float); p_ctrl = np.asarray(p_ctrl, float)
    out = float(delta) * p_eng + (1.0 - float(delta)) * p_ctrl
    out = np.clip(out, EPS, None)
    return out / out.sum()

def sample_pitch_given_support_blended(
    playback_f_hz, params, support_intervals, rng,
    ctrl_comp_weights,
    tau=400.0, p=2.0, kind="cauchy",
    alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0,
    delta=DELTA_BLEND, beta_pitch=BETA_PITCH,
):
    """Truncated Gaussian mixture: component weights ∝ (blended probs × truncated mass)."""
    mu, sd = params["mu"], params["sigma"]

    # engaged (attractor) over components
    p_eng = attractor_probs_local(playback_f_hz, mu, tau=tau, p=p, kind=kind,
                                  alpha_bg=alpha_bg, nearest_boost=nearest_boost,
                                  rq_nu=rq_nu, beta=beta_pitch)
    # blended base (before truncation mass)
    p_base = _blend(p_eng, ctrl_comp_weights, delta=delta)

    # truncated mass of each component inside union intervals
    def _mass_for_k(k_idx):
        mass = 0.0
        segs = []
        for (a, b) in support_intervals:
            Fa = normal_cdf(a, mu[k_idx], sd[k_idx])
            Fb = normal_cdf(b, mu[k_idx], sd[k_idx])
            m = max(0.0, float(Fb - Fa))
            mass += m
            segs.append((Fa, Fb, m))
        return mass, segs

    masses = []
    segstore = []
    for k in range(len(mu)):
        m_k, segs_k = _mass_for_k(k)
        masses.append(m_k); segstore.append(segs_k)
    masses = np.asarray(masses, float)

    w = p_base * np.clip(masses, 0.0, None)
    S = w.sum()
    if not np.isfinite(S) or S <= 1e-12:
        # no mass anywhere → snap to nearest support boundary to playback pitch
        bounds = np.array([v for ab in support_intervals for v in ab], float)
        return float(bounds[np.argmin(np.abs(bounds - playback_f_hz))])

    pis = w / S
    k = rng.choice(len(mu), p=pis)

    # choose interval for selected k proportional to its segment mass
    segs = segstore[k]
    mlist = np.array([m for (_, _, m) in segs], float)
    mlist = mlist / (mlist.sum() if mlist.sum() > 0 else 1.0)
    idx = rng.choice(len(segs), p=mlist)
    Fa, Fb, _ = segs[idx]
    u = rng.random()
    return float(normal_ppf(Fa + u * max(EPS, (Fb - Fa)), mu[k], sd[k]))

# ------------------------------------------------------------
# JOINT SIMULATION (Duration → Pitch) with blending
# ------------------------------------------------------------
def simulate_joint(playbacks, birds_dur, mu_log_b, sd_log_b,
                   bird_params_pitch, birds_pitch, support: SupportByDuration,
                   n_renditions=50, tau_hz=400.0, kind_pitch="cauchy",
                   p_exp=2.0, alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    b2i = {b: i for i, b in enumerate(birds_dur)}

    for b in birds_pitch:
        if b not in b2i:
            warnings.warn(f"Bird '{b}' in pitch Excel not found in duration Excel. Skipping.")
            continue
        bi = b2i[b]
        mu_log_k = mu_log_b[bi]
        sd_log_k = sd_log_b[bi]
        pitch_params = bird_params_pitch[b]

        # control weights over duration modes (constant per bird)
        ctrl_w_dur = ctrl_kde.ctrl_duration_mode_weights(mu_log_k)

        for (d_in, f_in) in playbacks:
            for r in range(n_renditions):
                # ---- Duration: engaged gate (playback duration) & control marginal blend
                p_d_eng = attractor_probs_no_alpha(d_in, mu_log_k, sd_log_k, w_ctrl=W_CTRL, beta=BETA_DUR)
                p_d = _blend(p_d_eng, ctrl_w_dur, delta=DELTA_BLEND)
                k_d = rng.choice(len(mu_log_k), p=p_d)
                y_log = rng.normal(loc=mu_log_k[k_d], scale=sd_log_k[k_d])
                x_dur = float(np.exp(y_log))

                # Support intervals for this duration
                intervals = support.intervals_for_duration(x_dur) or \
                            support.intervals_near_duration(x_dur, max_neighbors=3)
                if not intervals:
                    rows.append({
                        "bird": b, "rendition": r,
                        "playback_d_s": d_in, "playback_f_hz": f_in,
                        "resp_d_s": x_dur, "resp_f_hz": np.nan,
                        "dur_mode": int(k_d)
                    })
                    continue

                # ---- Pitch: control component weights from control KDE @ this duration
                mu = pitch_params["mu"]
                w_prior = _align_global_weights(mu)  # mild prior alignment
                ctrl_comp = ctrl_kde.ctrl_comp_weights_given_d(x_dur, mu, w_prior=w_prior)

                # blended + truncated mixture sampling
                # Pitch gate
                y_pitch = sample_pitch_given_support_blended(
                    playback_f_hz=f_in, params=pitch_params, support_intervals=intervals, rng=rng,
                    ctrl_comp_weights=ctrl_comp,
                    tau=tau_hz, p=p_exp, kind=kind_pitch,
                    alpha_bg=alpha_bg, nearest_boost=nearest_boost, rq_nu=rq_nu,
                    delta=DELTA_BLEND, beta_pitch=BETA_PITCH
                    )

                rows.append({
                    "bird": b, "rendition": r,
                    "playback_d_s": d_in, "playback_f_hz": f_in,
                    "resp_d_s": x_dur, "resp_f_hz": y_pitch,
                    "dur_mode": int(k_d)
                })
    return pd.DataFrame(rows)

# ============================================================
# Experiment 2 (unchanged plotting; now uses blended simulation)
# ============================================================

# ---------- Playback definitions (pitch in kHz → Hz) ----------
playback_exp2_d_A = [0.14, 0.14, 0.14]
playback_exp2_f_A = [6, 7, 8]          # kHz
playback_exp2_d_B = [0.6, 0.7, 0.8]
playback_exp2_f_B = [8, 7, 6]          # kHz
playback_exp2_d_C = [0.6, 0.7, 0.8]
playback_exp2_f_C = [1, 2, 3]          # kHz

playback_data_exp2 = {
    "A": (playback_exp2_d_A, playback_exp2_f_A),
    "B": (playback_exp2_d_B, playback_exp2_f_B),
    "C": (playback_exp2_d_C, playback_exp2_f_C),
}

def build_playbacks(d_list, f_khz_list):
    f_hz = [float(k)*1000.0 for k in f_khz_list]
    return list(zip(d_list, f_hz))

playbacks_by_region = {
    region: build_playbacks(d_list, f_list)
    for region, (d_list, f_list) in playback_data_exp2.items()
}

# ---------- Simulation settings ----------
N_RENDITIONS = 50   # per bird, per (d,f)
TAU_HZ = 100.0
KIND_PITCH = "cauchy"
P_EXP = 2.0
ALPHA_BG = 0.0
NEAREST_BOOST = 0.1
RQ_NU = 1.0
BASE_SEED = 123

# These match your support/KDE config from earlier
KDE_BW_ADJUST = 0.7
SUPPORT_THRESH = 0.10

# Colormap for KDE fill; use same hue for contour
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])
kde_contour_color = "olivedrab"

# ---------- Simulate each region (Duration → Pitch, blended) ----------
sim_by_region = {}
seed = BASE_SEED
for region, pb_list in playbacks_by_region.items():
    sim_by_region[region] = simulate_joint(
        playbacks=pb_list,
        birds_dur=birds_dur, mu_log_b=mu_log_birds, sd_log_b=sd_log_birds,
        bird_params_pitch=bird_params_pitch, birds_pitch=birds_pitch,
        support=support,
        n_renditions=N_RENDITIONS,
        tau_hz=TAU_HZ, kind_pitch=KIND_PITCH, p_exp=P_EXP,
        alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BOOST, rq_nu=RQ_NU,
        seed=seed
    )
    sim_by_region[region]["region"] = region
    seed += 101
# ============================================================
# Exp. 2 — Duration→Pitch: per-region KDE and KDE-ratio (2×3)
# Styling/behavior matches your example code
# ============================================================

# Global bins / grids & styling
global_x_bins = np.linspace(0, 1, 100)
global_y_bins = np.linspace(0, 10000, 100)

plt.style.use('default')
vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Build figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
region_keys = ["A", "B", "C"]

for i, region in enumerate(region_keys):
    # --- pull simulated data (resp_d_s, resp_f_hz)
    sim = sim_by_region[region]
    m = np.isfinite(sim["resp_d_s"]) & np.isfinite(sim["resp_f_hz"])
    data = sim.loc[m].copy()

    # playback points (already in Hz here!)
    pb = playbacks_by_region[region]
    if len(pb):
        playback_d, playback_f = zip(*pb)
    else:
        playback_d, playback_f = [], []

    # ------- First row: KDE + scatter + control contour -------
    ax_kde = axes[0, i]

    # playback markers
    ax_kde.plot(
        playback_d, playback_f,
        color="#12B568", marker='+', linestyle='None',
        markersize=6, markeredgewidth=3
    )

    # control outer contour
    sns.kdeplot(
        x=whistle_songs_control.d_median,
        y=whistle_songs_control.f_median,
        cmap="Reds_r", fill=False,
        bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde
    )

    # region KDE (simulated) + scatter
    if len(data) >= 5:
        sns.kdeplot(
            x=data["resp_d_s"], y=data["resp_f_hz"],
            cmap=custom_cmap, fill=True,
            bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1
        )
    #ax_kde.scatter(data["resp_d_s"], data["resp_f_hz"], c='k', s=4, zorder=1)

    ax_kde.set_title(f"Region {region}")
    ax_kde.set_xlim([0, 0.9])
    ax_kde.set_ylim([0, 9000])
    ax_kde.set_box_aspect(1)

    # ------------- Second row: KDE ratio vs control -----------
    ax_ratio = axes[1, i]

    # KDEs (exp/sim vs control) on common grid
    x_exp = data["resp_d_s"].to_numpy()
    y_exp = data["resp_f_hz"].to_numpy()

    if len(x_exp) >= 5:
        kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
    else:
        # empty safe KDE
        kde_exp = None

    x_ctrl = whistle_songs_control.d_median.to_numpy()
    y_ctrl = whistle_songs_control.f_median.to_numpy()
    valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

    x_grid = np.linspace(0, 0.9, 100)
    y_grid = np.linspace(0, 9000, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    if kde_exp is not None:
        Z_exp = kde_exp(positions).reshape(X.shape)
    else:
        Z_exp = np.zeros_like(X)

    Z_ctrl = kde_ctrl(positions).reshape(X.shape)

    # ratio (clip like your code, add small floor to denom)
    ratio = np.clip(Z_exp / (Z_ctrl + 0.0003), vmin, vmax)

    # filled contours
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)

    # overlay control contour
    sns.kdeplot(
        x=whistle_songs_control.d_median,
        y=whistle_songs_control.f_median,
        cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio
    )

    # playback markers
    ax_ratio.plot(
        playback_d, playback_f,
        color="#12B568", marker='+', linestyle='None',
        markersize=6, markeredgewidth=3
    )

    ax_ratio.set_xlim([0, 0.9])
    ax_ratio.set_ylim([0, 9000])
    ax_ratio.set_box_aspect(1)

# layout + shared labels + colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')
fig.suptitle("Attractor (β duration=1 β pitch=3, δ=0.6)")
mpl.rcParams['pdf.fonttype'] = 42
os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_2_sim_duration_first_and_ratio.pdf', transparent=True)
plt.show()


# ============================================================
# Pitch → Duration variant (pick pitch first, then duration)
# Same plotting and ratio analysis as before
# ============================================================

# --- Support (pitch → duration intervals) -------------------
BIN_WIDTH_HZ = 100.0  # pitch bin size for support map

class SupportByPitch:
    """For a given pitch, return the union of valid duration intervals (from control KDE mask)."""
    def __init__(self, d_median, f_median,
                 bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
                 gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_HZ):
        d = np.asarray(d_median, float)
        f = np.asarray(f_median, float)
        m = np.isfinite(d) & np.isfinite(f)
        d, f = d[m], f[m]
        if d.size < 5:
            raise ValueError("Not enough points to build support (pitch-first).")

        x_min, x_max = d.min(), d.max()
        y_min, y_max = f.min(), f.max()
        x_pad = (x_max - x_min) * cut_pad
        y_pad = (y_max - y_min) * cut_pad
        self.dd = np.linspace(x_min - x_pad, x_max + x_pad, gridsize)  # durations
        self.ff = np.linspace(y_min - y_pad, y_max + y_pad, gridsize)  # pitches
        D, F = np.meshgrid(self.dd, self.ff, indexing="ij")

        kde = gaussian_kde(np.vstack([d, f]))
        kde.covariance_factor = lambda: kde.scotts_factor() * bw_adjust
        kde._compute_covariance()
        Z = kde(np.vstack([D.ravel(), F.ravel()])).reshape(gridsize, gridsize)

        self.level = float(Z.max()) * float(thresh)
        self.mask = (Z >= self.level)
        self.bin_width = float(bin_width)

        # Precompute per pitch-bin duration unions
        rows = []
        edges = np.arange(self.ff.min(), self.ff.max() + bin_width, bin_width)
        for low, high in zip(edges[:-1], edges[1:]):
            y_rows = np.where((self.ff >= low) & (self.ff < high))[0]
            if y_rows.size == 0:
                rows.append((low, high, []))
                continue
            row_any = self.mask[:, y_rows].any(axis=1)  # along duration axis
            intervals = self._intervals_from_bool(row_any, self.dd)
            rows.append((low, high, intervals))
        self.rows = rows  # list of (f_low, f_high, [(d0,d1), ...])

    @staticmethod
    def _intervals_from_bool(bool_vec, x_axis):
        idx = np.flatnonzero(bool_vec)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [(float(x_axis[g[0]]), float(x_axis[g[-1]])) for g in groups]

    def intervals_for_pitch(self, y):
        y = float(y)
        for (lo, hi, ints) in self.rows:
            if lo <= y < hi:
                return ints
        return []

    def intervals_near_pitch(self, y, max_neighbors=3):
        y = float(y)
        edges = np.array([r[0] for r in self.rows] + [self.rows[-1][1]])
        centers = 0.5 * (edges[:-1] + edges[1:])
        if centers.size == 0:
            return []
        idx = int(np.argmin(np.abs(centers - y)))
        for k in range(max_neighbors + 1):
            left = max(0, idx - k); right = min(len(self.rows) - 1, idx + k)
            cand = []
            for j in range(left, right + 1):
                cand += self.rows[j][2]
            if cand:
                cand = sorted(cand)
                merged = []
                for a, b in cand:
                    if not merged or a > merged[-1][1]:
                        merged.append([a, b])
                    else:
                        merged[-1][1] = max(merged[-1][1], b)
                return [(float(a), float(b)) for a, b in merged]
        return []

support_by_pitch = SupportByPitch(
    d_median=whistle_songs_control["d_median"].to_numpy(),
    f_median=whistle_songs_control["f_median"].to_numpy(),
    bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
    gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_HZ
)

# --- Helpers for lognormal (duration modes) ------------------
def lognormal_cdf(x, mu_log, sd_log):
    x = float(x)
    if x <= 0:
        return 0.0
    return normal_cdf(np.log(x), mu_log, sd_log)

def lognormal_ppf(p, mu_log, sd_log):
    return float(np.exp(normal_ppf(p, mu_log, sd_log)))

def ctrl_duration_mode_weights_given_f(ctrl_kde, f_fixed, mu_log_vec):
    """Control weights over duration modes given a fixed pitch f_fixed."""
    d_modes = np.exp(np.asarray(mu_log_vec, float))  # medians of lognormal modes
    pts = np.vstack([d_modes, np.full_like(d_modes, float(f_fixed))])
    vals = np.clip(ctrl_kde.kde(pts), EPS, None)
    s = vals.sum()
    return vals / (s if s > 0 else 1.0)

def sample_duration_given_support_blended(
    playback_d_s, mu_log_vec, sd_log_vec, intervals, rng,
    ctrl_w_dur, delta=DELTA_BLEND
):
    """Truncated lognormal mixture for duration with blended (engaged vs control) weights."""
    mu_log_vec = np.asarray(mu_log_vec, float)
    sd_log_vec = np.asarray(sd_log_vec, float)

    # engaged gate from playback duration
    p_eng = attractor_probs_no_alpha(playback_d_s, mu_log_vec, sd_log_vec, w_ctrl=W_CTRL, beta=BETA_DUR)
    p_base = _blend(p_eng, ctrl_w_dur, delta=delta)

    # truncated mass per mode inside union of duration intervals
    masses = []
    segstore = []
    for k in range(len(mu_log_vec)):
        mu_k, sd_k = mu_log_vec[k], sd_log_vec[k]
        mass_k = 0.0
        segs_k = []
        for (a, b) in intervals:
            Fa = lognormal_cdf(a, mu_k, sd_k)
            Fb = lognormal_cdf(b, mu_k, sd_k)
            m = max(0.0, float(Fb - Fa))
            mass_k += m
            segs_k.append((Fa, Fb, m))
        masses.append(mass_k)
        segstore.append(segs_k)
    masses = np.asarray(masses, float)

    w = p_base * np.clip(masses, 0.0, None)
    S = w.sum()
    if not np.isfinite(S) or S <= 1e-12:
        # no mass anywhere → snap to nearest support boundary to requested playback duration
        bounds = np.array([v for ab in intervals for v in ab], float)
        return float(bounds[np.argmin(np.abs(bounds - playback_d_s))])

    pis = w / S
    k = rng.choice(len(mu_log_vec), p=pis)

    # Choose a segment proportional to its mass, then inverse-cdf sample within
    segs = segstore[k]
    mlist = np.array([m for (_, _, m) in segs], float)
    mlist = mlist / (mlist.sum() if mlist.sum() > 0 else 1.0)
    idx = rng.choice(len(segs), p=mlist)
    Fa, Fb, _ = segs[idx]
    u = rng.random()
    return lognormal_ppf(Fa + u * max(EPS, (Fb - Fa)), mu_log_vec[k], sd_log_vec[k])

# --- Simulation: Pitch then Duration -------------------------
def simulate_joint_pitch_first(playbacks, birds_dur, mu_log_b, sd_log_b,
                               bird_params_pitch, birds_pitch, support_by_pitch: SupportByPitch,
                               n_renditions=50, tau_hz=400.0, kind_pitch="cauchy",
                               p_exp=2.0, alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0, seed=777):
    rng = np.random.default_rng(seed)
    rows = []
    b2i = {b: i for i, b in enumerate(birds_dur)}

    for b in birds_pitch:
        if b not in b2i:
            warnings.warn(f"Bird '{b}' in pitch Excel not found in duration Excel. Skipping.")
            continue
        bi = b2i[b]
        mu_log_k = mu_log_b[bi]
        sd_log_k = sd_log_b[bi]
        pitch_params = bird_params_pitch[b]
        mu = pitch_params["mu"]; sd = pitch_params["sigma"]
        w_prior = _align_global_weights(mu)

        # control comp weights for pitch: use marginal p_ctrl(f=mu_k)
        ctrl_comp_pitch = ctrl_kde.ctrl_comp_weights_marginal_f(mu, w_prior=w_prior)

        for (d_in, f_in) in playbacks:
            for r in range(n_renditions):
                # ---- Pitch: engaged gate & control-marginal blend (no truncation here)
                p_eng_pitch = attractor_probs_local(
                    f_in, mu, tau=tau_hz, p=p_exp, kind=kind_pitch,
                    alpha_bg=alpha_bg, nearest_boost=nearest_boost, rq_nu=rq_nu, beta=BETA_PITCH
                )
                p_pitch = _blend(p_eng_pitch, ctrl_comp_pitch, delta=DELTA_BLEND)
                k_p = rng.choice(len(mu), p=p_pitch)
                y_pitch = float(rng.normal(loc=mu[k_p], scale=sd[k_p]))

                # Duration support for this chosen pitch
                intervals = support_by_pitch.intervals_for_pitch(y_pitch) or \
                            support_by_pitch.intervals_near_pitch(y_pitch, max_neighbors=3)

                if not intervals:
                    # Can't place a duration for this pitch
                    rows.append({
                        "bird": b, "rendition": r,
                        "playback_d_s": d_in, "playback_f_hz": f_in,
                        "resp_d_s": np.nan, "resp_f_hz": y_pitch,
                        "dur_mode": np.nan
                    })
                    continue

                # ---- Duration: control weights conditioned on this pitch
                ctrl_w_dur_f = ctrl_duration_mode_weights_given_f(ctrl_kde, y_pitch, mu_log_k)

                # Truncated mixture sampling for duration inside allowed intervals
                x_dur = sample_duration_given_support_blended(
                    playback_d_s=d_in,
                    mu_log_vec=mu_log_k, sd_log_vec=sd_log_k,
                    intervals=intervals, rng=rng,
                    ctrl_w_dur=ctrl_w_dur_f, delta=DELTA_BLEND
                )

                # Identify which duration mode the sampled x_dur likely came from (MAP over modes)
                # (purely diagnostic; optional)
                r_scores = -0.5 * ((np.log(x_dur) - mu_log_k) / sd_log_k) ** 2 + np.log(W_CTRL)
                k_d = int(np.argmax(r_scores))

                rows.append({
                    "bird": b, "rendition": r,
                    "playback_d_s": d_in, "playback_f_hz": f_in,
                    "resp_d_s": x_dur, "resp_f_hz": y_pitch,
                    "dur_mode": k_d
                })
    return pd.DataFrame(rows)

# ---------- Simulate each region (Pitch → Duration) ----------
sim_by_region_pf = {}
seed_pf = BASE_SEED + 202  # different seed from duration→pitch path
for region, pb_list in playbacks_by_region.items():
    sim_by_region_pf[region] = simulate_joint_pitch_first(
        playbacks=pb_list,
        birds_dur=birds_dur, mu_log_b=mu_log_birds, sd_log_b=sd_log_birds,
        bird_params_pitch=bird_params_pitch, birds_pitch=birds_pitch,
        support_by_pitch=support_by_pitch,
        n_renditions=N_RENDITIONS,
        tau_hz=TAU_HZ, kind_pitch=KIND_PITCH, p_exp=P_EXP,
        alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BOOST, rq_nu=RQ_NU,
        seed=seed_pf
    )
    sim_by_region_pf[region]["region"] = region
    seed_pf += 131

# ============================================================
# Exp. 2 — Pitch→Duration: per-region KDE and KDE-ratio (2×3)
# Mirrors the exact plotting style as above
# ============================================================

plt.style.use('default')
vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
region_keys = ["A", "B", "C"]

for i, region in enumerate(region_keys):
    sim = sim_by_region_pf[region]
    m = np.isfinite(sim["resp_d_s"]) & np.isfinite(sim["resp_f_hz"])
    data = sim.loc[m].copy()

    pb = playbacks_by_region[region]
    if len(pb):
        playback_d, playback_f = zip(*pb)
    else:
        playback_d, playback_f = [], []

    # ------- First row: KDE + scatter + control contour -------
    ax_kde = axes[0, i]

    ax_kde.plot(
        playback_d, playback_f,
        color="#12B568", marker='+', linestyle='None',
        markersize=6, markeredgewidth=3
    )

    sns.kdeplot(
        x=whistle_songs_control.d_median,
        y=whistle_songs_control.f_median,
        cmap="Reds_r", fill=False,
        bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde
    )

    if len(data) >= 5:
        sns.kdeplot(
            x=data["resp_d_s"], y=data["resp_f_hz"],
            cmap=custom_cmap, fill=True,
            bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1
        )
    #ax_kde.scatter(data["resp_d_s"], data["resp_f_hz"], c='k', s=4, zorder=1)

    ax_kde.set_title(f"Region {region}")
    ax_kde.set_xlim([0, 0.9])
    ax_kde.set_ylim([0, 9000])
    ax_kde.set_box_aspect(1)

    # ------------- Second row: KDE ratio vs control -----------
    ax_ratio = axes[1, i]

    x_exp = data["resp_d_s"].to_numpy()
    y_exp = data["resp_f_hz"].to_numpy()
    if len(x_exp) >= 5:
        kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
    else:
        kde_exp = None

    x_ctrl = whistle_songs_control.d_median.to_numpy()
    y_ctrl = whistle_songs_control.f_median.to_numpy()
    valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

    x_grid = np.linspace(0, 0.9, 100)
    y_grid = np.linspace(0, 9000, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    if kde_exp is not None:
        Z_exp = kde_exp(positions).reshape(X.shape)
    else:
        Z_exp = np.zeros_like(X)
    Z_ctrl = kde_ctrl(positions).reshape(X.shape)

    ratio = np.clip(Z_exp / (Z_ctrl + 0.0003), vmin, vmax)
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)

    sns.kdeplot(
        x=whistle_songs_control.d_median,
        y=whistle_songs_control.f_median,
        cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio
    )

    ax_ratio.plot(
        playback_d, playback_f,
        color="#12B568", marker='+', linestyle='None',
        markersize=6, markeredgewidth=3
    )

    ax_ratio.set_xlim([0, 0.9])
    ax_ratio.set_ylim([0, 9000])
    ax_ratio.set_box_aspect(1)

# layout + shared labels + colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')
fig.suptitle("Attractor (β duration=1 β pitch=3, δ=0.6)")
mpl.rcParams['pdf.fonttype'] = 42
os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_2_sim_pitch_first_and_ratio.pdf', transparent=True)
plt.show()

# ============================================================
# Experiment: Duration fixed at 0.44 s, pitch sweep
# (Duration → Pitch path; KDE + KDE ratio)
# ============================================================

# --- Define playbacks (Hz) ---
pitches_hz = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400,
              3600, 4000, 4500, 5000, 6000, 7000, 8000]
playbacks_sweep = [(0.44, float(p)) for p in pitches_hz]

# --- Simulate (uses existing settings: N_RENDITIONS, TAU_HZ, etc.) ---
sim_sweep = simulate_joint(
    playbacks=playbacks_sweep,
    birds_dur=birds_dur, mu_log_b=mu_log_birds, sd_log_b=sd_log_birds,
    bird_params_pitch=bird_params_pitch, birds_pitch=birds_pitch,
    support=support,
    n_renditions=N_RENDITIONS,
    tau_hz=TAU_HZ, kind_pitch=KIND_PITCH, p_exp=P_EXP,
    alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BOOST, rq_nu=RQ_NU,
    seed=BASE_SEED + 999
)

# --- Plot: top = KDE with playback markers; bottom = KDE ratio vs control ---
plt.style.use('default')
vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig, (ax_kde, ax_ratio) = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True)

# ---------- First row: KDE + control contour + playback markers ----------
m = np.isfinite(sim_sweep["resp_d_s"]) & np.isfinite(sim_sweep["resp_f_hz"])
data = sim_sweep.loc[m].copy()

# playback markers (all duration=0.44 s)
pb_d, pb_f = zip(*playbacks_sweep)
ax_kde.plot(pb_d, pb_f, color="#12B568", marker='+', linestyle='None',
            markersize=6, markeredgewidth=3)

# control outer contour
sns.kdeplot(
    x=whistle_songs_control.d_median,
    y=whistle_songs_control.f_median,
    cmap="Reds_r", fill=False,
    bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH, ax=ax_kde
)

# simulated KDE
if len(data) >= 5:
    sns.kdeplot(
        x=data["resp_d_s"], y=data["resp_f_hz"],
        cmap=LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"]),
        fill=True, bw_adjust=KDE_BW_ADJUST, levels=10, thresh=SUPPORT_THRESH,
        ax=ax_kde, zorder=-1
    )

ax_kde.set_title("Sweep (d = 0.44 s)")
ax_kde.set_xlim([0, 0.9])
ax_kde.set_ylim([0, 9000])
ax_kde.set_box_aspect(1)

# ---------- Second row: KDE ratio (simulated / control) ----------
x_exp = data["resp_d_s"].to_numpy()
y_exp = data["resp_f_hz"].to_numpy()
if len(x_exp) >= 5:
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
else:
    kde_exp = None

x_ctrl = whistle_songs_control.d_median.to_numpy()
y_ctrl = whistle_songs_control.f_median.to_numpy()
valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

x_grid = np.linspace(0, 0.9, 100)
y_grid = np.linspace(0, 9000, 100)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()])

if kde_exp is not None:
    Z_exp = kde_exp(positions).reshape(X.shape)
else:
    Z_exp = np.zeros_like(X)
Z_ctrl = kde_ctrl(positions).reshape(X.shape)

ratio = np.clip(Z_exp / (Z_ctrl + 3e-4), vmin, vmax)
ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)

# overlay control outer contour and playback markers
sns.kdeplot(
    x=whistle_songs_control.d_median,
    y=whistle_songs_control.f_median,
    cmap="Reds_r", fill=False, bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH, ax=ax_ratio
)
ax_ratio.plot(pb_d, pb_f, color="#12B568", marker='+', linestyle='None',
              markersize=6, markeredgewidth=3)

ax_ratio.set_xlim([0, 0.9])
ax_ratio.set_ylim([0, 9000])
ax_ratio.set_box_aspect(1)

# --- Shared labels, colorbar, save ---
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

for ax in (ax_kde, ax_ratio):
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')

fig.suptitle("Sweep (d = 0.44 s) — Attractor (β duration=%s β pitch=%s, δ=%.2f)" % (BETA_DUR, BETA_PITCH, DELTA_BLEND))
mpl.rcParams['pdf.fonttype'] = 42
os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_sweep_d044_duration_first_and_ratio.pdf', transparent=True)
plt.show()

# ============================================================
# Sweep (d = 0.44 s) — Duration-only distribution
# ============================================================

# Define sweep playbacks (Hz), fixed duration
pitches_hz = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400,
              3600, 4000, 4500, 5000, 6000, 7000, 8000]
playbacks_sweep = [(0.44, float(p)) for p in pitches_hz]

# Simulate via Duration→Pitch path (reuse your settings)
sim_sweep = simulate_joint(
    playbacks=playbacks_sweep,
    birds_dur=birds_dur, mu_log_b=mu_log_birds, sd_log_b=sd_log_birds,
    bird_params_pitch=bird_params_pitch, birds_pitch=birds_pitch,
    support=support,
    n_renditions=N_RENDITIONS,
    tau_hz=TAU_HZ, kind_pitch=KIND_PITCH, p_exp=P_EXP,
    alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BOOST, rq_nu=RQ_NU,
    seed=BASE_SEED + 4242
)

# Collect finite durations
m = np.isfinite(sim_sweep["resp_d_s"])
dur = sim_sweep.loc[m, "resp_d_s"].to_numpy()

# Plot: histogram (probability per bin) + 1D KDE × binwidth
plt.style.use('default')
fig, ax = plt.subplots(figsize=(7, 4))

# 20 ms bins
bins = np.arange(0, 0.9 + BIN_WIDTH_S, BIN_WIDTH_S)

# KDE scaled by bin width (so y is probability per bin)
x_grid = np.linspace(0, 0.9, 400)
kde1d = gaussian_kde(dur)
kde1d.covariance_factor = lambda: kde1d.scotts_factor() * 0.8  # match your bw_adjust
kde1d._compute_covariance()
y_density = kde1d(x_grid)
y_prob = y_density * BIN_WIDTH_S  # convert density to probability per bin

# Fill + black outline
ax.fill_between(x_grid, 0, y_prob, alpha=0.6, facecolor='lightseagreen')
ax.plot(x_grid, y_prob, color='k', linewidth=2)

# Mark playback duration
ax.axvline(0.44, linestyle='--', color='k', linewidth=1.2,
           label='Playback duration = 0.44 s')

ax.set_xlim(0, 0.9)
ax.set_ylim(bottom=0)
ax.set_xlabel('Whistle duration (s)')
ax.set_ylabel('Probability')
ax.set_title(f'Simulation pitch paper (β_dur={BETA_DUR}, β_pitch={BETA_PITCH}, δ={DELTA_BLEND})')
ax.legend(frameon=False)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_sweep_d044_duration_only.pdf', transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# (Optional) quick stats in console
print(f"N = {dur.size}, mean = {np.nanmean(dur):.3f} s, median = {np.nanmedian(dur):.3f} s")

# ============================================================
# Sweep (d = 0.44 s) — Pitch→Duration path — Duration-only plot
# ============================================================

# Define sweep playbacks (Hz), fixed duration
pitches_hz = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400,
              3600, 4000, 4500, 5000, 6000, 7000, 8000]
playbacks_sweep_pf = [(0.44, float(p)) for p in pitches_hz]

# Simulate via Pitch→Duration path (reuse your settings)
sim_sweep_pf = simulate_joint_pitch_first(
    playbacks=playbacks_sweep_pf,
    birds_dur=birds_dur, mu_log_b=mu_log_birds, sd_log_b=sd_log_birds,
    bird_params_pitch=bird_params_pitch, birds_pitch=birds_pitch,
    support_by_pitch=support_by_pitch,
    n_renditions=N_RENDITIONS,
    tau_hz=TAU_HZ, kind_pitch=KIND_PITCH, p_exp=P_EXP,
    alpha_bg=ALPHA_BG, nearest_boost=NEAREST_BOOST, rq_nu=RQ_NU,
    seed=BASE_SEED + 5252
)

# Collect finite durations
m_pf = np.isfinite(sim_sweep_pf["resp_d_s"])
dur_pf = sim_sweep_pf.loc[m_pf, "resp_d_s"].to_numpy()

# Plot: KDE × binwidth (probability per BIN_WIDTH_S bin) + jittered raw durations below
plt.style.use('default')
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6, 2), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# 20 ms bins (kept even if not displayed)
bins = np.arange(0, 0.9 + BIN_WIDTH_S, BIN_WIDTH_S)

# KDE scaled by bin width
x_grid = np.linspace(0, 0.9, 400)
kde1d_pf = gaussian_kde(dur_pf)
kde1d_pf.covariance_factor = lambda: kde1d_pf.scotts_factor() * 0.8  # match your bw_adjust
kde1d_pf._compute_covariance()
y_density_pf = kde1d_pf(x_grid)
y_prob_pf = y_density_pf * BIN_WIDTH_S

# --- TOP: fill + black outline ---
ax_top.fill_between(x_grid, 0, y_prob_pf, alpha=1, facecolor='mediumaquamarine')
ax_top.plot(x_grid, y_prob_pf, color='k', linewidth=2)

# Mark playback duration
ax_top.axvline(0.44, linestyle='--', color='k', linewidth=1.2,
               label='Playback duration = 0.44 s')

ax_top.set_xlim(0, 0.9)
ax_top.set_ylim(bottom=0)
ax_top.set_ylabel('Probability')
ax_top.set_title(f'Simulation pitch paper (β_dur={BETA_DUR}, β_pitch={BETA_PITCH}, δ={DELTA_BLEND})')

RANDOM_STATE = 12
# --- BOTTOM: jittered raw durations (one dot per sample) ---
rng = np.random.default_rng(RANDOM_STATE if 'RANDOM_STATE' in globals() else 0)
y_jit = rng.uniform(0.03, 0.19, size=len(dur_pf))  # tiny vertical band
ax_bot.scatter(dur_pf, y_jit, s=7, c='mediumaquamarine', alpha = 0.1, rasterized=True,
              )

ax_bot.set_ylim(0, 0.22)
ax_bot.set_yticks([])
ax_bot.set_xlabel('Whistle duration (s)')

# Hide duplicate x tick labels on the top (shared x)
plt.setp(ax_top.get_xticklabels(), visible=False)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_sweep_d044_duration_only_pitchfirst.pdf', transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# ---------- How many birds & data points per experiment ----------
def count_finite(df):
    m = np.isfinite(df["resp_d_s"]) & np.isfinite(df["resp_f_hz"])
    return int(m.sum())

common_birds = sorted(set(birds_dur) & set(birds_pitch))
n_birds = len(common_birds)
print("\n==========================")
print("SIMULATION COUNTS SUMMARY")
print("==========================")
print(f"Birds in duration Excel: {len(birds_dur)}")
print(f"Birds in pitch Excel:    {len(birds_pitch)}")
print(f"Common birds simulated:  {n_birds}")
print("--------------------------")

# --- Exp. 2: Duration → Pitch (per region) ---
print("Exp. 2 — Duration→Pitch")
tot_rows = 0
tot_finite = 0
for region in ["A", "B", "C"]:
    df = sim_by_region[region]
    rows = len(df)
    finite = count_finite(df)
    tot_rows += rows
    tot_finite += finite
    # nominal rows = 3 playbacks × N_RENDITIONS × n_birds
    nominal = 3 * N_RENDITIONS * n_birds
    print(f"  Region {region}: rows={rows:,} (nominal={nominal:,}), finite(d,f)={finite:,}")
print(f"  TOTAL (A+B+C): rows={tot_rows:,}, finite(d,f)={tot_finite:,}")
print("--------------------------")

# --- Exp. 2: Pitch → Duration (per region) ---
print("Exp. 2 — Pitch→Duration")
tot_rows_pf = 0
tot_finite_pf = 0
for region in ["A", "B", "C"]:
    df = sim_by_region_pf[region]
    rows = len(df)
    finite = count_finite(df)
    tot_rows_pf += rows
    tot_finite_pf += finite
    nominal = 3 * N_RENDITIONS * n_birds
    print(f"  Region {region}: rows={rows:,} (nominal={nominal:,}), finite(d,f)={finite:,}")
print(f"  TOTAL (A+B+C): rows={tot_rows_pf:,}, finite(d,f)={tot_finite_pf:,}")
print("--------------------------")

# --- Sweep (d = 0.44 s) — Duration → Pitch ---
L_sweep = 16  # number of sweep pitches you defined
rows_sweep = len(sim_sweep)
finite_sweep = count_finite(sim_sweep)
nominal_sweep = L_sweep * N_RENDITIONS * n_birds
print("Sweep (d = 0.44 s) — Duration→Pitch")
print(f"  rows={rows_sweep:,} (nominal={nominal_sweep:,}), finite(d,f)={finite_sweep:,}")
print("--------------------------")

# --- Sweep (d = 0.44 s) — Pitch → Duration ---
rows_sweep_pf = len(sim_sweep_pf)
finite_sweep_pf = count_finite(sim_sweep_pf)
nominal_sweep_pf = L_sweep * N_RENDITIONS * n_birds
print("Sweep (d = 0.44 s) — Pitch→Duration")
print(f"  rows={rows_sweep_pf:,} (nominal={nominal_sweep_pf:,}), finite(d,f)={finite_sweep_pf:,}")
print("==========================\n")

# (Optional) quick stats in console
print(f"[Pitch→Duration] N = {dur_pf.size}, mean = {np.nanmean(dur_pf):.3f} s, median = {np.nanmedian(dur_pf):.3f} s")
