# Nightingale: control KDE → duration-conditioned support → joint (d,f) simulation
import os, warnings
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.special import erf, erfinv

# --- Paths ---
DATA_PKL       = r"Control prev season/all_birds.pkl"
EXCEL_DURATION = "simulated_bird_params_duration.xlsx"
EXCEL_PITCH    = "simulated_bird_params_pitch.xlsx"
PLOT_OUT       = r"Plots/control_meanvspitch_with_clusters.pdf"
os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)

# --- Config ---
KDE_BW_ADJUST = 0.7
SUPPORT_THRESH = 0.1
GRIDSIZE = 50
CUT_PAD = 0.20
BIN_WIDTH_S = 0.020
MODE_LABELS = np.array(["Fast (short)", "Medium", "Slow (long)"])
MODE_TO_IDX = {m: i for i, m in enumerate(MODE_LABELS)}
W_CTRL = np.array([0.157, 0.365, 0.478])
W_GLOBAL = np.array([0.06823987, 0.11403142, 0.16720762, 0.20454148, 0.11624734,
                     0.08214482, 0.04874001, 0.09543889, 0.07369576, 0.02971277], float)
DELTA_BLEND = 0.5
BETA_DUR = 1
BETA_PITCH = 3
EPS = 1e-12

# --- Utils ---
def _softmax(logits):
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def linear_to_lognormal(mu_sec, sigma_sec):
    mu_sec, sigma_sec = float(mu_sec), float(sigma_sec)
    if mu_sec <= 0 or sigma_sec <= 0:
        raise ValueError("mu_sec and sigma_sec must be positive.")
    s2 = np.log(1.0 + (sigma_sec**2)/(mu_sec**2))
    sd_log = np.sqrt(s2)
    mu_log = np.log(mu_sec) - 0.5 * s2
    return mu_log, sd_log

def attractor_probs_no_alpha(x_dur, mu_log, sd_log, w_ctrl=W_CTRL, beta=BETA_DUR):
    z = float(np.log(x_dur))
    r = -0.5 * ((z - mu_log) / sd_log) ** 2
    return _softmax(np.log(w_ctrl) + beta * r).ravel()

def _align_global_weights(mu, w_global=W_GLOBAL, eps=EPS):
    Kb = len(mu)
    w = np.array(w_global[:Kb], float) if Kb <= len(w_global) else np.ones(Kb)/Kb
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(Kb)/Kb
    w = np.clip(w, eps, None)
    return w / w.sum()

def _distance_kernel(d, tau=100.0, p=2.0, kind="cauchy", nu=1.0):
    d = np.asarray(d, float)
    if kind == "gaussian": return np.exp(-(d/tau)**p)
    if kind == "laplace":  return np.exp(-(d/tau))
    if kind == "cauchy":   return 1.0/(1.0 + (d/tau)**2)
    if kind == "rq":       return (1.0 + (d/tau)**2/(2.0*nu))**(-nu)
    raise ValueError(f"unknown kernel kind={kind}")

def attractor_probs_local(x_hz, mu, *, tau=400.0, p=2.0, kind="cauchy",
                          alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0, beta=BETA_PITCH):
    mu = np.asarray(mu, float)
    w_prior = _align_global_weights(mu)
    d = np.abs(float(x_hz) - mu)
    K = _distance_kernel(d, tau=tau, p=p, kind=kind, nu=rq_nu)
    K = np.clip(K, EPS, None) ** float(beta)
    s = np.log(np.clip(w_prior, EPS, None)) + np.log(np.clip(K, EPS, None))
    if nearest_boost: s[np.argmin(d)] += float(nearest_boost)
    loc = _softmax(s).ravel()
    if alpha_bg > 0:
        bg = w_prior / w_prior.sum()
        loc = (1.0 - alpha_bg) * loc + alpha_bg * bg
        loc = loc / loc.sum()
    return loc

def normal_cdf(x, mu, sd):
    return 0.5 * (1.0 + erf((x - mu) / (sd * np.sqrt(2.0))))

def normal_ppf(p, mu, sd):
    return mu + sd * np.sqrt(2.0) * erfinv(2.0 * p - 1.0)

# --- Control medians & KDE plot ---
all_birds = pd.read_pickle(DATA_PKL)
control = all_birds[all_birds.phase != 'playback'].copy()
rows = []
for bird in control.bird.unique():
    cb = control[control.bird == bird]
    for phase_id in cb.phase.unique():
        if phase_id == 'playback': continue
        cp = cb[cb.phase == phase_id]
        for set_id in cp.set.unique():
            cs = cp[cp.set == set_id]
            for snippet in cs.snippet_idx.unique():
                s = cs[cs.snippet_idx == snippet]
                n = len(s['duration'])
                rows.append({
                    'bird': bird, 'song': snippet,
                    'last_d': s['duration'].iloc[-1],
                    'first_d': s['duration'].iloc[0],
                    'd_median': s['duration'].median(),
                    'd_average': s['duration'].mean(),
                    'd_range': s['duration'].max() - s['duration'].min(),
                    'last_f': s['pitch_whistles'].iloc[-1],
                    'first_f': s['pitch_whistles'].iloc[0],
                    'f_median': s['pitch_whistles'].median(),
                    'f_average': s['pitch_whistles'].mean(),
                    'n_syl': n,
                    'gap_median': s['gap'].median(),
                    'int_median': s['interval'].median(),
                    'last_int': s['interval'].iloc[-2] if n > 1 else np.nan,
                    'first_int': s['interval'].iloc[0] if n > 0 else np.nan,
                    'last_gap': s['gap'].iloc[-2] if n > 1 else np.nan,
                    'first_gap': s['gap'].iloc[0] if n > 0 else np.nan,
                    'int_first_to_last': (s['interval'].iloc[0] - s['interval'].iloc[-2]) if n > 1 else np.nan,
                    'gap_first_to_last': (s['gap'].iloc[0] - s['gap'].iloc[-2]) if n > 1 else np.nan,
                    'd_first_to_last': s['duration'].iloc[0] - s['duration'].iloc[-1],
                    'f_first_to_last': s['pitch_whistles'].iloc[0] - s['pitch_whistles'].iloc[-1],
                })
whistle_songs_control = pd.DataFrame(rows)

plt.style.use('default')
cmap_w2g = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])
fig = plt.figure(figsize=(5, 5))
sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
            cmap=cmap_w2g, fill=True, bw_adjust=KDE_BW_ADJUST, levels=20, thresh=SUPPORT_THRESH, zorder=-1)
sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
            cmap="Reds_r", fill=False, bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH)
plt.xlabel('Median of whistle durations per song (s)')
plt.ylabel('Median frequency (Hz)')
plt.gca().set_box_aspect(1)
plt.tight_layout(); mpl.rcParams['pdf.fonttype'] = 42
plt.savefig(PLOT_OUT, transparent=True); plt.show()

# --- Duration→Pitch support map ---
class SupportByDuration:
    def __init__(self, d_median, f_median, bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
                 gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_S):
        d, f = np.asarray(d_median, float), np.asarray(f_median, float)
        m = np.isfinite(d) & np.isfinite(f); d, f = d[m], f[m]
        if d.size < 5: raise ValueError("Not enough points to build support.")
        x_min, x_max = d.min(), d.max(); y_min, y_max = f.min(), f.max()
        x_pad, y_pad = (x_max - x_min) * cut_pad, (y_max - y_min) * cut_pad
        self.xx = np.linspace(x_min - x_pad, x_max + x_pad, gridsize)
        self.yy = np.linspace(y_min - y_pad, y_max + y_pad, gridsize)
        X, Y = np.meshgrid(self.xx, self.yy)
        kde = gaussian_kde(np.vstack([d, f])); kde.covariance_factor = lambda: kde.scotts_factor() * bw_adjust; kde._compute_covariance()
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(gridsize, gridsize)
        self.level = float(Z.max()) * float(thresh)
        self.mask = (Z >= self.level)
        self.bin_width = float(bin_width)
        rows = []
        edges = np.arange(self.xx.min(), self.xx.max() + bin_width, bin_width)
        for left, right in zip(edges[:-1], edges[1:]):
            x_cols = np.where((self.xx >= left) & (self.xx < right))[0]
            if x_cols.size == 0: rows.append((left, right, [])); continue
            col_any = self.mask[:, x_cols].any(axis=1)
            rows.append((left, right, self._intervals_from_bool(col_any, self.yy)))
        self.rows = rows

    @staticmethod
    def _intervals_from_bool(v, y):
        idx = np.flatnonzero(v)
        if idx.size == 0: return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        return [(float(y[g[0]]), float(y[g[-1]])) for g in np.split(idx, splits)]

    def intervals_for_duration(self, x):
        x = float(x)
        for (l, r, ints) in self.rows:
            if l <= x < r: return ints
        return []

    def intervals_near_duration(self, x, max_neighbors=3):
        x = float(x)
        edges = np.array([r[0] for r in self.rows] + [self.rows[-1][1]])
        centers = 0.5 * (edges[:-1] + edges[1:])
        if centers.size == 0: return []
        idx = int(np.argmin(np.abs(centers - x)))
        for k in range(max_neighbors + 1):
            left, right = max(0, idx - k), min(len(self.rows) - 1, idx + k)
            cand = sum((self.rows[j][2] for j in range(left, right + 1)), [])
            if cand:
                cand = sorted(cand); merged = []
                for a, b in cand:
                    if not merged or a > merged[-1][1]: merged.append([a, b])
                    else: merged[-1][1] = max(merged[-1][1], b)
                return [(float(a), float(b)) for a, b in merged]
        return []

support = SupportByDuration(whistle_songs_control["d_median"], whistle_songs_control["f_median"],
                            KDE_BW_ADJUST, SUPPORT_THRESH, GRIDSIZE, CUT_PAD, BIN_WIDTH_S)

# --- Control 2D KDE helpers ---
class ControlDFKDE:
    def __init__(self, df_ctrl, bw_adjust=KDE_BW_ADJUST, gridsize=220, cut_pad=CUT_PAD):
        d = np.asarray(df_ctrl["d_median"], float); f = np.asarray(df_ctrl["f_median"], float)
        m = np.isfinite(d) & np.isfinite(f); d, f = d[m], f[m]
        x_min,x_max=d.min(),d.max(); y_min,y_max=f.min(),f.max()
        x_pad=(x_max-x_min)*cut_pad; y_pad=(y_max-y_min)*cut_pad
        self.d_grid=np.linspace(x_min-x_pad,x_max+x_pad,gridsize)
        self.f_grid=np.linspace(y_min-y_pad,y_max+y_pad,gridsize)
        X,Y=np.meshgrid(self.d_grid,self.f_grid,indexing="ij")
        kde=gaussian_kde(np.vstack([d,f])); kde.covariance_factor=lambda: kde.scotts_factor()*bw_adjust; kde._compute_covariance()
        Z=kde(np.vstack([X.ravel(),Y.ravel()])).reshape(gridsize,gridsize)
        self.Z=np.clip(Z,0,None); self.kde=kde
        self.dd=np.gradient(self.d_grid); self.df=np.gradient(self.f_grid)
        self.p_d=(self.Z*self.df[np.newaxis,:]).sum(axis=1); self.p_f=(self.Z*self.dd[:,np.newaxis]).sum(axis=0)
        self.p_d/= self.p_d.sum() if self.p_d.sum()>0 else 1.0
        self.p_f/= self.p_f.sum() if self.p_f.sum()>0 else 1.0

    def _eval_along_f(self, d_fixed, f_vals):
        pts=np.vstack([np.full_like(f_vals,float(d_fixed)), f_vals])
        return np.clip(self.kde(pts),0,None)

    def ctrl_comp_weights_given_d(self, d_fixed, mu, w_prior=None):
        vals=self._eval_along_f(d_fixed, np.asarray(mu,float))
        w_prior = _align_global_weights(mu) if w_prior is None else np.asarray(w_prior,float)
        comp=np.clip(vals,EPS,None)*w_prior
        return comp/comp.sum()

    def ctrl_comp_weights_marginal_f(self, mu, w_prior=None):
        mu=np.asarray(mu,float)
        pf=np.interp(mu,self.f_grid,self.p_f,left=EPS,right=EPS)
        w_prior = _align_global_weights(mu) if w_prior is None else np.asarray(w_prior,float)
        comp=np.clip(pf,EPS,None)*w_prior
        return comp/comp.sum()

    def ctrl_duration_mode_weights(self, mu_log):
        d_modes=np.exp(np.asarray(mu_log,float))
        pd=np.interp(d_modes,self.d_grid,self.p_d,left=EPS,right=EPS)
        w=np.clip(pd,EPS,None)
        return w/w.sum()

ctrl_kde = ControlDFKDE(whistle_songs_control)

# --- Load per-bird params ---
def load_bird_params_from_excel_duration(path, expected_modes=MODE_LABELS):
    df = pd.read_excel(path)
    req = {"sim_bird","mode","mu_sec","sigma_sec"}
    miss = req - set(df.columns)
    if miss: raise ValueError(f"Duration Excel missing columns: {miss}")
    df = df[df["mode"].isin(expected_modes)].copy()
    birds = df["sim_bird"].unique().tolist(); K = len(expected_modes)
    mu_log_birds = np.full((len(birds), K), np.nan); sd_log_birds = np.full((len(birds), K), np.nan)
    for i,b in enumerate(birds):
        sub=df[df["sim_bird"]==b]
        for _,r in sub.iterrows():
            k=MODE_TO_IDX[r["mode"]]; mu_log,sd_log=linear_to_lognormal(r["mu_sec"], r["sigma_sec"])
            mu_log_birds[i,k]=mu_log; sd_log_birds[i,k]=sd_log
        if np.isnan(mu_log_birds[i]).any() or np.isnan(sd_log_birds[i]).any():
            miss=[expected_modes[j] for j in np.where(np.isnan(mu_log_birds[i]))[0]]
            raise ValueError(f"Bird '{b}' missing duration modes: {miss}")
    return birds, mu_log_birds, sd_log_birds

def load_bird_params_from_excel_pitch(path):
    df = pd.read_excel(path); df.columns=[c.strip().lower() for c in df.columns]
    bird_col = "bird" if "bird" in df.columns else ("sim_bird" if "sim_bird" in df.columns else None)
    if bird_col is None: raise ValueError("Pitch Excel must have 'bird' or 'sim_bird'.")
    mu_cands=["mu_hz","mu (hz)","mu"]; sg_cands=["sigma_hz","sigma (hz)","sigma"]
    try: mu_col=next(c for c in mu_cands if c in df.columns); sg_col=next(c for c in sg_cands if c in df.columns)
    except StopIteration: raise ValueError("Pitch Excel must include μ and σ columns.")
    cols=[bird_col,mu_col,sg_col]+(["weight"] if "weight" in df.columns else [])
    df_use=df[cols].rename(columns={bird_col:"bird",mu_col:"mu_hz",sg_col:"sigma_hz"}).copy()
    df_use=df_use.replace([np.inf,-np.inf],np.nan).dropna(subset=["bird","mu_hz","sigma_hz"])
    df_use["sigma_hz"]=df_use["sigma_hz"].clip(lower=1e-6)
    bird_params={}
    for b,sub in df_use.groupby("bird",sort=False):
        sub=sub.sort_values("mu_hz")
        mu=sub["mu_hz"].to_numpy(float); sd=sub["sigma_hz"].to_numpy(float)
        if "weight" in sub.columns:
            w=sub["weight"].to_numpy(float)
            if (not np.all(np.isfinite(w))) or w.sum()<=0: w=np.ones_like(mu)
        else: w=np.ones_like(mu)
        bird_params[str(b)]=dict(mu=mu, sigma=sd, w=w/w.sum())
    birds=sorted(bird_params.keys())
    if not birds: raise ValueError("No birds found in pitch Excel after cleaning.")
    return bird_params, birds

birds_dur, mu_log_birds, sd_log_birds = load_bird_params_from_excel_duration(EXCEL_DURATION)
bird_params_pitch, birds_pitch = load_bird_params_from_excel_pitch(EXCEL_PITCH)

# --- Truncated mixture & joint sim (Duration→Pitch) ---
def _blend(p_eng, p_ctrl, delta=DELTA_BLEND):
    out = float(delta)*np.asarray(p_eng,float) + (1.0-float(delta))*np.asarray(p_ctrl,float)
    out = np.clip(out, EPS, None)
    return out/out.sum()

def sample_pitch_given_support_blended(playback_f_hz, params, support_intervals, rng,
                                       ctrl_comp_weights, tau=400.0, p=2.0, kind="cauchy",
                                       alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0,
                                       delta=DELTA_BLEND, beta_pitch=BETA_PITCH):
    mu, sd = params["mu"], params["sigma"]
    p_eng = attractor_probs_local(playback_f_hz, mu, tau=tau, p=p, kind=kind,
                                  alpha_bg=alpha_bg, nearest_boost=nearest_boost, rq_nu=rq_nu, beta=beta_pitch)
    p_base = _blend(p_eng, ctrl_comp_weights, delta=delta)

    def _mass_k(k):
        mass = 0.0; segs = []
        for a,b in support_intervals:
            Fa = normal_cdf(a, mu[k], sd[k]); Fb = normal_cdf(b, mu[k], sd[k])
            m = max(0.0, float(Fb - Fa)); mass += m; segs.append((Fa,Fb,m))
        return mass, segs

    masses, segs_all = [], []
    for k in range(len(mu)):
        m, segs = _mass_k(k); masses.append(m); segs_all.append(segs)
    w = p_base * np.clip(masses, 0.0, None); S = w.sum()
    if not np.isfinite(S) or S <= 1e-12:
        bounds = np.array([v for ab in support_intervals for v in ab], float)
        return float(bounds[np.argmin(np.abs(bounds - playback_f_hz))])
    pis = w / S; k = rng.choice(len(mu), p=pis)
    segs = segs_all[k]; mlist = np.array([m for _,_,m in segs], float); mlist /= mlist.sum() if mlist.sum()>0 else 1.0
    idx = rng.choice(len(segs), p=mlist); Fa,Fb,_ = segs[idx]; u = rng.random()
    return float(normal_ppf(Fa + u * max(EPS, (Fb - Fa)), mu[k], sd[k]))

def simulate_joint(playbacks, birds_dur, mu_log_b, sd_log_b, bird_params_pitch, birds_pitch, support: SupportByDuration,
                   n_renditions=50, tau_hz=400.0, kind_pitch="cauchy", p_exp=2.0, alpha_bg=0.05,
                   nearest_boost=0.5, rq_nu=1.0, seed=123):
    rng = np.random.default_rng(seed); rows = []; b2i = {b:i for i,b in enumerate(birds_dur)}
    for b in birds_pitch:
        if b not in b2i: warnings.warn(f"Bird '{b}' in pitch Excel not found in duration Excel. Skipping."); continue
        bi=b2i[b]; mu_log_k=mu_log_b[bi]; sd_log_k=sd_log_b[bi]; pitch_params=bird_params_pitch[b]
        ctrl_w_dur = ctrl_kde.ctrl_duration_mode_weights(mu_log_k)
        for (d_in, f_in) in playbacks:
            for r in range(n_renditions):
                p_d = _blend(attractor_probs_no_alpha(d_in, mu_log_k, sd_log_k, w_ctrl=W_CTRL, beta=BETA_DUR),
                             ctrl_w_dur, delta=DELTA_BLEND)
                k_d = rng.choice(len(mu_log_k), p=p_d)
                x_dur = float(np.exp(rng.normal(loc=mu_log_k[k_d], scale=sd_log_k[k_d])))
                intervals = support.intervals_for_duration(x_dur) or support.intervals_near_duration(x_dur, 3)
                if not intervals:
                    rows.append({"bird": b,"rendition": r,"playback_d_s": d_in,"playback_f_hz": f_in,
                                 "resp_d_s": x_dur,"resp_f_hz": np.nan,"dur_mode": int(k_d)})
                    continue
                mu = pitch_params["mu"]; w_prior = _align_global_weights(mu)
                ctrl_comp = ctrl_kde.ctrl_comp_weights_given_d(x_dur, mu, w_prior=w_prior)
                y_pitch = sample_pitch_given_support_blended(f_in, pitch_params, intervals, rng, ctrl_comp,
                                                             tau=tau_hz, p=p_exp, kind=kind_pitch,
                                                             alpha_bg=alpha_bg, nearest_boost=nearest_boost, rq_nu=rq_nu,
                                                             delta=DELTA_BLEND, beta_pitch=BETA_PITCH)
                rows.append({"bird": b,"rendition": r,"playback_d_s": d_in,"playback_f_hz": f_in,
                             "resp_d_s": x_dur,"resp_f_hz": y_pitch,"dur_mode": int(k_d)})
    return pd.DataFrame(rows)

# --- Exp. 2 config & simulation (Duration→Pitch) ---
playback_data_exp2 = {
    "A": ([0.14, 0.14, 0.14], [6, 7, 8]),
    "B": ([0.6, 0.7, 0.8],    [8, 7, 6]),
    "C": ([0.6, 0.7, 0.8],    [1, 2, 3]),
}
def build_playbacks(d_list, f_khz_list): return list(zip(d_list, [float(k)*1000 for k in f_khz_list]))
playbacks_by_region = {k: build_playbacks(d, f) for k,(d,f) in playback_data_exp2.items()}

N_RENDITIONS = 50; TAU_HZ = 100.0; KIND_PITCH = "cauchy"; P_EXP = 2.0
ALPHA_BG = 0.0; NEAREST_BOOST = 0.1; RQ_NU = 1.0; BASE_SEED = 123
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])
sim_by_region, seed = {}, BASE_SEED
for region, pb in playbacks_by_region.items():
    sim_by_region[region] = simulate_joint(pb, birds_dur, mu_log_birds, sd_log_birds,
                                           bird_params_pitch, birds_pitch, support,
                                           N_RENDITIONS, TAU_HZ, KIND_PITCH, P_EXP,
                                           ALPHA_BG, NEAREST_BOOST, RQ_NU, seed)
    sim_by_region[region]["region"] = region; seed += 101

# --- Exp. 2 plot (Duration→Pitch): KDE + ratio (2×3) ---
global_x_bins = np.linspace(0, 1, 100); global_y_bins = np.linspace(0, 10000, 100)
plt.style.use('default'); vmin, vmax = 0, 2; cmap = plt.cm.RdGy_r; norm = plt.Normalize(vmin=vmin, vmax=vmax)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
for i, region in enumerate(["A","B","C"]):
    sim = sim_by_region[region]; m = np.isfinite(sim["resp_d_s"]) & np.isfinite(sim["resp_f_hz"]); data = sim.loc[m].copy()
    pb = playbacks_by_region[region]; playback_d, playback_f = zip(*pb) if len(pb) else ([],[])
    ax_kde = axes[0, i]
    ax_kde.plot(playback_d, playback_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde)
    if len(data) >= 5:
        sns.kdeplot(x=data["resp_d_s"], y=data["resp_f_hz"], cmap=custom_cmap, fill=True,
                    bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1)
    ax_kde.set_title(f"Region {region}"); ax_kde.set_xlim([0, 0.9]); ax_kde.set_ylim([0, 9000]); ax_kde.set_box_aspect(1)
    ax_ratio = axes[1, i]
    x_exp, y_exp = data["resp_d_s"].to_numpy(), data["resp_f_hz"].to_numpy()
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3) if len(x_exp) >= 5 else None
    x_ctrl = whistle_songs_control.d_median.to_numpy(); y_ctrl = whistle_songs_control.f_median.to_numpy()
    vm = np.isfinite(x_ctrl) & np.isfinite(y_ctrl); kde_ctrl = gaussian_kde(np.vstack([x_ctrl[vm], y_ctrl[vm]]))
    x_grid = np.linspace(0, 0.9, 100); y_grid = np.linspace(0, 9000, 100); X, Y = np.meshgrid(x_grid, y_grid)
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z_exp = kde_exp(pos).reshape(X.shape) if kde_exp is not None else np.zeros_like(X)
    Z_ctrl = kde_ctrl(pos).reshape(X.shape)
    ratio = np.clip(Z_exp / (Z_ctrl + 3e-4), vmin, vmax)
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio)
    ax_ratio.plot(playback_d, playback_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
    ax_ratio.set_xlim([0, 0.9]); ax_ratio.set_ylim([0, 9000]); ax_ratio.set_box_aspect(1)
fig.subplots_adjust(right=0.85); cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
for ax in axes.flat: ax.set_xlabel(''); ax.set_ylabel('')
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')
fig.suptitle("Attractor (β duration=1 β pitch=3, δ=0.6)")
mpl.rcParams['pdf.fonttype'] = 42; os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_2_sim_duration_first_and_ratio.pdf', transparent=True); plt.show()

# --- Pitch→Duration support and path ---
BIN_WIDTH_HZ = 100.0
class SupportByPitch:
    def __init__(self, d_median, f_median, bw_adjust=KDE_BW_ADJUST, thresh=SUPPORT_THRESH,
                 gridsize=GRIDSIZE, cut_pad=CUT_PAD, bin_width=BIN_WIDTH_HZ):
        d,f=np.asarray(d_median,float),np.asarray(f_median,float)
        m=np.isfinite(d)&np.isfinite(f); d,f=d[m],f[m]
        if d.size<5: raise ValueError("Not enough points to build support (pitch-first).")
        x_min,x_max=d.min(),d.max(); y_min,y_max=f.min(),f.max()
        x_pad=(x_max-x_min)*cut_pad; y_pad=(y_max-y_min)*cut_pad
        self.dd=np.linspace(x_min-x_pad,x_max+x_pad,gridsize)
        self.ff=np.linspace(y_min-y_pad,y_max+y_pad,gridsize)
        D,F=np.meshgrid(self.dd,self.ff,indexing="ij")
        kde=gaussian_kde(np.vstack([d,f])); kde.covariance_factor=lambda: kde.scotts_factor()*bw_adjust; kde._compute_covariance()
        Z=kde(np.vstack([D.ravel(),F.ravel()])).reshape(gridsize,gridsize)
        self.level=float(Z.max())*float(thresh); self.mask=(Z>=self.level); self.bin_width=float(bin_width)
        rows=[]; edges=np.arange(self.ff.min(), self.ff.max()+bin_width, bin_width)
        for lo,hi in zip(edges[:-1], edges[1:]):
            y_rows=np.where((self.ff>=lo)&(self.ff<hi))[0]
            if y_rows.size==0: rows.append((lo,hi,[])); continue
            row_any=self.mask[:,y_rows].any(axis=1); rows.append((lo,hi,self._intervals_from_bool(row_any,self.dd)))
        self.rows=rows

    @staticmethod
    def _intervals_from_bool(v,x):
        idx=np.flatnonzero(v)
        if idx.size==0: return []
        splits=np.where(np.diff(idx)>1)[0]+1
        return [(float(x[g[0]]), float(x[g[-1]])) for g in np.split(idx,splits)]

    def intervals_for_pitch(self, y):
        y=float(y)
        for lo,hi,ints in self.rows:
            if lo<=y<hi: return ints
        return []

    def intervals_near_pitch(self, y, max_neighbors=3):
        y=float(y)
        edges=np.array([r[0] for r in self.rows]+[self.rows[-1][1]])
        centers=0.5*(edges[:-1]+edges[1:])
        if centers.size==0: return []
        idx=int(np.argmin(np.abs(centers-y)))
        for k in range(max_neighbors+1):
            left,right=max(0,idx-k),min(len(self.rows)-1,idx+k)
            cand=sum((self.rows[j][2] for j in range(left,right+1)),[])
            if cand:
                cand=sorted(cand); merged=[]
                for a,b in cand:
                    if not merged or a>merged[-1][1]: merged.append([a,b])
                    else: merged[-1][1]=max(merged[-1][1],b)
                return [(float(a),float(b)) for a,b in merged]
        return []

support_by_pitch = SupportByPitch(whistle_songs_control["d_median"], whistle_songs_control["f_median"],
                                  KDE_BW_ADJUST, SUPPORT_THRESH, GRIDSIZE, CUT_PAD, BIN_WIDTH_HZ)

def lognormal_cdf(x, mu_log, sd_log):
    x=float(x); 
    if x<=0: return 0.0
    return normal_cdf(np.log(x), mu_log, sd_log)

def lognormal_ppf(p, mu_log, sd_log):
    return float(np.exp(normal_ppf(p, mu_log, sd_log)))

def ctrl_duration_mode_weights_given_f(ctrl_kde, f_fixed, mu_log_vec):
    d_modes = np.exp(np.asarray(mu_log_vec, float))
    pts = np.vstack([d_modes, np.full_like(d_modes, float(f_fixed))])
    vals = np.clip(ctrl_kde.kde(pts), EPS, None); s = vals.sum()
    return vals / (s if s > 0 else 1.0)

def sample_duration_given_support_blended(playback_d_s, mu_log_vec, sd_log_vec, intervals, rng,
                                          ctrl_w_dur, delta=DELTA_BLEND):
    mu_log_vec, sd_log_vec = np.asarray(mu_log_vec,float), np.asarray(sd_log_vec,float)
    p_base = _blend(attractor_probs_no_alpha(playback_d_s, mu_log_vec, sd_log_vec, w_ctrl=W_CTRL, beta=BETA_DUR),
                    ctrl_w_dur, delta=delta)
    masses=[]; segs_all=[]
    for k in range(len(mu_log_vec)):
        mu_k,sd_k=mu_log_vec[k],sd_log_vec[k]; mass_k=0.0; segs_k=[]
        for a,b in intervals:
            Fa=lognormal_cdf(a,mu_k,sd_k); Fb=lognormal_cdf(b,mu_k,sd_k); m=max(0.0,float(Fb-Fa))
            mass_k+=m; segs_k.append((Fa,Fb,m))
        masses.append(mass_k); segs_all.append(segs_k)
    w = p_base * np.clip(masses,0.0,None); S=w.sum()
    if not np.isfinite(S) or S<=1e-12:
        bounds=np.array([v for ab in intervals for v in ab], float)
        return float(bounds[np.argmin(np.abs(bounds - playback_d_s))])
    pis=w/S; k=np.random.default_rng().choice(len(mu_log_vec), p=pis)
    segs=segs_all[k]; mlist=np.array([m for _,_,m in segs], float); mlist/= mlist.sum() if mlist.sum()>0 else 1.0
    idx=np.random.default_rng().choice(len(segs), p=mlist); Fa,Fb,_=segs[idx]; u=np.random.default_rng().random()
    return lognormal_ppf(Fa + u * max(EPS, (Fb - Fa)), mu_log_vec[k], sd_log_vec[k])

def simulate_joint_pitch_first(playbacks, birds_dur, mu_log_b, sd_log_b, bird_params_pitch, birds_pitch,
                               support_by_pitch: SupportByPitch, n_renditions=50, tau_hz=400.0, kind_pitch="cauchy",
                               p_exp=2.0, alpha_bg=0.05, nearest_boost=0.5, rq_nu=1.0, seed=777):
    rng=np.random.default_rng(seed); rows=[]; b2i={b:i for i,b in enumerate(birds_dur)}
    for b in birds_pitch:
        if b not in b2i: warnings.warn(f"Bird '{b}' in pitch Excel not found in duration Excel. Skipping."); continue
        bi=b2i[b]; mu_log_k, sd_log_k = mu_log_b[bi], sd_log_b[bi]
        pp=bird_params_pitch[b]; mu,sd=pp["mu"],pp["sigma"]; w_prior=_align_global_weights(mu)
        ctrl_comp_pitch = ctrl_kde.ctrl_comp_weights_marginal_f(mu, w_prior=w_prior)
        for d_in,f_in in playbacks:
            for r in range(n_renditions):
                p_pitch = _blend(attractor_probs_local(f_in, mu, tau=tau_hz, p=p_exp, kind=kind_pitch,
                                                       alpha_bg=alpha_bg, nearest_boost=nearest_boost, rq_nu=rq_nu, beta=BETA_PITCH),
                                 ctrl_comp_pitch, delta=DELTA_BLEND)
                k_p = rng.choice(len(mu), p=p_pitch)
                y_pitch = float(rng.normal(loc=mu[k_p], scale=sd[k_p]))
                intervals = support_by_pitch.intervals_for_pitch(y_pitch) or support_by_pitch.intervals_near_pitch(y_pitch, 3)
                if not intervals:
                    rows.append({"bird": b,"rendition": r,"playback_d_s": d_in,"playback_f_hz": f_in,
                                 "resp_d_s": np.nan,"resp_f_hz": y_pitch,"dur_mode": np.nan})
                    continue
                ctrl_w_dur_f = ctrl_duration_mode_weights_given_f(ctrl_kde, y_pitch, mu_log_k)
                x_dur = sample_duration_given_support_blended(d_in, mu_log_k, sd_log_k, intervals, rng, ctrl_w_dur_f, delta=DELTA_BLEND)
                r_scores = -0.5 * ((np.log(x_dur) - mu_log_k) / sd_log_k) ** 2 + np.log(W_CTRL)
                k_d = int(np.argmax(r_scores))
                rows.append({"bird": b,"rendition": r,"playback_d_s": d_in,"playback_f_hz": f_in,
                             "resp_d_s": x_dur,"resp_f_hz": y_pitch,"dur_mode": k_d})
    return pd.DataFrame(rows)

# --- Exp. 2 plot (Pitch→Duration) ---
sim_by_region_pf, seed_pf = {}, BASE_SEED + 202
for region, pb in playbacks_by_region.items():
    sim_by_region_pf[region] = simulate_joint_pitch_first(pb, birds_dur, mu_log_birds, sd_log_birds,
                                                          bird_params_pitch, birds_pitch, support_by_pitch,
                                                          N_RENDITIONS, TAU_HZ, KIND_PITCH, P_EXP,
                                                          ALPHA_BG, NEAREST_BOOST, RQ_NU, seed_pf)
    sim_by_region_pf[region]["region"] = region; seed_pf += 131

plt.style.use('default'); vmin, vmax = 0, 2; cmap = plt.cm.RdGy_r; norm = plt.Normalize(vmin=vmin, vmax=vmax)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
for i, region in enumerate(["A","B","C"]):
    sim = sim_by_region_pf[region]; m = np.isfinite(sim["resp_d_s"]) & np.isfinite(sim["resp_f_hz"]); data = sim.loc[m].copy()
    pb = playbacks_by_region[region]; playback_d, playback_f = zip(*pb) if len(pb) else ([],[])
    ax_kde = axes[0, i]
    ax_kde.plot(playback_d, playback_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde)
    if len(data) >= 5:
        sns.kdeplot(x=data["resp_d_s"], y=data["resp_f_hz"], cmap=cmap_w2g, fill=True,
                    bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1)
    ax_kde.set_title(f"Region {region}"); ax_kde.set_xlim([0, 0.9]); ax_kde.set_ylim([0, 9000]); ax_kde.set_box_aspect(1)
    ax_ratio = axes[1, i]
    x_exp, y_exp = data["resp_d_s"].to_numpy(), data["resp_f_hz"].to_numpy()
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3) if len(x_exp) >= 5 else None
    x_ctrl = whistle_songs_control.d_median.to_numpy(); y_ctrl = whistle_songs_control.f_median.to_numpy()
    vm = np.isfinite(x_ctrl) & np.isfinite(y_ctrl); kde_ctrl = gaussian_kde(np.vstack([x_ctrl[vm], y_ctrl[vm]]))
    x_grid = np.linspace(0, 0.9, 100); y_grid = np.linspace(0, 9000, 100); X, Y = np.meshgrid(x_grid, y_grid)
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z_exp = kde_exp(pos).reshape(X.shape) if kde_exp is not None else np.zeros_like(X)
    Z_ctrl = kde_ctrl(pos).reshape(X.shape)
    ratio = np.clip(Z_exp / (Z_ctrl + 3e-4), vmin, vmax)
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio)
    ax_ratio.plot(playback_d, playback_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
    ax_ratio.set_xlim([0, 0.9]); ax_ratio.set_ylim([0, 9000]); ax_ratio.set_box_aspect(1)
fig.subplots_adjust(right=0.85); cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
for ax in axes.flat: ax.set_xlabel(''); ax.set_ylabel('')
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')
fig.suptitle("Attractor (β duration=1 β pitch=3, δ=0.6)")
mpl.rcParams['pdf.fonttype'] = 42; os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_2_sim_pitch_first_and_ratio.pdf', transparent=True); plt.show()

# --- Sweep: d=0.44 s, pitch sweep (Duration→Pitch) ---
pitches_hz = [1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,4000,4500,5000,6000,7000,8000]
playbacks_sweep = [(0.44, float(p)) for p in pitches_hz]
sim_sweep = simulate_joint(playbacks_sweep, birds_dur, mu_log_birds, sd_log_birds,
                           bird_params_pitch, birds_pitch, support,
                           N_RENDITIONS, TAU_HZ, KIND_PITCH, P_EXP,
                           ALPHA_BG, NEAREST_BOOST, RQ_NU, BASE_SEED + 999)

plt.style.use('default'); vmin, vmax = 0, 2; cmap = plt.cm.RdGy_r; norm = plt.Normalize(vmin=vmin, vmax=vmax)
fig, (ax_kde, ax_ratio) = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True)
m = np.isfinite(sim_sweep["resp_d_s"]) & np.isfinite(sim_sweep["resp_f_hz"]); data = sim_sweep.loc[m].copy()
pb_d, pb_f = zip(*playbacks_sweep)
ax_kde.plot(pb_d, pb_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
            cmap="Reds_r", fill=False, bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH, ax=ax_kde)
if len(data) >= 5:
    sns.kdeplot(x=data["resp_d_s"], y=data["resp_f_hz"],
                cmap=LinearSegmentedColormap.from_list("white_to_olivedrab", ["white","olivedrab"]),
                fill=True, bw_adjust=KDE_BW_ADJUST, levels=10, thresh=SUPPORT_THRESH, ax=ax_kde, zorder=-1)
ax_kde.set_title("Sweep (d = 0.44 s)"); ax_kde.set_xlim([0, 0.9]); ax_kde.set_ylim([0, 9000]); ax_kde.set_box_aspect(1)

x_exp, y_exp = data["resp_d_s"].to_numpy(), data["resp_f_hz"].to_numpy()
kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3) if len(x_exp) >= 5 else None
x_ctrl = whistle_songs_control.d_median.to_numpy(); y_ctrl = whistle_songs_control.f_median.to_numpy()
vm = np.isfinite(x_ctrl) & np.isfinite(y_ctrl); kde_ctrl = gaussian_kde(np.vstack([x_ctrl[vm], y_ctrl[vm]]))
x_grid = np.linspace(0, 0.9, 100); y_grid = np.linspace(0, 9000, 100); X, Y = np.meshgrid(x_grid, y_grid)
pos = np.vstack([X.ravel(), Y.ravel()])
Z_exp = kde_exp(pos).reshape(X.shape) if kde_exp is not None else np.zeros_like(X)
Z_ctrl = kde_ctrl(pos).reshape(X.shape)
ratio = np.clip(Z_exp / (Z_ctrl + 3e-4), vmin, vmax)
ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)
sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
            cmap="Reds_r", fill=False, bw_adjust=KDE_BW_ADJUST, levels=1, thresh=SUPPORT_THRESH, ax=ax_ratio)
ax_ratio.plot(pb_d, pb_f, color="#12B568", marker='+', linestyle='None', markersize=6, markeredgewidth=3)
ax_ratio.set_xlim([0, 0.9]); ax_ratio.set_ylim([0, 9000]); ax_ratio.set_box_aspect(1)
fig.subplots_adjust(right=0.85); cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
for ax in (ax_kde, ax_ratio): ax.set_xlabel(''); ax.set_ylabel('')
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')
fig.suptitle(f"Sweep (d = 0.44 s) — Attractor (β_dur={BETA_DUR} β_pitch={BETA_PITCH}, δ={DELTA_BLEND})")
mpl.rcParams['pdf.fonttype'] = 42; os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_sweep_d044_duration_first_and_ratio.pdf', transparent=True); plt.show()

# --- Sweep duration-only views ---
playbacks_sweep_pf = [(0.44, float(p)) for p in pitches_hz]
sim_sweep_pf = simulate_joint_pitch_first(playbacks_sweep_pf, birds_dur, mu_log_birds, sd_log_birds,
                                          bird_params_pitch, birds_pitch, support_by_pitch,
                                          N_RENDITIONS, TAU_HZ, KIND_PITCH, P_EXP,
                                          ALPHA_BG, NEAREST_BOOST, RQ_NU, BASE_SEED + 5252)
dur_pf = sim_sweep_pf.loc[np.isfinite(sim_sweep_pf["resp_d_s"]), "resp_d_s"].to_numpy()
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 2), sharex=True, gridspec_kw={"height_ratios":[3,1]})
xg = np.linspace(0, 0.9, 400)
kde_pf = gaussian_kde(dur_pf); kde_pf.covariance_factor = lambda: kde_pf.scotts_factor()*0.8; kde_pf._compute_covariance()
yprob_pf = kde_pf(xg) * BIN_WIDTH_S
ax_top.fill_between(xg, 0, yprob_pf, alpha=1, facecolor='mediumaquamarine'); ax_top.plot(xg, yprob_pf, color='k', lw=2)
ax_top.axvline(0.44, ls='--', color='k', lw=1.2)
ax_top.set_xlim(0, 0.9); ax_top.set_ylim(bottom=0); ax_top.set_ylabel('Probability')
ax_top.set_title(f'Simulation pitch paper (β_dur={BETA_DUR}, β_pitch={BETA_PITCH}, δ={DELTA_BLEND})')
rng = np.random.default_rng(12); yj = rng.uniform(0.03, 0.19, size=len(dur_pf))
ax_bot.scatter(dur_pf, yj, s=7, c='mediumaquamarine', alpha=0.1, rasterized=True)
ax_bot.set_ylim(0, 0.22); ax_bot.set_yticks([]); ax_bot.set_xlabel('Whistle duration (s)')
plt.setp(ax_top.get_xticklabels(), visible=False)
plt.tight_layout(); mpl.rcParams['pdf.fonttype'] = 42; os.makedirs("Plots", exist_ok=True)
plt.savefig('Plots/Exp_sweep_d044_duration_only_pitchfirst.pdf', transparent=True, dpi=300, bbox_inches="tight"); plt.show()

# --- Counts summary ---
def count_finite(df): return int((np.isfinite(df["resp_d_s"]) & np.isfinite(df["resp_f_hz"])).sum())
common_birds = sorted(set(birds_dur) & set(birds_pitch)); n_birds = len(common_birds)
print("\n==========================\nSIMULATION COUNTS SUMMARY\n==========================")
print(f"Birds in duration Excel: {len(birds_dur)}")
print(f"Birds in pitch Excel:    {len(birds_pitch)}")
print(f"Common birds simulated:  {n_birds}\n--------------------------")
print("Exp. 2 — Duration→Pitch")
tot_rows=tot_finite=0
for region in ["A","B","C"]:
    df=sim_by_region[region]; rows=len(df); finite=count_finite(df)
    tot_rows+=rows; tot_finite+=finite; nominal=3*N_RENDITIONS*n_birds
    print(f"  Region {region}: rows={rows:,} (nominal={nominal:,}), finite(d,f)={finite:,}")
print(f"  TOTAL (A+B+C): rows={tot_rows:,}, finite(d,f)={tot_finite:,}\n--------------------------")
print("Exp. 2 — Pitch→Duration")
tot_rows_pf=tot_finite_pf=0
for region in ["A","B","C"]:
    df=sim_by_region_pf[region]; rows=len(df); finite=count_finite(df)
    tot_rows_pf+=rows; tot_finite_pf+=finite; nominal=3*N_RENDITIONS*n_birds
    print(f"  Region {region}: rows={rows:,} (nominal={nominal:,}), finite(d,f)={finite:,}")
print(f"  TOTAL (A+B+C): rows={tot_rows_pf:,}, finite(d,f)={tot_finite_pf:,}\n--------------------------")
L_sweep=16
rows_sweep=len(sim_sweep); finite_sweep=count_finite(sim_sweep); nominal_sweep=L_sweep*N_RENDITIONS*n_birds
print("Sweep (d = 0.44 s) — Duration→Pitch")
print(f"  rows={rows_sweep:,} (nominal={nominal_sweep:,}), finite(d,f)={finite_sweep:,}\n--------------------------")
rows_sweep_pf=len(sim_sweep_pf); finite_sweep_pf=count_finite(sim_sweep_pf); nominal_sweep_pf=L_sweep*N_RENDITIONS*n_birds
print("Sweep (d = 0.44 s) — Pitch→Duration")
print(f"  rows={rows_sweep_pf:,} (nominal={nominal_sweep_pf:,}), finite(d,f)={finite_sweep_pf:,}")
print("==========================\n")
print(f"[Pitch→Duration] N = {dur_pf.size}, mean = {np.nanmean(dur_pf):.3f} s, median = {np.nanmedian(dur_pf):.3f} s")
