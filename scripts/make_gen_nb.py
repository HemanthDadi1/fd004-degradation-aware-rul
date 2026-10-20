"""Generate the IID (07) and DS-B (08) symmetric rerun notebooks: same training rule as the DS-A rerun (06)."""
import json, sys

COND = sys.argv[1]          # "IID", "DS-A" or "DS-B"
WIN = int(sys.argv[2])
OUTFILE = sys.argv[3]
DATASET = sys.argv[4] if len(sys.argv) > 4 else "FD002"
ARCH = sys.argv[5] if len(sys.argv) > 5 else "bilstm_attn"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})


slug = ("" if DATASET == "FD004" else f"{DATASET}_") + {"IID": "IID", "DS-A": "DS_A", "DS-B": "DS_B"}[COND] + f"_W{WIN}" + ("" if ARCH == "bilstm_attn" else f"_{ARCH}")
title = {"IID": "standard (IID) evaluation", "DS-A": "DS-A held-out Regime 0", "DS-B": "DS-B restricted operating envelope"}[COND]
md(f"""
# {DATASET}, {WIN}-cycle context, architecture {ARCH}: {title}, five seeds, converged training

Part of the Paper 1/Paper 2 study. Same pipeline, pre-specified training rule and calibration as notebooks 06–11; only the dataset, window length or encoder architecture differ, as named in the title. The DS-A rerun (`06_FD004_DS_A_Rerun_Converged_FiveSeed.ipynb`) trains under a pre-specified,
training-only convergence rule. This notebook applies **the same rule** to the {COND} protocol so that IID, DS-A and DS-B are
reported on one training protocol, five seeds each (42, 123, 456, 789, 1000), seed 42 as the single-run reference.

Unchanged from the frozen {COND} pipeline: data split, preprocessing, windows, architecture (20→18 projection, BiLSTM(64),
temporal attention, q10/q50/q90 pinball loss), Adam 1e-3, batch 256, and every calibration rule.

Changed (Part A, written before any training): 20% of training engines (split seed 42) become internal validation; up to
150 epochs; early stopping on internal-validation pinball loss, patience 15, not before epoch 30, best weights restored;
converged if best validation loss ≤ 0.60 × constant-quantile loss. Every seed is reported; none is replaced.

Part B trains and saves all models without touching evaluation windows; Part C evaluates afterwards.
Set `FD004_SMOKE=1` for a 1-seed, 3-epoch pipeline check.
""")

code(r"""
import os, sys, json, time, random, hashlib, platform
from pathlib import Path

SMOKE_TEST = os.environ.get("FD004_SMOKE") == "1"

def _find_root():
    for c in [os.environ.get("FD004_ROOT", ""), "/content/drive/MyDrive/FD004_Research", str(Path.home() / "Downloads/FD004_Research")]:
        if c and Path(c, "CMAPSSData/train_FD004.txt").exists():
            return Path(c)
    raise FileNotFoundError("Set FD004_ROOT to the FD004_Research folder.")

if "google.colab" in sys.modules and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

ROOT = _find_root()
""" + f'SLUG = "{slug}"\nCONDITION = "{COND}"\nW = {WIN}\nDATASET = "{DATASET}"\nARCH = "{ARCH}"\n' + r"""
OUT = Path(os.environ.get("FD004_OUT") or ROOT / (f"FD004_{SLUG}_Results" + ("_SMOKE" if SMOKE_TEST else "")))
(OUT / "models").mkdir(parents=True, exist_ok=True)
(OUT / "histories").mkdir(exist_ok=True)
print("Root:", ROOT, "| Output:", OUT)
""")

md("## Part A: protocol (written before any training or evaluation)")
code(r"""
PROTOCOL = {
    "experiment": f"{DATASET} replication of Paper 1, {CONDITION}, {W}-cycle context, converged five-seed run",
    "dataset": DATASET,
    "architecture": ARCH,
    "window_size": W,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "data_seed": 42,
    "model_seeds": [42] if SMOKE_TEST else [42, 123, 456, 789, 1000],
    "internal_validation": {"fraction_of_train_engines": 0.20, "split_seed": 42},
    "model": {"projection": 18, "bilstm_units": 64, "quantiles": [0.10, 0.50, 0.90], "learning_rate": 1e-3, "batch_size": 256},
    "training_rule": {"max_epochs": 3 if SMOKE_TEST else 150, "monitor": "val_loss", "patience": 15,
                      "start_from_epoch": 0 if SMOKE_TEST else 30, "restore_best_weights": True},
    "convergence_check": {"ratio": 0.60, "reference": "constant training quantiles evaluated on internal validation",
                          "failed_seeds": "reported and flagged; never replaced or dropped"},
    "calibration": {"alpha": 0.20, "methods": ["Raw Quantiles", "Window CQR", "Asymmetric CQR",
                                               "Regime-conditional Asymmetric CQR", "Balanced Engine CQR"],
                    "window": "finite-sample higher quantile at ceil((n+1)(1-alpha))/n",
                    "asymmetric": "per tail at ceil((n+1)(1-alpha/2))/n, unclipped scores",
                    "regime_conditional": "asymmetric rule within each endpoint regime",
                    "balanced": "within-engine np.quantile(.,0.80), then ceil((M+1)(1-alpha))/M higher",
                    "interval_clip_at_zero": True},
    "life_stage": {"Early": "RUL > 80", "Mid": "40 < RUL <= 80", "Late": "RUL <= 40"},
    "bootstrap": {"replicates": 10000, "unit": "evaluation engine", "rng_seed": 42},
}
protocol_path = OUT / "protocol.json"
if protocol_path.exists() and not SMOKE_TEST:
    old = json.loads(protocol_path.read_text())
    changed = {k for k in PROTOCOL if k != "created_utc" and old.get(k) != PROTOCOL[k]}
    assert not changed, f"Protocol already registered and differs in {changed}."
    PROTOCOL["created_utc"] = old["created_utc"]
else:
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2))
print("Protocol sha256:", hashlib.sha256(protocol_path.read_bytes()).hexdigest()[:16])
""")

md("## Data and windows (identical to the frozen pipeline)")
code(r"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
P = PROTOCOL
SET = ["setting_1", "setting_2", "setting_3"]
SENS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
cols = ["unit_id", "time_cycles"] + SET + [f"s_{i}" for i in range(1, 22)]
raw = pd.read_csv(ROOT / f"CMAPSSData/train_{DATASET}.txt", sep=r"\s+", header=None).iloc[:, :len(cols)]
raw.columns = cols
raw["RUL"] = (raw.groupby("unit_id")["time_cycles"].transform("max") - raw["time_cycles"]).clip(upper=125)
setting_scaler = StandardScaler().fit(raw[SET])
km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(setting_scaler.transform(raw[SET]))
raw["regime"] = km.labels_
# Held-out regime = the cluster at operating point (42, 0.84, 100), the condition held out as Regime 0 in FD004.
centers = setting_scaler.inverse_transform(km.cluster_centers_)
HO = int(np.argmin(np.abs(centers - np.array([42.0, 0.84, 100.0])).sum(axis=1)))
print("Regime centres:", np.round(centers, 2).tolist(), "| held-out regime:", HO)
engines = np.array(sorted(raw.unit_id.unique()))
train_engines, tmp = train_test_split(engines, test_size=0.30, random_state=42)
cal_engines, test_engines = train_test_split(tmp, test_size=0.50, random_state=42)

mask_tr = raw.unit_id.isin(train_engines)
if DATASET == "FD004" and CONDITION == "IID":
    # Paper 1 IID protocol: RandomState(42) permutation, first 20% (49) validation engines; k-means and scalers fitted on
    # training engines only; validation engines indexed 1..49 by unit id, then 25 calibration / 24 evaluation with seed 42.
    perm = np.random.RandomState(42).permutation(raw.unit_id.unique())
    val_units, train_engines = np.sort(perm[:49]), np.sort(perm[49:])  # sorted = the index order notebook 07 split on
    mask_tr = raw.unit_id.isin(train_engines)
    setting_scaler = StandardScaler().fit(raw.loc[mask_tr, SET])
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(setting_scaler.transform(raw.loc[mask_tr, SET]))
    raw["regime"] = km.transform(setting_scaler.transform(raw[SET])).argmin(axis=1)
    raw.loc[mask_tr, "regime"] = km.labels_
    centers = setting_scaler.inverse_transform(km.cluster_centers_)
    np.random.seed(42)
    cal_idx = np.random.choice(np.arange(1, 50), size=25, replace=False)
    cal_engines, test_engines = val_units[cal_idx - 1], np.setdiff1d(val_units, val_units[cal_idx - 1])
    s_ = np.load(ROOT / "models/b6_variants/b6_cal_eval_split_ids.npz")
    assert set(s_["cal_engine_ids"]) == set(cal_idx)
    unit_to_index = {u: i + 1 for i, u in enumerate(val_units)}
if CONDITION == "DS-B":
    stats = raw[mask_tr].groupby("regime")[SET].agg(["mean", "std"])
    def envelope_score(d):
        out = np.full(len(d), np.nan)
        for r in sorted(d.regime.unique()):
            m = (d.regime == r).to_numpy()
            mu = np.array([stats.loc[r, (c, "mean")] for c in SET]); sd = np.array([stats.loc[r, (c, "std")] for c in SET])
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            out[m] = np.sqrt(np.mean(((d.loc[m, SET].to_numpy(float) - mu) / sd) ** 2, axis=1))
        return out
    raw["score"] = envelope_score(raw)
    thr = raw[mask_tr].groupby("regime")["score"].quantile(0.80).to_dict()
    raw["inside"] = raw.score <= raw.regime.map(thr)
    pd.Series(thr, name="envelope_threshold").rename_axis("regime").to_csv(OUT / "envelope_thresholds.csv")
    scaler_rows = mask_tr & raw.inside
elif CONDITION == "DS-A":
    scaler_rows = mask_tr & (raw.regime != HO)
else:
    scaler_rows = mask_tr
traj = raw.sort_values(["unit_id", "time_cycles"]).reset_index(drop=True)
traj[SENS] = StandardScaler().fit(raw.loc[scaler_rows, SENS]).transform(traj[SENS])

def eligible(g, mode):
    if mode == "all":
        return np.ones(len(g), bool)
    if mode in ("seen", "heldout"):
        return (g.regime.to_numpy() != HO) if mode == "seen" else (g.regime.to_numpy() == HO)
    return g.inside.to_numpy() if mode == "inside" else ~g.inside.to_numpy()

def make_windows(ids, mode):
    X, y, eid, cyc, reg = [], [], [], [], []
    eye = np.eye(6, dtype=np.float32)
    for e in sorted(ids):
        g = traj[traj.unit_id == e]
        sens, rg, rul, tc, ok = g[SENS].to_numpy(np.float32), g.regime.to_numpy(), g.RUL.to_numpy(np.float32), g.time_cycles.to_numpy(), eligible(g, mode)
        for end in range(W - 1, len(g)):
            if not ok[end]:
                continue
            sl = slice(end - W + 1, end + 1)
            X.append(np.concatenate([sens[sl], eye[rg[sl]]], -1)); y.append(rul[end]); eid.append(e); cyc.append(tc[end]); reg.append(rg[end])
    return np.asarray(X), np.asarray(y, np.float32), np.asarray(eid), np.asarray(cyc), np.asarray(reg)

train_mode, test_mode = {"IID": ("all", "all"), "DS-A": ("seen", "heldout"), "DS-B": ("inside", "outside")}[CONDITION]
fit_engines, ival_engines = train_test_split(train_engines, test_size=P["internal_validation"]["fraction_of_train_engines"],
                                             random_state=P["internal_validation"]["split_seed"])
X_fit, y_fit, _, _, _ = make_windows(fit_engines, train_mode)
X_iv, y_iv, _, _, _ = make_windows(ival_engines, train_mode)
X_cal, y_cal, id_cal, _, reg_cal = make_windows(cal_engines, train_mode)
X_test, y_test, id_test, cyc_test, reg_test = make_windows(test_engines, test_mode)
if DATASET == "FD004" and CONDITION == "IID":
    id_cal = np.array([unit_to_index[u] for u in id_cal]); id_test = np.array([unit_to_index[u] for u in id_test])
    # Integrity: rebuilt windows must equal the archived tensors at 30 and 60 cycles.
    ref = {30: ROOT / "artifacts/FD004_baseline_tensors.npz", 60: ROOT / "artifacts/P2_E8_60cycle_tensors.npz"}.get(W)
    if ref is not None:
        t_ = np.load(ref); Xv = np.concatenate([X_cal, X_test]); idv = np.concatenate([id_cal, id_test]); o = np.argsort(idv, kind="stable")
        Xr = np.concatenate([Xv[o][:, :, :14], np.argmax(Xv[o][:, :, 14:], axis=-1)[..., None].astype(np.float32)], -1)
        assert np.allclose(Xr, t_["X_val"], atol=1e-5), "rebuilt IID windows differ from the archived tensors"
        print("IID windows match the archived", W, "cycle tensors")
assert set(id_cal).isdisjoint(id_test) and len(np.unique(id_test)) > 0
print(f"Engines: train {len(train_engines)} (fit {len(fit_engines)} + ival {len(ival_engines)}), cal {len(cal_engines)}, test {len(test_engines)}")
print("fit", X_fit.shape, "| ival", X_iv.shape, "| cal", X_cal.shape, "| test", X_test.shape, "| test engines", len(np.unique(id_test)))
pd.concat([pd.DataFrame({"split": n, "unit_id": v}) for n, v in
           [("fit", fit_engines), ("internal_val", ival_engines), ("calibration", cal_engines), ("test", test_engines)]]
          ).to_csv(OUT / "engine_split.csv", index=False)
(OUT / "held_out_regime.json").write_text(json.dumps({"held_out_regime": HO, "centres": np.round(centers, 3).tolist()}))
""")

md("## Part B: train five seeds and check convergence (no evaluation data)")
code(r"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Layer
from tensorflow.keras.models import Model
Q = P["model"]["quantiles"]

@tf.keras.utils.register_keras_serializable(package="fd004")
class FastTemporalAttention(Layer):
    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(d, 1), initializer="glorot_uniform")
        super().build(input_shape)
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        return tf.reduce_sum(x * tf.nn.softmax(tf.tensordot(e, self.v, axes=1), axis=1), axis=1)

@tf.keras.utils.register_keras_serializable(package="fd004")
def multi_quantile_pinball_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
    return tf.reduce_mean(tf.add_n([tf.maximum(q * (y_true - y_pred[:, i:i+1]), (q - 1.0) * (y_true - y_pred[:, i:i+1]))
                                    for i, q in enumerate(Q)]))

def build_model():
    inp = Input(shape=(W, 20), name="input_window")
    h = Dense(18, use_bias=False, name="projection")(inp)
    if ARCH == "bilstm_attn":
        h = Bidirectional(LSTM(64, return_sequences=True), name="bilstm")(h)
        h = FastTemporalAttention(name="temporal_attention")(h)
    elif ARCH == "lstm_attn":
        h = LSTM(64, return_sequences=True, name="lstm")(h)
        h = FastTemporalAttention(name="temporal_attention")(h)
    elif ARCH == "bilstm_noattn":
        h = Bidirectional(LSTM(64), name="bilstm")(h)
    elif ARCH == "lstm_noattn":
        h = LSTM(64, name="lstm")(h)
    else:
        raise ValueError(ARCH)
    m = Model(inp, Dense(3, name="quantiles")(h))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=multi_quantile_pinball_loss)
    return m

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s); os.environ["PYTHONHASHSEED"] = str(s)

const_iv = float(sum(np.mean(np.maximum(q * (y_iv - np.quantile(y_fit, q)), (q - 1) * (y_iv - np.quantile(y_fit, q)))) for q in Q))
print(f"Constant-quantile loss on internal validation: {const_iv:.3f}")
(OUT / "environment.json").write_text(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
    "tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]}, indent=2))

tr = P["training_rule"]; train_log = []
for seed in P["model_seeds"]:
    t0 = time.time()
    mpath, hpath = OUT / "models" / f"{SLUG}_rerun_seed{seed}.keras", OUT / "histories" / f"seed{seed}_history.csv"
    if mpath.exists() and hpath.exists() and not SMOKE_TEST:
        # Resume: this seed finished in an earlier, interrupted run (model is saved after its history).
        h = pd.read_csv(hpath); print(f"seed {seed}: resumed from saved model")
    else:
        tf.keras.backend.clear_session(); set_seed(seed); model = build_model()
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=tr["patience"], start_from_epoch=tr["start_from_epoch"],
                                                restore_best_weights=True)
        hist = model.fit(X_fit, y_fit, validation_data=(X_iv, y_iv), epochs=tr["max_epochs"], batch_size=256, shuffle=True,
                         callbacks=[stop], verbose=2)
        h = pd.DataFrame(hist.history); h.insert(0, "epoch", np.arange(1, len(h) + 1))
        h.to_csv(hpath, index=False)
        model.save(mpath)
    # Keras start_from_epoch ignores earlier epochs, so the restored weights are the best from epoch start+1 on.
    hr = h[h.epoch > tr["start_from_epoch"]] if (h.epoch > tr["start_from_epoch"]).any() else h
    best = float(hr.val_loss.min())
    train_log.append({"seed": seed, "epochs_run": len(h), "best_epoch": int(hr.loc[hr.val_loss.idxmin(), "epoch"]),
                      "best_val_loss": best, "constant_val_loss": const_iv, "ratio": best / const_iv,
                      "converged": best <= P["convergence_check"]["ratio"] * const_iv, "minutes": (time.time() - t0) / 60})
    print(train_log[-1])
train_log = pd.DataFrame(train_log); train_log.to_csv(OUT / "convergence_check.csv", index=False)
print(train_log.round(3).to_string(index=False))
""")

md("## Part C: evaluation (run only after Part B is saved)")
code(r"""
assert (OUT / "convergence_check.csv").exists()
from scipy.stats import pearsonr, spearmanr
A = P["calibration"]["alpha"]

def cq(scores, a):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher"))

def stage_of(r):
    return np.where(r > 80, "Early", np.where(r > 40, "Mid", "Late"))

rows, glob_rows, eng_rows, stage_rows, corr_rows, corrections = [], [], [], [], [], []
for seed in P["model_seeds"]:
    model = tf.keras.models.load_model(OUT / "models" / f"{SLUG}_rerun_seed{seed}.keras")
    pc = model.predict(X_cal, batch_size=512, verbose=0); pt = model.predict(X_test, batch_size=512, verbose=0)
    lc, uc, lt, ut = pc[:, 0], pc[:, 2], pt[:, 0], pt[:, 2]
    qw = cq(np.maximum(lc - y_cal, y_cal - uc), A)
    ql, qu = cq(lc - y_cal, A / 2), cq(y_cal - uc, A / 2)
    err = np.maximum(lc - y_cal, y_cal - uc)
    eq = np.array([np.quantile(err[id_cal == e], 0.80) for e in np.unique(id_cal)]); M = len(eq)
    qb = float(np.quantile(eq, min(1.0, np.ceil((M + 1) * (1 - A)) / M), method="higher"))
    rl, ru = np.zeros_like(lt), np.zeros_like(ut)
    for r in (range(6) if CONDITION != "DS-A" else []):  # not applicable to DS-A: Regime 0 has no calibration windows
        mcal, mt = reg_cal == r, reg_test == r
        a1, a2 = cq(lc[mcal] - y_cal[mcal], A / 2), cq(y_cal[mcal] - uc[mcal], A / 2)
        rl[mt], ru[mt] = lt[mt] - a1, ut[mt] + a2
        corrections.append({"seed": seed, "method": "Regime-conditional", "regime": r, "q_lower": a1, "q_upper": a2})
    corrections += [{"seed": seed, "method": "Window", "q": qw}, {"seed": seed, "method": "Asymmetric", "q_lower": ql, "q_upper": qu},
                    {"seed": seed, "method": "Balanced", "q": qb}]
    methods = {"Raw Quantiles": (lt, ut), "Window CQR": (lt - qw, ut + qw), "Asymmetric CQR": (lt - ql, ut + qu),
               "Regime-conditional Asym. CQR": (rl, ru), "Balanced Engine CQR": (lt - qb, ut + qb)}
    if CONDITION == "DS-A":
        methods.pop("Regime-conditional Asym. CQR")
    df = pd.DataFrame({"seed": seed, "engine": id_test, "cycle": cyc_test, "endpoint_regime": reg_test, "true_RUL": y_test,
                       "q10": lt, "q50": pt[:, 1], "q90": ut, "stage": stage_of(y_test)})
    df["error_q50_minus_true"] = df.q50 - df.true_RUL
    for name, (lo, hi) in methods.items():
        lo, hi = np.maximum(0, lo), np.maximum(0, hi)
        k = name.split()[0].lower().replace("-", "_")
        df[f"{k}_lower"], df[f"{k}_upper"] = lo, hi
        cov = (y_test >= lo) & (y_test <= hi)
        wink = (hi - lo) + (2 / A) * (lo - y_test) * (y_test < lo) + (2 / A) * (y_test - hi) * (y_test > hi)
        ec = pd.Series(cov).groupby(id_test).mean() * 100
        glob_rows.append({"seed": seed, "method": name, "RMSE": float(np.sqrt(np.mean(df.error_q50_minus_true ** 2))),
                          "MAE": float(df.error_q50_minus_true.abs().mean()), "PICP": 100 * cov.mean(), "MPIW": float((hi - lo).mean()),
                          "Winkler": float(wink.mean()), "lower_violation": 100 * (y_test < lo).mean(), "upper_violation": 100 * (y_test > hi).mean(),
                          "eng_mean": ec.mean(), "eng_median": ec.median(), "eng_p10": np.percentile(ec, 10), "eng_p25": np.percentile(ec, 25),
                          "eng_worst": ec.min(), "below80": int((ec < 80).sum()), "below70": int((ec < 70).sum())})
        df[f"{k}_covered"] = cov
    for e, g in df.groupby("engine"):
        eng_rows.append({"seed": seed, "engine": e, "n": len(g), "RMSE": float(np.sqrt(np.mean(g.error_q50_minus_true ** 2))),
                         "MAE": float(g.error_q50_minus_true.abs().mean()), "bias": float(g.error_q50_minus_true.mean()),
                         "asym_cov": 100 * g.asymmetric_covered.mean(), "bal_cov": 100 * g.balanced_covered.mean(),
                         "bal_below_lower": 100 * (g.true_RUL < g.balanced_lower).mean(), "bal_above_upper": 100 * (g.true_RUL > g.balanced_upper).mean(),
                         "mid_bal_cov": 100 * g.balanced_covered[g.stage == "Mid"].mean() if (g.stage == "Mid").any() else np.nan,
                         "dominant_regime": int(g.endpoint_regime.mode().iloc[0]), "dominant_share": float((g.endpoint_regime == g.endpoint_regime.mode().iloc[0]).mean())})
    for s_, g in df.groupby("stage"):
        stage_rows.append({"seed": seed, "stage": s_, "N": len(g), "RMSE": float(np.sqrt(np.mean(g.error_q50_minus_true ** 2))),
                           "signed_bias": float(g.error_q50_minus_true.mean()), "asym_cov": 100 * g.asymmetric_covered.mean(),
                           "bal_cov": 100 * g.balanced_covered.mean()})
    pe = pd.DataFrame([r for r in eng_rows if r["seed"] == seed])
    corr_rows.append({"seed": seed, "pearson_r": pearsonr(pe.RMSE, pe.bal_cov)[0], "pearson_p": pearsonr(pe.RMSE, pe.bal_cov)[1],
                      "spearman_rho": spearmanr(pe.RMSE, pe.bal_cov)[0],
                      "problem_engines": " ".join(map(str, sorted(pe.engine[pe.bal_cov < 80])))})
    rows.append(df)

pred = pd.concat(rows, ignore_index=True); pred.to_csv(OUT / "test_predictions_rowlevel.csv", index=False)
glob = pd.DataFrame(glob_rows).merge(train_log[["seed", "converged"]], on="seed"); glob.to_csv(OUT / "global_by_seed.csv", index=False)
per_engine = pd.DataFrame(eng_rows); per_engine.to_csv(OUT / "per_engine_by_seed.csv", index=False)
stages = pd.DataFrame(stage_rows); stages.to_csv(OUT / "life_stage_by_seed.csv", index=False)
pd.DataFrame(corr_rows).to_csv(OUT / "rmse_coverage_correlation_by_seed.csv", index=False)
pd.DataFrame(corrections).to_csv(OUT / "corrections_by_seed.csv", index=False)
print(glob[glob.seed == P["model_seeds"][0]].round(2).to_string(index=False))
""")

code(r"""
# Engine-cluster bootstrap and five-seed summary
boot = []
for seed in P["model_seeds"]:
    g = pred[pred.seed == seed]
    agg = g.groupby("engine").agg(n=("true_RUL", "size"), ca=("asymmetric_covered", "sum"), cb=("balanced_covered", "sum"))
    agg["wa"] = (g.asymmetric_upper - g.asymmetric_lower).groupby(g.engine).sum()
    agg["wb"] = (g.balanced_upper - g.balanced_lower).groupby(g.engine).sum()
    n = agg.n.to_numpy(); rng = np.random.default_rng(42); idx = rng.integers(0, len(n), size=(10000, len(n))); N = n[idx].sum(1)
    covA, covB = 100 * agg.ca.to_numpy() / n, 100 * agg.cb.to_numpy() / n
    S = {"asym_picp": (100 * agg.ca.sum() / n.sum(), 100 * agg.ca.to_numpy()[idx].sum(1) / N),
         "bal_picp": (100 * agg.cb.sum() / n.sum(), 100 * agg.cb.to_numpy()[idx].sum(1) / N),
         "asym_mpiw": (agg.wa.sum() / n.sum(), agg.wa.to_numpy()[idx].sum(1) / N),
         "bal_mpiw": (agg.wb.sum() / n.sum(), agg.wb.to_numpy()[idx].sum(1) / N),
         "asym_eng_mean": (covA.mean(), covA[idx].mean(1)), "bal_eng_mean": (covB.mean(), covB[idx].mean(1)),
         "asym_eng_p10": (np.percentile(covA, 10), np.percentile(covA[idx], 10, axis=1)),
         "bal_eng_p10": (np.percentile(covB, 10), np.percentile(covB[idx], 10, axis=1)),
         "asym_eng_worst": (covA.min(), covA[idx].min(1)), "bal_eng_worst": (covB.min(), covB[idx].min(1))}
    S["bal_minus_asym_picp"] = (S["bal_picp"][0] - S["asym_picp"][0], S["bal_picp"][1] - S["asym_picp"][1])
    for k, (pt_, v) in S.items():
        boot.append({"seed": seed, "stat": k, "point": pt_, "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5)})
pd.DataFrame(boot).to_csv(OUT / "bootstrap_by_seed.csv", index=False)

five = glob.groupby("method")[["PICP", "MPIW", "Winkler", "RMSE", "MAE", "below80", "eng_worst"]].agg(["mean", "std"])
five.columns = [f"{a}_{b}" for a, b in five.columns]; five.to_csv(OUT / "five_seed_summary.csv")
stages.groupby("stage")[["RMSE", "signed_bias", "bal_cov", "asym_cov"]].agg(["mean", "std"]).to_csv(OUT / "life_stage_five_seed_summary.csv")
print(five.round(2).to_string())
manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob("*")) if p.is_file()}
(OUT / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2)); print("Done:", len(manifest), "files")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open(OUTFILE, "w"), indent=1)
print("wrote", OUTFILE)
