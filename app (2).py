

from pathlib import Path
import ast

# Find the original Streamlit application
app = Path("/content/app.py")

if not app.exists():
    raise FileNotFoundError(
        "Please run the %%writefile app.py cell first."
    )

# Verify Python syntax
ast.parse(app.read_text(encoding="utf-8"))

# Create the dependency file
requirements = """streamlit
numpy
pandas
scipy
matplotlib
openpyxl
"""

Path("/content/requirements.txt").write_text(
    requirements,
    encoding="utf-8"
)

print("Ready for deployment!")

# Download the two required files
files.download("/content/requirements.txt")

"""
Distribution Fitting Dashboard (Streamlit)
Run:  pip install streamlit pandas numpy scipy matplotlib openpyxl
      streamlit run app.py
"""
import io
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="DistriFit Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container{padding-top:4rem;max-width:1300px}
div[data-testid="stMetric"]{background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.25);
  border-radius:12px;padding:12px 16px}
.hero{padding:30px 34px 26px;border-radius:18px;background:linear-gradient(135deg,#4f46e5 0%,#2563eb 50%,#06b6d4 100%);
  margin-bottom:20px;box-shadow:0 8px 24px rgba(79,70,229,.30)}
.chips span{display:inline-block;background:rgba(255,255,255,.20);border-radius:999px;padding:4px 13px;
  margin:14px 8px 0 0;font-size:.82rem;color:#fff}
.hero h1{margin:0;font-size:2.4rem;color:#fff;padding:0;line-height:1.2}
.hero p{margin:8px 0 0;color:#fff;opacity:.95;font-size:1.05rem}
.verdict{border-left:5px solid #22c55e;background:rgba(34,197,94,.12);padding:12px 16px;border-radius:8px;margin:10px 0 14px}
.step{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:16px;min-height:120px}
.step b{font-size:1.05rem}
button[data-baseweb="tab"]{font-size:1rem;font-weight:600}
footer{visibility:hidden}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600}
.plan{border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:16px;min-height:210px}
.plan.best{border:2px solid #4f46e5;background:rgba(79,70,229,.08)}
.plan h4{margin:0}.plan .price{font-size:1.5rem;font-weight:700;margin:4px 0 8px}
.plan ul{padding-left:18px;margin:0;font-size:.92rem}
@media (max-width:640px){.hero h1{font-size:1.6rem}.hero{padding:20px 18px}
  .block-container{padding:3.5rem .7rem .8rem}div[data-testid="stMetric"]{padding:8px 10px}}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linestyle": "--"})

# Keep widget values when a widget is not rendered (e.g. when switching App mode)
for _k in list(st.session_state.keys()):
    if str(_k).startswith("p_"):
        st.session_state[_k] = st.session_state[_k]


def init(key, **kw):
    """Pass a widget's default only on first creation (avoids Session State warnings)."""
    return {} if key in st.session_state else kw


CONTINUOUS = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
}
DISCRETE = {
    "Poisson": stats.poisson,
    "Geometric": stats.geom,
    "Binomial": stats.binom,
    "Negative Binomial": stats.nbinom,
}


# ----------------------------------------------------------------------
# DATA INPUT
# ----------------------------------------------------------------------
@st.cache_data
def read_file(name, content):

    if not name.lower().endswith(".csv"):
        raise ValueError(
            "Unsupported format. Only CSV files are accepted."
        )

    return pd.read_csv(io.BytesIO(content))

def parse_pasted(text):
    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    vals, bad = [], []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            bad.append(t)
    return np.array(vals), bad


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def period_filter(series, key, box):
    """Sidebar dropdowns to pick a time window (year/month if the labels allow it)."""
    periods = [str(p) for p in series.index]
    if len(periods) < 2:
        return series
    parsed = []
    for p in periods:
        m = re.match(r"^\s*(\d{4})\s*[-_/ ]?\s*([A-Za-z]{3})?", p)
        if not m:
            parsed = None
            break
        mon = m.group(2).title() if m.group(2) else None
        parsed.append((int(m.group(1)), MONTHS.index(mon) if mon in MONTHS else None))

    box.markdown("**Time period**")
    c1, c2 = box.columns(2)
    if parsed is None:  # unknown label format: plain From / To dropdowns
        a = c1.selectbox("From", periods, key=f"{key}_a")
        b = c2.selectbox("To", periods, key=f"{key}_b", **init(f"{key}_b", index=len(periods) - 1))
        i, j = periods.index(a), periods.index(b)
        keep = [i <= n <= j for n in range(len(periods))]
    else:
        years = sorted({y for y, _ in parsed})
        monthly = all(mo is not None for _, mo in parsed)
        y1 = c1.selectbox("From year", years, key=f"{key}_y1")
        y2 = c2.selectbox("To year", years, key=f"{key}_y2", **init(f"{key}_y2", index=len(years) - 1))
        lo, hi = (y1, 0), (y2, 0)
        if monthly:
            m1 = c1.selectbox("From month", MONTHS, key=f"{key}_m1", **init(f"{key}_m1", index=parsed[0][1]))
            m2 = c2.selectbox("To month", MONTHS, key=f"{key}_m2", **init(f"{key}_m2", index=parsed[-1][1]))
            lo, hi = (y1, MONTHS.index(m1)), (y2, MONTHS.index(m2))
        keep = [lo <= (y, mo if monthly else 0) <= hi for y, mo in parsed]
    if not any(keep):
        box.error("The start of the period is after its end.")
        return series.iloc[0:0]
    box.caption(f"{sum(keep)} of {len(periods)} periods selected")
    return series.iloc[np.flatnonzero(keep)]


def _set_state(key, val):
    st.session_state[key] = val


def pick_many(box, label, options, key, is_pro, fmt=str):
    """Multi-select with Select all / Clear buttons (Free plan: one at a time)."""
    sel = box.multiselect(label, options, format_func=fmt, key=key,
                          max_selections=None if is_pro else 1,
                          help=None if is_pro else "Free plan: one series at a time. Pro lets you pick many and compare them.",
                          **init(key, default=[options[0]]))
    if is_pro and len(options) > 1:
        c1, c2 = box.columns(2)
        c1.button("✔ Select all", key=f"all_{key}", on_click=_set_state, args=(key, list(options)))
        c2.button("✖ Clear", key=f"clr_{key}", on_click=_set_state, args=(key, []))
    return sel


def file_dataset(fname, fbytes, box, is_pro):
    """Widgets (inside `box`) to choose one or more series from a file. Returns a list of dicts."""
    raw = read_file(fname, fbytes)
    first = raw.columns[0]
    has_labels = pd.to_numeric(raw[first], errors="coerce").notna().mean() < 0.5
    wide = has_labels and (len(raw.columns) - 1) > len(raw)
    lk = f"p_layout_{fname}"
    layout = box.radio(
        "Data layout", ["Series in rows (periods across columns)", "Series in columns"],
        key=lk, **init(lk, index=0 if wide else 1),
        help="Auto-selected from your file. Use the first option if each ROW is a data series "
             "(e.g. 'Total', 'Land & Buildings') and the column headers are years/months.")

    if layout.startswith("Series in rows"):
        labels = raw[first].astype(str).str.strip().tolist()
        sel = pick_many(box, "Data series (tick one or more rows)", list(range(len(raw))),
                        f"p_rows_{fname}_{is_pro}", is_pro, fmt=lambda i: labels[i])
        periods = [str(c) for c in (raw.columns[1:] if has_labels else raw.columns)]
        order = np.arange(len(periods))
        try:  # chronological order if the file is newest-first
            if int(periods[0][:4]) > int(periods[-1][:4]):
                order = order[::-1]
        except ValueError:
            pass
        pos = pd.Series(np.arange(len(order)), index=[periods[i] for i in order])
        cols = order[period_filter(pos, f"p_per_{fname}", box).to_numpy()]  # same window for every series
        if not sel:
            box.info("Tick at least one series.")
            return []
        span = f"{periods[cols[0]]} to {periods[cols[-1]]}" if len(cols) else "no periods"
        out = []
        for r in sel:
            rowvals = pd.to_numeric(raw.iloc[r, 1:] if has_labels else raw.iloc[r], errors="coerce").to_numpy(dtype=float)
            out.append(dict(values=rowvals[cols], label=f"{fname} · {labels[r]} ({span})", preview=raw, short=labels[r]))
        return out

    num = pd.DataFrame({str(c): pd.to_numeric(raw[c], errors="coerce") for c in raw.columns})
    num = num.loc[:, num.notna().mean() >= 0.5]
    if num.empty:
        box.error("No numeric columns found in this file.")
        return []
    if is_pro and len(num.columns) > 1 and box.checkbox("Combine all numeric columns into one sample",
                                                         key=f"p_combine_{fname}"):
        return [dict(values=num.to_numpy(dtype=float).ravel(), label=f"{fname} (all numeric columns)",
                     preview=raw, short=Path(fname).stem)]
    sel = pick_many(box, "Columns to analyse (tick one or more)", list(num.columns), f"p_cols_{fname}_{is_pro}", is_pro)
    if not sel:
        box.info("Tick at least one column.")
    return [dict(values=num[c].to_numpy(dtype=float), label=f"{fname} · {c}", preview=raw, short=str(c)) for c in sel]


# ============================================================
# DATA UPLOADER WITH CSV VALIDATION AND NOTIFICATIONS
# ============================================================

def data_uploader():

    st.sidebar.header("1 · Data input")

    st.sidebar.info(
        "📄 **Accepted format: CSV (.csv) only.**\n\n"
        "Please convert Excel or other file formats "
        "to CSV before uploading."
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV files",
        type=None,
        accept_multiple_files=True,
        key="upload",
        help="Only CSV files can be processed."
    )

    if not uploaded_files:
        return []

    valid_files = []

    for uploaded_file in uploaded_files:

        if not uploaded_file.name.lower().endswith(".csv"):

            # Persistent error message
            st.sidebar.error(
                f"❌ Unsupported file: {uploaded_file.name}\n\n"
                "Only CSV (.csv) files are accepted. "
                "Please convert your file to CSV and try again."
            )

            # Temporary pop-up notification
            st.toast(
                f"Unsupported file: {uploaded_file.name}. "
                "CSV files only.",
                icon="⚠️"
            )

            continue

        valid_files.append(uploaded_file)

    if len(valid_files) < len(uploaded_files):
        st.sidebar.warning(
            "Some uploaded files were rejected because "
            "they are not in CSV format."
        )

    return valid_files


def get_datasets(up, is_pro):
    """Build the list of datasets from uploaded files (single source of truth), pasted values and example data."""
    out = []
    files = {f.name: f.getvalue() for f in up or []}
    names = list(files)
    if not is_pro and len(names) > 1:
        st.sidebar.warning("Free plan analyses one dataset at a time. Upgrade to compare several.")
        names = names[:1]
    for i, nm in enumerate(names):
        box = st.sidebar.expander(f"📄 {nm}", expanded=i < 2)
        out += [{**d, "name": nm} for d in file_dataset(nm, files[nm], box, is_pro)]
    if len(names) > 1:  # keep tab names unique across files
        for d in out:
            d["short"] = f"{Path(d['name']).stem[:14]} · {d['short']}"

    with st.sidebar.expander("✍ Paste values"):
        text = st.text_area("Numbers separated by commas, spaces or new lines", height=130,
                            placeholder="12.5, 14.1, 9.8, 11.2 ...", key="p_paste")
    if text.strip():
        vals, bad = parse_pasted(text)
        if bad:
            st.sidebar.warning(f"Ignored {len(bad)} non-numeric pasted entries: {bad[:5]}")
        out.append(dict(name="Pasted values", short="Pasted values", values=vals, label="Pasted values",
                        preview=pd.DataFrame({"value": vals})))

    if st.sidebar.checkbox("Use example data", key="p_use_example"):
        kind = st.sidebar.selectbox("Example", ["Continuous (lognormal)", "Count data (Poisson)"], key="p_example")
        rng = np.random.default_rng(42)
        vals = rng.lognormal(3, 0.5, 200) if kind.startswith("Cont") else rng.poisson(6, 200).astype(float)
        out.append(dict(name="Example data", short="Example data", values=vals, label=f"Example: {kind}",
                        preview=pd.DataFrame({"value": vals})))

    if not is_pro and len(out) > 1:
        st.sidebar.warning("Free plan analyses one dataset at a time. Upgrade to compare several.")
        out = out[:1]
    return out


# ----------------------------------------------------------------------
# FITTING
# ----------------------------------------------------------------------
def chi_square(x, dist, params, k):
    """Chi-square GoF for discrete data; bins merged until expected >= 5."""
    lo, hi = int(x.min()), int(x.max())
    if hi - lo > 5000:
        return np.nan, np.nan
    vals = np.arange(lo, hi + 1)
    obs = np.array([(x == v).sum() for v in vals], float)
    exp = len(x) * dist.pmf(vals, *params)
    exp[0] += len(x) * dist.cdf(lo - 1, *params)
    exp[-1] += len(x) * dist.sf(hi, *params)
    O, E, co, ce = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        co, ce = co + o, ce + e
        if ce >= 5:
            O.append(co); E.append(ce); co = ce = 0.0
    if ce > 0:
        if E:
            O[-1] += co; E[-1] += ce
        else:
            O.append(co); E.append(ce)
    O, E = np.array(O), np.array(E)
    dof = len(O) - 1 - k
    if dof < 1:
        return np.nan, np.nan
    stat = ((O - E) ** 2 / E).sum()
    return stat, stats.chi2.sf(stat, dof)


# ============================================================
# CONTINUOUS DISTRIBUTION FITTING
# ============================================================

def fit_continuous(x, names):
    results, skipped = {}, []

    # Convert input into numerical values
    x = np.asarray(x, dtype=float)

    # Validate input data
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError(
            "Continuous data must contain valid finite numbers."
        )

    positive = np.min(x) > 0

    # Fit each selected distribution
    for n in names:
        d = CONTINUOUS[n]

        try:
            # Normal Distribution
            if n == "Normal":
                p = d.fit(x)
                k = 2

            # Positive-only distributions
            elif not positive:
                skipped.append(
                    f"{n} (requires all values > 0)"
                )
                continue

            # Exponential, Gamma, Lognormal, Weibull
            else:
                p = d.fit(x, floc=0)
                k = 1 if n == "Exponential" else 2

            # Calculate log-likelihood
            ll = float(np.sum(d.logpdf(x, *p)))

            if not np.isfinite(ll):
                skipped.append(
                    f"{n} (invalid log-likelihood)"
                )
                continue

            # Kolmogorov-Smirnov goodness-of-fit test
            ks = stats.kstest(
                x,
                d.cdf,
                args=p
            )

            # Store fitted results
            results[n] = {
                "dist": d,
                "params": p,
                "k": k,
                "ll": ll,
                "test": "Kolmogorov-Smirnov",
                "stat": float(ks.statistic),
                "pval": float(ks.pvalue)
            }

        except Exception as e:
            skipped.append(
                f"{n} (fit failed: {e})"
            )

    return results, skipped


def fit_discrete(x, names, binomial_trials=None):
    results, skipped = {}, []

    x = np.asarray(x, dtype=float)

    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Discrete data must contain finite numbers.")

    if not np.all(x >= 0) or not np.all(x == np.floor(x)):
        raise ValueError(
            "Discrete data must contain non-negative integers."
        )

    x = x.astype(int)
    m = x.mean()
    v = x.var(ddof=1) if len(x) > 1 else 0

    for n in names:
        d = DISCRETE[n]

        try:
            if n == "Poisson":
                p, k = (m,), 1

            elif n == "Geometric":
                if x.min() >= 1:
                    p, k = (1 / m, 0), 1
                else:
                    p, k = (1 / (m + 1), -1), 1

            elif n == "Binomial":
                if binomial_trials is None:
                    skipped.append(
                        "Binomial (fixed number of trials required)"
                    )
                    continue

                trials = int(binomial_trials)

                if trials < 1 or x.max() > trials:
                    skipped.append("Binomial (invalid trial count)")
                    continue

                phat = m / trials
                p, k = (trials, phat), 1

            elif n == "Negative Binomial":
                if v <= m or m <= 0:
                    skipped.append(
                        "Negative Binomial (requires variance > mean)"
                    )
                    continue

                pr = m / v
                r = m * m / (v - m)
                p, k = (r, pr), 2

            else:
                continue

            ll = np.sum(d.logpmf(x, *p))

            if not np.isfinite(ll):
                skipped.append(f"{n} (invalid likelihood)")
                continue

            stat, pval = chi_square(x, d, p, k)

            results[n] = {
                "dist": d,
                "params": p,
                "k": k,
                "ll": ll,
                "test": "Chi-square",
                "stat": stat,
                "pval": pval
            }

        except Exception as e:
            skipped.append(f"{n} (fit failed: {e})")

    return results, skipped



def rank_table(results, n):
    rows = []
    for name, r in results.items():
        rows.append({"Distribution": name, "Parameters (k)": r["k"], "Log-Likelihood": r["ll"],
                     "AIC": 2 * r["k"] - 2 * r["ll"], "BIC": r["k"] * np.log(n) - 2 * r["ll"],
                     "GoF test": r["test"], "GoF statistic": r["stat"], "p-value": r["pval"]})
    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    df["ΔAIC"] = df["AIC"] - df["AIC"].min()
    df["BIC Rank"] = df["BIC"].rank().astype(int)
    wts = np.exp(-df["ΔAIC"] / 2)
    df["Akaike weight"] = wts / wts.sum()
    return df


def verdict(table):
    w = table.iloc[0]
    txt = f"<b>{w['Distribution']}</b> has the lowest AIC. "
    if len(table) > 1:
        d, r = table.loc[1, "ΔAIC"], table.loc[1, "Distribution"]
        if d < 2:
            txt += f"It is essentially tied with <b>{r}</b> (ΔAIC = {d:.1f}), so either is defensible. "
        elif d < 10:
            txt += f"It beats <b>{r}</b> by ΔAIC = {d:.1f} (moderate evidence). "
        else:
            txt += f"It clearly beats the runner-up <b>{r}</b> (ΔAIC = {d:.1f}). "
    p = w["p-value"]
    if pd.notna(p):
        txt += ("The goodness-of-fit test finds no significant misfit" if p > 0.05
                else "The goodness-of-fit test flags a possible misfit") + f" (p = {p:.3f})."
    return txt


def build_report(label, n, discrete, table, winner, params, mom):
    return "\n".join([
        "# Distribution Fitting Report", "",
        f"- Source: {label}", f"- Observations: {n}",
        f"- Data type: {'Discrete' if discrete else 'Continuous'}",
        f"- Best fit (lowest AIC): **{winner}**", "",
        "## Ranking", "```", table.round(4).to_string(index=False), "```", "",
        "## Winner parameters", *[f"- {k}: {v:.6g}" for k, v in params.items()], "",
        "## Moments: raw vs fitted", "```", mom.round(4).to_string(index=False), "```",
    ])


def named_params(name, p):
    return {
        "Normal": lambda: {"μ (mean)": p[0], "σ (std dev)": p[1]},
        "Exponential": lambda: {"scale (1/λ)": p[1], "λ (rate)": 1 / p[1]},
        "Gamma": lambda: {"shape (α)": p[0], "scale (θ)": p[2]},
        "Lognormal": lambda: {"σ (shape)": p[0], "μ (mean of ln x)": np.log(p[2])},
        "Weibull": lambda: {"shape (k)": p[0], "scale (λ)": p[2]},
        "Poisson": lambda: {"λ (mean rate)": p[0]},
        "Geometric": lambda: {"p (success prob.)": p[0]},
        "Binomial": lambda: {
       "n (fixed trials)": p[0],
        "p (estimated success probability)": p[1]
},
        "Negative Binomial": lambda: {"r (successes)": p[0], "p (success prob.)": p[1]},
    }[name]()


def moments_table(x, r):
    mean, var, skew, kurt = r["dist"].stats(*r["params"], moments="mvsk")
    raw = [x.mean(), x.var(ddof=1), stats.skew(x), stats.kurtosis(x, fisher=False)]
    fit = [float(mean), float(var), float(skew), float(kurt) + 3]
    df = pd.DataFrame({"Moment": ["Mean", "Variance", "Skewness", "Kurtosis (Pearson)"],
                       "Raw data": raw, "Fitted distribution": fit})
    df["Difference"] = df["Fitted distribution"] - df["Raw data"]
    return df


# ----------------------------------------------------------------------
# DASHBOARD PLOTS
# ----------------------------------------------------------------------
def dashboard(x, results, winner, discrete, show_all):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    w = results[winner]
    grid = (np.arange(int(x.min()), int(x.max()) + 1) if discrete
            else np.linspace(x.min(), x.max(), 400))
    curve = (lambda r, g: r["dist"].pmf(g, *r["params"])) if discrete else \
            (lambda r, g: r["dist"].pdf(g, *r["params"]))

    # 1. empirical vs fitted density
    a = ax[0, 0]
    if discrete:
        vals, counts = np.unique(x.astype(int), return_counts=True)
        a.bar(vals, counts / len(x), color="lightgray", edgecolor="black", label="Empirical")
    else:
        a.hist(x, bins="auto", density=True, color="lightgray", edgecolor="black", label="Empirical")
    if show_all:
        for n, r in results.items():
            if n != winner:
                a.plot(grid, curve(r, grid), alpha=0.5, lw=1.2, label=n)
    a.plot(grid, curve(w, grid), "r-", lw=3, marker="o" if discrete else None, label=f"Best: {winner}")
    a.set(title="Empirical vs fitted " + ("PMF" if discrete else "PDF"), ylabel="Probability" if discrete else "Density")
    a.legend()

    # 2. CDF
    a = ax[0, 1]
    a.step(xs, ecdf, where="post", color="black", label="Empirical CDF")
    if show_all:
        for n, r in results.items():
            if n != winner:
                a.plot(grid, r["dist"].cdf(grid, *r["params"]), alpha=0.5, lw=1.2, label=n)
    a.plot(grid, w["dist"].cdf(grid, *w["params"]), "r-", lw=2.5, label=f"Best: {winner}")
    a.set(title="CDF comparison", ylabel="Cumulative probability")
    a.legend()

    # 3. Q-Q plot
    a = ax[1, 0]
    q = w["dist"].ppf((np.arange(1, len(xs) + 1) - 0.5) / len(xs), *w["params"])
    a.scatter(q, xs, s=14, alpha=0.7)
    lim = [min(q.min(), xs.min()), max(q.max(), xs.max())]
    a.plot(lim, lim, "r--")
    a.set(title=f"Q-Q plot ({winner})", xlabel="Theoretical quantiles", ylabel="Observed quantiles")

    # 4. P-P plot
    a = ax[1, 1]
    a.scatter(w["dist"].cdf(xs, *w["params"]), ecdf, s=14, alpha=0.7)
    a.plot([0, 1], [0, 1], "r--")
    a.set(title=f"P-P plot ({winner})", xlabel="Theoretical CDF", ylabel="Empirical CDF")

    for a in ax.ravel():
        a.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig



# ============================================================
# BUILD COMPLETE REPORT AS CSV
# ============================================================

def build_csv_report(label, n, discrete, table, winner, params, mom):

    rows = []

    # Report summary
    rows.extend([
        {"Section": "Summary", "Item": "Source", "Value": label},
        {"Section": "Summary", "Item": "Observations", "Value": n},
        {"Section": "Summary", "Item": "Data type",
         "Value": "Discrete" if discrete else "Continuous"},
        {"Section": "Summary", "Item": "Best fit", "Value": winner}
    ])

    # Estimated parameters
    for name, value in params.items():
        rows.append({
            "Section": "Estimated Parameters",
            "Item": name,
            "Value": value
        })

    # Distribution rankings
    for _, result in table.iterrows():
        for column, value in result.items():
            rows.append({
                "Section": f"Ranking - {result['Distribution']}",
                "Item": column,
                "Value": value
            })

    # Moments comparison
    for _, result in mom.iterrows():
        for column in ["Raw data", "Fitted distribution", "Difference"]:
            rows.append({
                "Section": f"Moments - {result['Moment']}",
                "Item": column,
                "Value": result[column]
            })

    return pd.DataFrame(rows).to_csv(index=False)


# ----------------------------------------------------------------------
# MONTE CARLO VALIDATION
# ----------------------------------------------------------------------
TRUE_PARAMS = {  # (label, default) for the data-generating distribution
    "Normal": [("Mean μ", 50.0), ("Std dev σ", 10.0)],
    "Exponential": [("Scale (1/λ)", 5.0)],
    "Gamma": [("Shape α", 2.0), ("Scale θ", 3.0)],
    "Lognormal": [("σ (shape)", 0.5), ("μ (mean of ln x)", 2.0)],
    "Weibull": [("Shape k", 1.8), ("Scale λ", 10.0)],
    "Poisson": [("λ (mean)", 6.0)],
    "Geometric": [("p (success prob.)", 0.25)],
    "Binomial": [("n (trials)", 20.0), ("p (success prob.)", 0.3)],
    "Negative Binomial": [("r (successes)", 5.0), ("p (success prob.)", 0.4)],
}


def true_scipy_params(name, v):
    return {
        "Normal": lambda: (v[0], v[1]),
        "Exponential": lambda: (0, v[0]),
        "Gamma": lambda: (v[0], 0, v[1]),
        "Lognormal": lambda: (v[0], 0, float(np.exp(v[1]))),
        "Weibull": lambda: (v[0], 0, v[1]),
        "Poisson": lambda: (v[0],),
        "Geometric": lambda: (v[0], 0),
        "Binomial": lambda: (int(v[0]), v[1]),
        "Negative Binomial": lambda: (v[0], v[1]),
    }[name]()


def run_monte_carlo(true_name, discrete, args, sizes, reps, seed, bar):
    rng = np.random.default_rng(seed)

    pool = DISCRETE if discrete else CONTINUOUS
    names = list(pool)
    dist = pool[true_name]

    wins, est = [], []
    done = 0
    total = len(sizes) * reps

    for n in sizes:
        for _ in range(reps):

            x = dist.rvs(*args, size=n, random_state=rng)

            if discrete:
                x = np.asarray(x, dtype=int)

                trials = (
                    int(args[0])
                    if true_name == "Binomial"
                    else None
                )

                candidates = [
                    name for name in names
                    if name != "Binomial" or trials is not None
                ]

                res, _ = fit_discrete(
                    x, candidates, binomial_trials=trials
                )

            else:
                res, _ = fit_continuous(x, names)

            done += 1
            bar.progress(
                done / total,
                text=f"Simulating... {done}/{total}"
            )

            if not res:
                continue

            aic = {
                name: 2 * r["k"] - 2 * r["ll"]
                for name, r in res.items()
            }

            order = sorted(aic, key=aic.get)

            wins.append({
                "n": n,
                "winner": order[0],
                "top2": true_name in order[:2]
            })

            if true_name in res:
                est.append({
                    "n": n,
                    **named_params(
                        true_name,
                        res[true_name]["params"]
                    )
                })

    return pd.DataFrame(wins), pd.DataFrame(est)


def show_mc_results(r):
    wins, est, true_name, args, sizes = r["wins"], r["est"], r["true_name"], r["args"], r["sizes"]
    if wins.empty:
        st.error("No successful fits. Try different true parameters.")
        return
    g = wins.assign(hit=wins.winner == true_name).groupby("n")
    rec = pd.DataFrame({"Recovery rate (%)": g.hit.mean() * 100,
                        "True model in top 2 (%)": g.top2.mean() * 100,
                        "Replications": g.size()})
    cols = list(DISCRETE if r["discrete"] else CONTINUOUS)
    conf = (pd.crosstab(wins.n, wins.winner, normalize="index") * 100).reindex(columns=cols, fill_value=0)

    truth = named_params(true_name, args)
    rows = []
    for n, gp in est.groupby("n"):
        for p, tv in truth.items():
            e = gp[p]
            rows.append({"n": n, "Parameter": p, "True value": tv, "Mean estimate": e.mean(),
                         "Bias": e.mean() - tv, "Relative bias (%)": 100 * (e.mean() - tv) / tv if tv else np.nan,
                         "Std dev of estimates": e.std(ddof=1), "RMSE": np.sqrt(((e - tv) ** 2).mean())})
    ptab = pd.DataFrame(rows)

    best = rec["Recovery rate (%)"]
    st.success(f"True model: **{true_name}** · recovery rate: "
               + ", ".join(f"n={n}: **{v:.0f}%**" for n, v in best.items()))

    t1, t2, t3 = st.tabs(["🎯 Model recovery", "📐 Parameter recovery", "📈 Charts"])
    with t1:
        st.markdown("**How often did the app pick the true distribution as the winner (lowest AIC)?**")
        st.dataframe(rec.round(1))
        st.markdown("**Which distribution won, % of replications** (rows add to 100%)")
        st.dataframe(conf.round(1))
        st.caption("Gamma, Lognormal and Weibull are similar in shape at small n, so some confusion between them is expected. "
                   "AIC picks the best *approximation*, which is not always the true family.")
    with t2:
        st.markdown(f"**Estimates of the true family ({true_name}) across replications**")
        st.dataframe(ptab.round(4), hide_index=True)
        if true_name == "Binomial":
            st.caption("Binomial's n is estimated as the maximum observed value, so it is biased downward by construction.")
        st.download_button("Download parameter table (CSV)", ptab.to_csv(index=False), "mc_parameters.csv", "text/csv")
    with t3:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        conf.plot(kind="bar", stacked=True, ax=ax[0], colormap="Set2", edgecolor="black", rot=0)
        ax[0].set(title="Winning distribution by sample size", xlabel="Sample size n", ylabel="% of replications")
        ax[0].legend(title="Winner", fontsize=8)
        ax[1].plot(rec.index.astype(str), rec["Recovery rate (%)"], "o-", lw=2.5, label="True model wins")
        ax[1].plot(rec.index.astype(str), rec["True model in top 2 (%)"], "s--", label="True model in top 2")
        ax[1].set(title=f"Recovery of {true_name} vs sample size", xlabel="Sample size n",
                  ylabel="% of replications", ylim=(0, 105))
        ax[1].legend(); ax[1].grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        if not est.empty:
            fig2, axs = plt.subplots(1, len(truth), figsize=(6 * len(truth), 4.5), squeeze=False)
            for a, (p, tv) in zip(axs[0], truth.items()):
                a.boxplot([est.loc[est.n == n, p].dropna() for n in sizes])
                a.set_xticklabels(sizes)
                a.axhline(tv, color="red", ls="--", label=f"True = {tv:.4g}")
                a.set(title=f"Estimates of {p}", xlabel="Sample size n")
                a.legend(); a.grid(alpha=0.3, linestyle="--")
            fig2.tight_layout()
            st.pyplot(fig2)


def mc_page():
    st.subheader("🧪 Monte Carlo Validation")
    st.write("Generate data from a distribution **you choose**, run this app's fitting engine on it many times, "
             "and check whether it recovers the truth (model choice and parameters).")
    c1, c2 = st.columns(2)
    family = c1.radio("Family", ["Continuous", "Discrete"], horizontal=True, key="p_mc_family")
    discrete = family == "Discrete"
    true_name = c2.selectbox("True (data-generating) distribution", list(DISCRETE if discrete else CONTINUOUS),
                             key=f"p_mc_true_{family}")
    spec = TRUE_PARAMS[true_name]
    cols = st.columns(len(spec))
    vals = []
    for c, (lab, default) in zip(cols, spec):
        k = f"p_mc_{true_name}_{lab}"
        vals.append(c.number_input(lab, key=k, **init(k, value=default)))

    c3, c4, c5 = st.columns(3)
    sizes = c3.multiselect("Sample sizes", [20, 30, 50, 100, 200, 500, 1000], key="p_mc_sizes",
                           **init("p_mc_sizes", default=[30, 100, 500]))
    reps = c4.slider("Replications per size", 20, 500, step=10, key="p_mc_reps", **init("p_mc_reps", value=100))
    seed = int(c5.number_input("Random seed", step=1, key="p_mc_seed", **init("p_mc_seed", value=42)))
    st.caption("Gamma and Weibull fits are slower; start with ~50–100 replications.")

    positive = {"Normal": vals[1:], "Lognormal": vals[:1]}.get(true_name, vals)
    valid = all(v > 0 for v in positive)
    if true_name in ("Geometric", "Binomial", "Negative Binomial") and not 0 < vals[-1] < 1:
        valid = False
    if not valid:
        st.error("Invalid parameters (scales/shapes must be positive and probabilities must be between 0 and 1).")
        return
    if not sizes:
        st.warning("Select at least one sample size.")
        return

    if st.button("▶ Run Monte Carlo simulation", type="primary"):
        args = true_scipy_params(true_name, vals)
        bar = st.progress(0.0, text="Simulating…")
        wins, est = run_monte_carlo(true_name, discrete, args, sorted(sizes), reps, seed, bar)
        bar.empty()
        st.session_state["mc"] = dict(wins=wins, est=est, true_name=true_name, args=args,
                                      sizes=sorted(sizes), discrete=discrete)
    r = st.session_state.get("mc")
    if r:
        st.divider()
        show_mc_results(r)


def paywall(feature):
    st.markdown(f'<div class="verdict" style="border-color:#f59e0b;background:rgba(245,158,11,.12)">'
                f'🔒 <b>{feature}</b> is a Pro feature. Compare plans below.</div>', unsafe_allow_html=True)
    pro_feats = ["All 9 distributions", "Unlimited rows", "Monte Carlo Validation", "Report export (.md)"]
    plans = [("Free", "$0", ["3 distributions at a time", "Up to 1,000 rows", "Ranking & dashboard"], ""),
             ("Pro Monthly", "$4.99 / mo", pro_feats, ""),
             ("Pro Yearly", "$29.99 / yr", pro_feats + ["<b>Save 50%</b>"], " best")]
    for col, (name, price, feats, cls) in zip(st.columns(3), plans):
        col.markdown(f'<div class="plan{cls}"><h4>{name}</h4><div class="price">{price}</div><ul>'
                     + "".join(f"<li>{f}</li>" for f in feats) + "</ul></div>", unsafe_allow_html=True)
    st.caption("Illustrative pricing. This demo build has no payment backend; use **Preview Free plan** "
               "in the sidebar to switch plans.")


# ----------------------------------------------------------------------
# PER-DATASET ANALYSIS & RENDERING
# ----------------------------------------------------------------------
def prep_dataset(ds, mode, cand_c, cand_d, is_pro):
    vals = ds["values"]
    x = vals[np.isfinite(vals)]
    out = dict(error=None, notes=[], analyses=[], dropped=len(vals) - len(x), x=x)
    if len(x) < 10:
        out["error"] = f"Only {len(x)} valid numbers found. Please provide at least 10."
        return out
    if not is_pro and len(x) > 1000:
        x = out["x"] = x[:1000]
        out["notes"].append("Free plan analyses the first 1,000 values. Upgrade for unlimited rows.")
    is_count = bool(
    np.all(np.isfinite(x))
    and np.all(x >= 0)
    and np.all(x == np.floor(x))
)
    if mode == "Auto-detect":
        fams = ["Discrete"] if (is_count and len(np.unique(x)) <= 0.5 * len(x)) else ["Continuous"]
    elif mode == "Continuous":
        fams = ["Continuous"]
    elif mode.startswith("Discrete"):
        fams = ["Discrete"]
    else:
        fams = ["Continuous", "Discrete"]
    for fam in fams:
        discrete = fam == "Discrete"
        xf = x
        if discrete and not is_count:
          out["notes"].append("Discrete fitting skipped: this dataset does not contain valid non-negative integer counts.")
          continue


        chosen = cand_d if discrete else cand_c
        if not chosen:
            out["notes"].append(f"No {fam.lower()} distributions are selected in the sidebar.")
            continue
        results, skipped = (fit_discrete if discrete else fit_continuous)(xf, chosen)
        if not results:
            out["notes"].append(f"None of the selected {fam.lower()} distributions could be fitted.")
            continue
        table = rank_table(results, len(xf))
        out["analyses"].append(dict(family=fam, discrete=discrete, x=xf, results=results, skipped=skipped,
                                    table=table, winner=table.loc[0, "Distribution"]))
    if not out["analyses"]:
        out["error"] = "Nothing could be fitted. See the messages above."
    return out


def render_dataset(ds, prep, idx, is_pro, show_all):
    st.caption(f"Source: {ds['label']}")
    for n in dict.fromkeys(prep["notes"]):
        st.warning(n)
    if prep["error"]:
        st.error(prep["error"])
        st.info("💡 If each **row** of your file is a data series and the column headers are years/months, "
                "set **Data layout** in the sidebar to *Series in rows (periods across columns)*.")
        return
    analyses, multi = prep["analyses"], len(prep["analyses"]) > 1

    for a in analyses:
        t = a["table"]
        if multi:
            st.markdown(f"#### {a['family']} family")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Best fit (lowest AIC)", a["winner"])
        c2.metric("AIC", f"{t.loc[0, 'AIC']:.2f}")
        c3.metric("Observations", len(a["x"]))
        c4.metric("Data type", a["family"])
        st.markdown(f'<div class="verdict">💡 {verdict(t)}</div>', unsafe_allow_html=True)
        if a["skipped"]:
            st.warning("Skipped: " + "; ".join(a["skipped"]))
    if multi:
        st.caption("Continuous (density) and discrete (probability) AIC values are not comparable, "
                   "so each family is ranked separately.")

    tab1, tab2, tab3, tab4 = st.tabs(["🏁 Ranking", "🎯 Winner: parameters & moments", "📈 Dashboard", "🗂 Data"])

    with tab1:
        for a in analyses:
            st.subheader(f"{a['family']} ranking" if multi else "Goodness-of-fit ranking")
            st.dataframe(a["table"].round(4).style.apply(
                lambda r: ["background-color: rgba(34,197,94,.20)" if r.name == 0 else "" for _ in r], axis=1),
                hide_index=True)
            st.download_button("⬇ Download ranking (CSV)", a["table"].to_csv(index=False),
                               f"ranking_{a['family'].lower()}.csv", "text/csv", key=f"dl_rank_{idx}_{a['family']}")
        st.markdown(
            "- **AIC / BIC**: lower is better. A ΔAIC below ~2 means the model is practically tied with the best one.\n"
            "- **Rank** is by AIC; **BIC Rank** shows whether BIC agrees. **Akaike weight** is the relative support for each model (sums to 1).\n"
            "- **p-value**: above 0.05 means no significant evidence of misfit. "
            + ("Chi-square degrees of freedom account for estimated parameters. " if any(a["discrete"] for a in analyses) else "")
            + ("KS p-values are optimistic because parameters were estimated from the same data; treat them as a guide."
               if any(not a["discrete"] for a in analyses) else ""))

    with tab2:
        for a in analyses:
            winner, wres = a["winner"], a["results"][a["winner"]]
            st.subheader(f"Winning {a['family'].lower()} distribution: {winner}" if multi
                         else f"Winning distribution: {winner}")
            params = named_params(winner, wres["params"])
            st.markdown("**Estimated parameters**")
            for c, (k, v) in zip(st.columns(len(params)), params.items()):
                c.metric(k, f"{v:.4f}")
            st.markdown("**Moments: raw data vs fitted distribution**")
            mom = moments_table(a["x"], wres)
            st.dataframe(mom.round(4), hide_index=True)
            kk = f"{idx}_{a['family']}"
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("⬇ Parameters (CSV)",
                               pd.DataFrame({"Parameter": list(params), "Estimate": list(params.values())}).to_csv(index=False),
                               "winner_parameters.csv", "text/csv", key=f"dl_par_{kk}")
            d2.download_button("⬇ Moments (CSV)", mom.to_csv(index=False), "winner_moments.csv", "text/csv",
                               key=f"dl_mom_{kk}")


            if is_pro:

                report_csv = build_csv_report(ds["label"],len(a["x"]),a["discrete"],a["table"],winner,params,mom)

                d3.download_button("⬇ Full report (CSV)",data=report_csv,file_name="distribution_fit_report.csv",mime="text/csv",key=f"dl_report_{kk}")

            else:
                d3.caption("🔒 Full report exports are a Pro feature.")


    with tab3:
        for a in analyses:
            if multi:
                st.subheader(f"{a['family']} dashboard")
            st.pyplot(dashboard(a["x"], a["results"], a["winner"], a["discrete"], show_all))

    with tab4:
        x = prep["x"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Valid values used", len(x))
        d2.metric("Dropped (NaN/inf)", prep["dropped"])
        d3.metric("Minimum", f"{x.min():.4g}")
        d4.metric("Maximum", f"{x.max():.4g}")
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Data preview**")
            st.dataframe(ds["preview"].head(200))
        with right:
            st.markdown("**Descriptive statistics**")
            st.dataframe(pd.Series(x, name="value").describe().to_frame().round(4))


def render_compare(shorts, preps):
    ok = [(n, p) for n, p in zip(shorts, preps) if not p["error"]]
    if len(ok) < 2:
        st.info("At least two datasets with valid data are needed for a comparison.")
        return
    names = [n for n, _ in ok]
    raw = {n: p["x"] for n, p in ok}
    st.subheader("Dataset comparison")
    meds = [abs(np.median(a)) for a in raw.values() if np.median(a) != 0]
    very_different = len(meds) > 1 and max(meds) / min(meds) > 10
    scale = st.radio("Compare plots and tests on", ["Raw values", "Standardised (z-score)"], horizontal=True,
                     key="p_cmp_scale", **init("p_cmp_scale", index=1 if very_different else 0),
                     help="Use z-scores when the datasets are on different scales (e.g. GWh vs $ million): "
                          "this compares the SHAPE of the distributions, not their size.")
    use_raw = scale == "Raw values"


    if very_different and use_raw:
        st.caption(
            "These datasets differ substantially in scale. "
            "Try the standardised view for clearer comparisons."
        )

    if not use_raw:
        constant = [
            name for name, values in raw.items()
            if (
                not np.isfinite(np.std(values, ddof=1))
                or np.std(values, ddof=1) == 0
            )
        ]

        if constant:
            st.error(
                "Cannot standardise these datasets: "
                + ", ".join(constant)
            )
            return

    data = {
        name: (
            values if use_raw
            else (values - values.mean()) / values.std(ddof=1)
        )
        for name, values in raw.items()
    }

    summ = pd.DataFrame([
        {
            "Dataset": name,
            "n": len(values),
            "Mean": values.mean(),
            "Median": np.median(values),
            "Std dev": values.std(ddof=1),
            "CV (std/mean)": (
                values.std(ddof=1) / abs(values.mean())
                if values.mean() else np.nan
            ),
            "Min": values.min(),
            "Max": values.max(),
            "Skewness": stats.skew(values),
            "Kurtosis": stats.kurtosis(
                values, fisher=False
            )
        }
        for name, values in raw.items()
    ])

    st.markdown("**Descriptive statistics (raw values)**")
    st.dataframe(summ.round(4), hide_index=True)

    if not use_raw:
        z_summary = pd.DataFrame([
            {
                "Dataset": name,
                "n": len(values),
                "Mean (z)": np.mean(values),
                "Median (z)": np.median(values),
                "Std dev (z)": np.std(values, ddof=1),
                "Min (z)": np.min(values),
                "Max (z)": np.max(values),
                "Skewness": stats.skew(values),
                "Kurtosis": stats.kurtosis(
                    values, fisher=False
                )
            }
            for name, values in data.items()
        ])

        st.markdown("**Standardised descriptive statistics**")
        st.dataframe(z_summary.round(4), hide_index=True)

    st.caption(
        "The best-fitting distributions and AIC values "
        "below were calculated using raw observations. "
        "Standardisation affects comparison plots and tests, "
        "not the original fitted models."
    )


    # 2. best fits side by side
    rows = []
    for n, p in ok:
        for a in p["analyses"]:
            params = named_params(a["winner"], a["results"][a["winner"]]["params"])
            rows.append({"Dataset": n, "Family": a["family"], "Best fit": a["winner"],
                         "AIC": a["table"].loc[0, "AIC"],
                         "Parameters": ", ".join(f"{k} = {v:.4g}" for k, v in params.items())})
    fits = pd.DataFrame(rows)
    st.markdown("**Best-fitting distribution per dataset**")
    st.dataframe(fits.round(3), hide_index=True)
    for fam in fits["Family"].unique():
        w = fits[fits["Family"] == fam]
        if len(w) > 1:
            if w["Best fit"].nunique() == 1:
                st.success(f"All datasets share the same best-fit {fam.lower()} distribution: **{w['Best fit'].iloc[0]}**.")
            else:
                st.warning(f"Best-fit {fam.lower()} distributions differ: "
                           + ", ".join(f"{d} → **{bf}**" for d, bf in zip(w["Dataset"], w["Best fit"])))

    # 3. overlay plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    edges = np.histogram_bin_edges(np.concatenate(list(data.values())), "auto")
    for n in names:
        ax[0].hist(data[n], bins=edges, density=True, alpha=0.35, label=n)
        xs = np.sort(data[n])
        ax[1].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", label=n)
    ax[0].set(title="Distribution overlay", ylabel="Density")
    ax[1].set(title="Empirical CDFs", ylabel="Cumulative probability")
    ax[2].boxplot([data[n] for n in names])
    ax[2].set_xticklabels(names, rotation=15)
    ax[2].set(title="Boxplots")
    unit = "value" if use_raw else "z-score"
    ax[0].set_xlabel(unit); ax[1].set_xlabel(unit)
    ax[0].legend(); ax[1].legend()
    fig.tight_layout()
    st.pyplot(fig)

    # 4. pairwise tests
    tests = []
    for a, b in itertools.combinations(names, 2):
        ks = stats.ks_2samp(data[a], data[b])
        mw = stats.mannwhitneyu(data[a], data[b], alternative="two-sided")
        lev = stats.levene(raw[a], raw[b]).pvalue if use_raw else np.nan
        tests.append({"Dataset A": a, "Dataset B": b, "KS statistic": ks.statistic, "KS p-value": ks.pvalue,
                      "Mann-Whitney p-value": mw.pvalue, "Levene p-value (equal variance)": lev})
    tdf = pd.DataFrame(tests)
    st.markdown("**Pairwise tests: do the datasets come from the same distribution?**")
    st.dataframe(tdf.round(4), hide_index=True)
    st.caption("Small p-values (< 0.05) suggest the two datasets differ. The KS statistic (0 to 1) is the size of the "
               "difference; with large samples p-values become tiny even for small differences, so read both. "
               "Time series are autocorrelated, which makes p-values optimistic. "
               + ("" if use_raw else "On z-scores, the tests compare shape only, and the variance test is not applicable."))

    # ========================================================
    # DOWNLOAD DATASET COMPARISON AS CSV
    # ========================================================

    comparison_rows = []

    for section, dataframe in [
        ("Descriptive Statistics", summ),
        ("Best-fitting Distributions", fits),
        ("Pairwise Tests", tdf)
    ]:

        for _, record in dataframe.iterrows():

            for column, value in record.items():

                comparison_rows.append({
                    "Section": section,
                    "Record": str(record.get("Dataset", record.get("Dataset A", ""))),
                    "Statistic": column,
                    "Value": value
                })

    comparison_csv = pd.DataFrame(
        comparison_rows
    ).to_csv(index=False)

    st.download_button(
        "⬇ Download comparison (CSV)",
        data=comparison_csv,
        file_name="dataset_comparison.csv",
        mime="text/csv",
        key="dl_cmp"
    )

# ----------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------
st.markdown('<div class="hero"><h1>📊 DistriFit Pro</h1>'
            '<p>Find the best-fitting probability distribution for any dataset in seconds</p>'
            '<div class="chips"><span>9 distributions</span><span>AIC / BIC ranking</span>'
            '<span>Moments &amp; parameters</span><span>Compare datasets</span><span>Monte Carlo Validation</span></div></div>',
            unsafe_allow_html=True)


# =====================================================
# USER123 DEMO ACCOUNT - DISTRIFIT PRO
# =====================================================

if "current_user" not in st.session_state:
    st.session_state.current_user = "user123"

# Account menu positioned above the main banner
left_space, right_account = st.columns([8, 2])

with right_account:
    with st.popover(
        f"👤 {st.session_state.current_user} ▼",
        use_container_width=True
    ):
        st.markdown("### User Profile")
        st.write("**Username:**", st.session_state.current_user)
        st.caption("BR2207 Group 4 | Demo Reviewer")

        st.divider()

        if st.button("View Profile", key="view_user_profile"):
            st.session_state.show_demo_profile = True

        if st.button("Switch to user123", key="switch_demo_user"):
            st.session_state.current_user = "user123"
            st.rerun()

        if st.button("Log Out", key="logout_demo_user"):
            st.session_state.current_user = "Guest"
            st.rerun()

if st.session_state.get("show_demo_profile", False):
    with st.expander("User Profile", expanded=True):
        st.write("Username:", st.session_state.current_user)
        st.write("Account type: Demo Reviewer")
        st.write("Project: BR2207 Group 4")

        if st.button("Close Profile", key="close_demo_profile"):
            st.session_state.show_demo_profile = False
            st.rerun()

st.sidebar.markdown("### 📊 DistriFit Pro")
preview_free = st.sidebar.toggle("Preview Free plan",
                                 help="Demo build: all Pro features are unlocked. Turn this on to see the Free plan limits.")
is_pro = not preview_free
st.sidebar.caption("💎 **Pro plan active**" if is_pro
                   else "🆓 **Free plan:** 3 models, 1,000 rows, no Monte Carlo or reports")
app_mode = st.sidebar.radio("App mode", ["📂 Fit my data", "🧪 Monte Carlo Validation"])
up = data_uploader()  # rendered in BOTH modes, at the same position, so uploaded files stay loaded
if app_mode.startswith("🧪"):
    if up:
        st.sidebar.caption("📎 Your files stay loaded while you run the validation.")
    mc_page() if is_pro else paywall("Monte Carlo Validation")
    st.stop()

datasets = get_datasets(up, is_pro)
if not datasets:
    if up:
        st.info("👈 Tick at least one data series in the sidebar.")
    for col, (t, d) in zip(st.columns(3), [
            ("1 · Load data", "Upload one or more CSV/Excel files, paste numerical values, or try the built-in example from the sidebar."),
            ("2 · Choose settings", "Pick the data type and which candidate distributions to test."),
            ("3 · Read the results", "See the ranking, the winner's parameters and moments, the dashboard, and a comparison across files.")]):
        col.markdown(f'<div class="step"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)
    st.caption("👈 Start in the sidebar. Switch App mode to run the Monte Carlo Validation.")
    st.stop()

st.sidebar.header("2 · Analysis settings")
mode = st.sidebar.radio(
    "Data type", ["Auto-detect", "Continuous", "Discrete (counts)", "Combined (continuous + discrete)"], key="p_mode",
    help="Continuous and discrete models can't be ranked against each other (density vs probability). "
         "**Combined** fits both families and ranks each separately.")
cand_c, cand_d = [], []
lim = dict(max_selections=None if is_pro else 3, help=None if is_pro else "Free plan: up to 3 distributions per family.")
if mode != "Discrete (counts)":
    k1 = f"p_cand_c_{is_pro}"
    cand_c = st.sidebar.multiselect("Continuous candidates", list(CONTINUOUS), key=k1, **lim,
                                    **init(k1, default=list(CONTINUOUS) if is_pro else list(CONTINUOUS)[:3]))
if mode != "Continuous":
    k2 = f"p_cand_d_{is_pro}"
    cand_d = st.sidebar.multiselect("Discrete candidates", list(DISCRETE), key=k2, **lim,
                                    **init(k2, default=list(DISCRETE) if is_pro else list(DISCRETE)[:3]))
show_all = st.sidebar.checkbox("Show all fits in plots", key="p_showall", **init("p_showall", value=True))

shorts = []
for ds in datasets:
    nm = (ds.get("short") or Path(ds["name"]).stem)[:30]
    while nm in shorts:
        nm += "'"
    shorts.append(nm)
preps = [prep_dataset(ds, mode, cand_c, cand_d, is_pro) for ds in datasets]

if len(datasets) == 1:
    render_dataset(datasets[0], preps[0], 0, is_pro, show_all)
else:
    tabs = st.tabs([f"📄 {n}" for n in shorts] + ["⚖ Compare"])
    for i, tb in enumerate(tabs[:-1]):
        with tb:
            render_dataset(datasets[i], preps[i], i, is_pro, show_all)
    with tabs[-1]:
        render_compare(shorts, preps)

st.divider()
st.caption("DistriFit Pro v1.0 · Your data is processed in-session and is not stored by this app.")
