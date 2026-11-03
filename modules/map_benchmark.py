"""
map_benchmark.py
================

Output benchmark for codex_rna.ipynb (section 12, for Google Colab). Covers two tests only.

1. Held-out marker test (comparing RNA mapping methods) — holdout_compare
   * Link markers (protein ↔ gene) are split into n_folds groups; per group those markers are left out and **every method is mapped under the same conditions**
     (knn / scmodal / maxfuse). Spearman r between the mapped RNA of held-out markers and the actual CODEX protein is measured within CODEX types.
   * For a fair comparison, only **cells mapped by every method in that group (shared cells)** are used.
   * null = mapping rows shuffled within the same type (= a mapping that only matches type means, the same information as the 11-1 type-mean mapping).
     gain = r − null. The type-mean mapping has gain = 0 by definition.
   * Unit of judgement = marker (median across types per marker). Types of the same marker are not independent, so cells are not tested as units.
     Per method: is marker gain > 0 (one-sided Wilcoxon), and is it larger than the reference method (methods[0]) (paired one-sided Wilcoxon).
2. Niche remapping judgement summary — remap_judgement (only reads the section 11 results)
   * One judgement = (receiver type, sender type, route). If there are two routes (nichenet → similarity), both count as tests.
   * Pass count vs chance expectation (tests × α): binomial test. Running many tests produces pairs that pass by chance,
     so passes must exceed this before saying 'hypothesis-based remapping is right more often than chance'.
   * Gain over random placement: per test, (gain − null mean) averaged over judging proteins → greater than 0? (one-sided Wilcoxon, per test).
     null = the CODEX-side hypothesis (distance to the sender type) shuffled among receiver cells, i.e. remapping with the hypothesis placed at random positions.
     All attempted tests are used rather than only passing pairs, so there is no selection bias.
   * BH-FDR (reference): the smallest p is 1/(N_NULL+1), so with many tests almost none can pass — the resolution limit is noted alongside.
   * Judging-protein correlation of the baseline mapping (reference): correlation between the actual proteins and a mapping (σ=∞) using only proteins other than the judging ones.
     This is one more held-out marker test for those proteins.

Results are saved in the same table format as mesmer's benchmark.py (a csv accumulating rows per run).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Benchmark", "load_runs", "holdout_compare", "HoldoutComparison", "remap_judgement"]


class Benchmark:
    """Collects output-test results into one table. run = name of this run (e.g. 'knn k=20')."""

    COLS = ["run", "output", "test", "metric", "value", "baseline", "baseline_name", "null_mean", "p", "passes", "note",
            "detail"]

    def __init__(self, run: str):
        self.run = run
        self.rows = []

    def add(self, output, test, metric, value, baseline=None, baseline_name=None, null_mean=None, p=None,
            passes=None, note="", detail=False):
        """detail=True = detail row (per marker·protein). headline() and load_runs() show only headline rows."""
        def num(v):
            if v is None or isinstance(v, (str, bool, np.bool_)):
                return v
            v = float(v)
            return None if np.isnan(v) else round(v, 4)

        self.rows.append(dict(run=self.run, output=output, test=test, metric=metric, value=num(value),
                              baseline=num(baseline), baseline_name=baseline_name, null_mean=num(null_mean), p=num(p),
                              passes=None if passes is None else bool(passes), note=note, detail=bool(detail)))

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLS)

    def headline(self) -> pd.DataFrame:
        T = self.table
        return T[~T["detail"]][["output", "test", "metric", "value", "baseline", "baseline_name", "null_mean", "p",
                                "passes", "note"]]

    def save(self, path: str):
        """Replace this run's rows in path (csv) (earlier rows with the same run name are removed). Rows of other runs are kept."""
        T = self.table.assign(time=time.strftime("%Y-%m-%d %H:%M"))
        if os.path.exists(path):
            old = pd.read_csv(path)
            T = pd.concat([old[old["run"] != self.run], T], ignore_index=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        T.to_csv(path, index=False)
        print(f"[bench] {len(self.rows)} rows → {path} (runs: {list(pd.unique(T['run']))})")


def load_runs(path: str, headline_only: bool = True) -> pd.DataFrame:
    """Comparison table of several saved runs, spread as (output, test, metric) × run."""
    T = pd.read_csv(path)
    if headline_only and "detail" in T:
        T = T[~T["detail"].astype(bool)]
    return T.pivot_table(index=["output", "test", "metric"], columns="run", values="value", aggfunc="first")


def _wilcoxon_greater(x) -> float:
    from scipy.stats import wilcoxon

    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) < 5:
        return np.nan
    return float(wilcoxon(x, alternative="greater").pvalue)


# ---------------------------------------------------------------------------
# 1. Held-out marker test: comparing methods
# ---------------------------------------------------------------------------
@dataclass
class HoldoutComparison:
    cells: pd.DataFrame          # fold, marker, codex_type, method, n, r, null, gain (shared cells)
    coverage: pd.DataFrame       # fold × method: mapped CODEX cells, shared cells
    markers: pd.DataFrame        # marker × method: median gain across types
    summary: pd.DataFrame        # method: median marker gain, fraction of cells with gain>0.1, p (vs 0), difference·p vs reference
    params: Dict = field(default_factory=dict)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.cells.to_csv(os.path.join(out_dir, "holdout_cells.csv"), index=False)
        self.coverage.to_csv(os.path.join(out_dir, "holdout_coverage.csv"), index=False)
        self.markers.to_csv(os.path.join(out_dir, "holdout_markers.csv"))
        self.summary.to_csv(os.path.join(out_dir, "holdout_summary.csv"))
        with open(os.path.join(out_dir, "holdout_params.json"), "w") as f:
            json.dump(self.params, f, indent=1, default=str)

    def to_benchmark(self, bm: Benchmark):
        ref = self.params["methods"][0]
        for m, s in self.summary.iterrows():
            bm.add("RNA map", "held-out markers", f"median marker gain ({m})", s["median_gain"], 0.0,
                   "type mean (within-type shuffle)", None, s["p_vs_type_mean"],
                   bool(s["p_vs_type_mean"] <= 0.05 and s["median_gain"] > 0) if np.isfinite(s["p_vs_type_mean"])
                   else None,
                   note=f"{int(s['n_markers'])} markers, on shared cells; cells with gain>0.1 {s['frac_cells_gain_gt_0.1']:.0%}")
            if m != ref and np.isfinite(s.get("diff_vs_ref", np.nan)):
                bm.add("RNA map", "held-out markers", f"gain {m} − {ref}", s["diff_vs_ref"], 0.0, ref, None,
                       s["p_vs_ref"], bool(s["p_vs_ref"] <= 0.05 and s["diff_vs_ref"] > 0)
                       if np.isfinite(s["p_vs_ref"]) else None, note="median of per-marker paired differences")
        for mk, row in self.markers.iterrows():
            for m in self.markers.columns:
                bm.add("RNA map", "held-out markers", f"gain: {mk} ({m})", row[m], 0.0, detail=True)
        cov = self.coverage.groupby("method")[["mapped", "common"]].sum()
        for m, c in cov.iterrows():
            bm.add("RNA map", "held-out markers", f"coverage ({m})", c["mapped"] / max(c["common"], 1), detail=True,
                   note="mapped cells / shared cells (summed over folds)")


def _map_with(method, codex, rna, plan, lk, k, seed, maxfuse_mask, maxfuse_params, scmodal_dir, codex_key, rna_key):
    import rna_map as rm

    if method in ("knn", "scmodal"):
        return rm.map_cells(codex, rna, plan, lk, codex_key=codex_key, rna_key=rna_key, k=k, method=method,
                            scmodal_dir=scmodal_dir, seed=seed, verbose=False)
    if method == "maxfuse":
        import maxfuse_map as mfm

        mf = mfm.run_maxfuse(codex, rna, lk, params=maxfuse_params, rna_key=rna_key, codex_mask=maxfuse_mask,
                             seed=seed, verbose=False)
        return mf.to_cellmap(codex, rna, consistent_only=True, codex_key=codex_key, rna_key=rna_key)
    raise ValueError("method must be 'knn', 'scmodal', 'maxfuse'")


def holdout_compare(codex, rna, plan: pd.DataFrame, links: pd.DataFrame, methods: Sequence[str] = ("knn",),
                    n_folds: int = 4, n_null: int = 5, k: int = 20, min_cells: int = 50, max_tie_frac: float = 0.9,
                    seed: int = 0,
                    maxfuse_mask: Optional[np.ndarray] = None, maxfuse_params: Optional[Dict] = None,
                    scmodal_dir: str = "scmodal_models", codex_key: str = "cell_type", rna_key: str = "codex_type",
                    verbose: bool = True) -> HoldoutComparison:
    """Map methods with the same folds (held-out marker groups) and compare recovery of held-out markers. methods[0] is the reference method.

    maxfuse_mask : CODEX cells for which MaxFuse finds matches (a subset to save time). Shared cells shrink to this subset.
    max_tie_frac : if a single protein value exceeds this fraction within a type (almost all negative, e.g. CK7 in macrophages), that cell is dropped —
                   it carries no information and becomes a degenerate cell whose rank correlation is unchanged by shuffling.
    scmodal / maxfuse are retrained per fold, so they take long (it would not be fair if held-out markers entered training).
    """
    from scipy.stats import spearmanr

    methods = list(dict.fromkeys(methods))
    rng = np.random.default_rng(seed)
    used = list(links.index[links.used])
    folds = np.array_split(rng.permutation(used), max(1, min(n_folds, len(used))))
    ctype = codex.obs[codex_key].astype(str).to_numpy()
    names = list(map(str, codex.var_names))
    Xc = np.asarray(codex.layers["input"] if "input" in codex.layers else codex.X, dtype=np.float32)
    rows, cov = [], []
    t0 = time.time()
    for fi, fold in enumerate(folds):
        lk = links.copy()
        lk.loc[list(fold), "used"] = False
        cms = {}
        for m in methods:
            t1 = time.time()
            cms[m] = _map_with(m, codex, rna, plan, lk, k, seed, maxfuse_mask, maxfuse_params, scmodal_dir,
                               codex_key, rna_key)
            if verbose:
                print(f"[bench] fold {fi + 1}/{len(folds)} {m}: mapped cells {int(cms[m].mapped.sum()):,} "
                      f"({time.time() - t1:.0f}s)")
        common = np.logical_and.reduce([cm.mapped for cm in cms.values()])
        for m, cm in cms.items():
            cov.append(dict(fold=fi, method=m, mapped=int(cm.mapped.sum()), common=int(common.sum())))
        gl = {mk: links.loc[mk, "found"].split(",") for mk in fold}
        genes = sorted({g for v in gl.values() for g in v})
        imps = {m: cm.impute(rna, genes) for m, cm in cms.items()}
        for mk in fold:
            prot = Xc[:, names.index(mk)]
            for t in np.unique(ctype[common]):
                ii = np.where((ctype == t) & common)[0]
                if len(ii) < min_cells or np.unique(prot[ii], return_counts=True)[1].max() > max_tie_frac * len(ii):
                    continue
                perms = [rng.permutation(ii) for _ in range(n_null)]      # same shuffles for every method
                for m in methods:
                    pred = imps[m][gl[mk]].sum(1).to_numpy()
                    if np.ptp(pred[ii]) == 0:
                        continue
                    r = spearmanr(pred[ii], prot[ii])[0]
                    nul = float(np.nanmean([spearmanr(pred[p], prot[ii])[0] for p in perms]))
                    rows.append(dict(fold=fi, marker=mk, codex_type=t, method=m, n=len(ii), r=r, null=nul,
                                     gain=r - nul))
    cells = pd.DataFrame(rows, columns=["fold", "marker", "codex_type", "method", "n", "r", "null", "gain"])
    markers = cells.pivot_table(index="marker", columns="method", values="gain", aggfunc="median") \
        .reindex(columns=methods)
    ref = methods[0]
    summ = []
    for m in methods:
        g = markers[m].dropna() if m in markers else pd.Series(dtype=float)
        c = cells[cells.method == m]
        d = dict(method=m, n_markers=len(g), median_gain=float(g.median()) if len(g) else np.nan,
                 p_vs_type_mean=_wilcoxon_greater(g),
                 **{"frac_cells_gain_gt_0.1": float((c.gain > 0.1).mean()) if len(c) else np.nan},
                 median_r=float(c.r.median()) if len(c) else np.nan)
        if m != ref and m in markers and ref in markers:
            diff = (markers[m] - markers[ref]).dropna()
            d.update(diff_vs_ref=float(diff.median()) if len(diff) else np.nan, p_vs_ref=_wilcoxon_greater(diff))
        summ.append(d)
    summary = pd.DataFrame(summ).set_index("method")
    res = HoldoutComparison(cells=cells, coverage=pd.DataFrame(cov), markers=markers, summary=summary,
                            params=dict(methods=methods, n_folds=len(folds), n_null=n_null, k=k, min_cells=min_cells,
                                        max_tie_frac=max_tie_frac,
                                        seed=seed, folds=[list(map(str, f)) for f in folds],
                                        maxfuse_subset=None if maxfuse_mask is None else int(np.sum(maxfuse_mask))))
    if verbose:
        print(f"[bench] {len(used)} held-out markers × methods {methods} ({time.time() - t0:.0f}s)")
        print(summary.round(3).to_string())
    return res


# ---------------------------------------------------------------------------
# 2. Niche remapping judgement summary
# ---------------------------------------------------------------------------
def _remap_tables(src):
    """NicheRemapResult or a res.save folder → (routes, validation, params). The CellMap is not read."""
    if isinstance(src, str):
        j = lambda f: os.path.join(src, f)
        with open(j("niche_params.json")) as f:
            params = json.load(f)
        return pd.read_csv(j("routes.csv")), pd.read_csv(j("validation.csv")), params
    return src.routes, src.validation, src.params


def remap_judgement(bm: Benchmark, src, alpha: Optional[float] = None, min_gain: Optional[float] = None,
                    fdr: float = 0.1, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Summarise the section 11 niche remapping result (src = NicheRemapResult or saved folder) as benchmark rows."""
    from scipy.stats import binomtest

    routes, val, params = _remap_tables(src)
    alpha = params.get("alpha", 0.05) if alpha is None else alpha
    min_gain = params.get("min_gain", 0.02) if min_gain is None else min_gain
    n_null = int(params.get("n_null", 20))
    if val is None or not len(val):
        bm.add("niche remap", "judgement", "tests", 0, note="no pairs judged (no pair passed the adjacency·ligand conditions)")
        return {}
    val = val.assign(excess=val["gain"] - val["null_mean"])
    tests = val.groupby(["receiver", "sender", "route"]).agg(
        p=("p_max", "first"), max_gain=("gain", "max"), mean_excess=("excess", "mean"),
        best_protein=("gain", lambda s: val.loc[s.idxmax(), "protein"])).reset_index()
    tests["pass"] = (tests.p <= alpha) & (tests.max_gain >= min_gain)
    n, k_pass = len(tests), int(tests["pass"].sum())
    p_bin = float(binomtest(k_pass, n, alpha, alternative="greater").pvalue) if n else np.nan
    p_min = 1.0 / (n_null + 1)

    # BH (reference)
    order = np.argsort(tests.p.to_numpy())
    q = np.empty(n)
    q[order] = np.minimum.accumulate((tests.p.to_numpy()[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    tests["q_bh"] = np.minimum(q, 1)
    n_fdr = int(((tests.q_bh <= fdr) & (tests.max_gain >= min_gain)).sum())

    bm.add("niche remap", "judgement", "passed tests", k_pass, n * alpha, f"expected by chance (n×α, n={n})", None,
           p_bin, bool(p_bin <= 0.05) if np.isfinite(p_bin) else None,
           note=f"pass = p ≤ {alpha} and max gain ≥ {min_gain}; binomial test")
    bm.add("niche remap", "judgement", "mean excess over random placement", float(tests.mean_excess.median()), 0.0,
           "assumption shuffled among receiver cells", None, _wilcoxon_greater(tests.mean_excess),
           bool(_wilcoxon_greater(tests.mean_excess) <= 0.05 and tests.mean_excess.median() > 0)
           if np.isfinite(_wilcoxon_greater(tests.mean_excess)) else None,
           note=f"all {n} attempted tests (not only passing ones); mean over judging proteins of (gain − null mean)")
    bm.add("niche remap", "judgement", f"tests passing BH q ≤ {fdr}", n_fdr, note=(
        f"reference: the smallest p is 1/(N_NULL+1) = {p_min:.3f}, so passing BH with {n} tests "
        f"needs p ≤ {fdr / n:.4f}" + (" → impossible with this N_NULL, increase N_NULL" if p_min > fdr / n else "")))
    for rt, g in tests.groupby("route"):
        bm.add("niche remap", "judgement", f"passed ({rt})", int(g["pass"].sum()), len(g) * alpha,
               f"expected by chance (n={len(g)})", detail=True)
    base = val.drop_duplicates(["receiver", "protein"]).groupby("protein")["base"].median()
    bm.add("niche remap", "baseline map", "median ρ (validation proteins, protein-only map)", float(base.median()),
           note="reference: baseline map using only proteins other than the judging ones ↔ actual protein (neighbour mean, within the receiver type)")
    for pr, v in base.items():
        bm.add("niche remap", "baseline map", f"ρ: {pr}", v, detail=True)
    for _, t in tests.iterrows():
        bm.add("niche remap", "judgement", f"{t.receiver} ← {t.sender} ({t.route})", t.max_gain, None, None, None,
               t.p, bool(t["pass"]), note=f"best {t.best_protein}; mean excess {t.mean_excess:.3f}", detail=True)
    if verbose:
        print(f"[bench] niche remapping: {n} tests (routes {dict(tests.route.value_counts())}), {k_pass} passed "
              f"(chance expectation {n * alpha:.1f}, binomial p = {p_bin:.3g}) | median gain over random placement "
              f"{tests.mean_excess.median():.4f}")
        if p_min > alpha:
            print(f"[bench] note: with N_NULL={n_null} the smallest p is {p_min:.3f} > α={alpha}, so no pair can pass")
    return dict(tests=tests, baseline=base.to_frame("rho"))
