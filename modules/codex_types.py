"""
codex_types.py
==============

Module that sets cell types from markers in a CODEX cell table (AnnData: cell × marker, obsm['spatial']) (for Google Colab).

Method: cluster → annotate (cluster-then-annotate)
-----------------------------------------
Marker values of a single cell carry a lot of noise and spillover from neighbouring cells, so rather than thresholding each cell
it is more stable to group similar cells first and set types from group means.

1. preprocess   : tidy intensities (if already z-scored, clip the top only; raw intensities → arcsinh → z-score)
2. cluster      : PCA → kNN graph → Leiden (pynndescent + igraph without scanpy, to avoid version conflicts)
                  a high (fine) resolution so rare types (plasma cells etc.) get their own clusters
3. annotate     : MARKER_SETS scores on cluster-mean marker profiles → per-cluster type, score, gap to the runner-up (margin)
                  score = mean of the top k positive markers − 0.7 × max negative marker (positive part only).
                  low scores or a small gap between 1st·2nd are left as 'Unknown' / 'A|B?' for a person to check
3b. refine      : re-split and re-annotate only clusters mixing two types (small 1st·2nd gap, or cell scores poorly matching the cluster type)
4. check        : fraction of cells whose own scores match the cluster type (type_agree), flags co-positive exclusive markers (spillover),
                  and compares with given labels at lineage level if available (ARI, confusion table)

MARKER_SETS are defaults tuned to this data (tonsil·Barrett's oesophagus, 44 markers); markers missing from the panel are dropped automatically.
Type names serve as the shared vocabulary for matching scRNA types in rna_link (lineage is a coarser unit).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["MARKER_SETS", "GIVEN_TO_LINEAGE", "preprocess", "cluster", "refine", "cluster_profiles", "annotate",
           "set_cluster_types",
           "spillover_flags", "compare_labels", "plot_profiles", "plot_types", "plot_composition",
           "plot_confusion"]

# ---------------------------------------------------------------------------
# type definitions: pos = positive markers, k = average only the top k of pos (default: half rounded up → registers even if only some are strong,
#            k = len(pos) requires all), neg = markers that must be absent, lineage = coarse lineage
# ---------------------------------------------------------------------------
MARKER_SETS: Dict[str, Dict] = {
    # lymphocytes
    "B":             dict(pos=["CD20"], neg=["CD3"], lineage="B"),
    "CD4 T":         dict(pos=["CD3", "CD4"], k=2, neg=["CD20", "CD8", "FoxP3"], lineage="T"),
    "CD8 T":         dict(pos=["CD3", "CD8"], k=2, neg=["CD20", "CD4"], lineage="T"),
    "Treg":          dict(pos=["FoxP3", "CD25", "CD3"], k=2, neg=["CD20", "CD8"], lineage="T"),
    "Tfh":           dict(pos=["PD1", "CD4"], k=2, neg=["CD8", "FoxP3"], lineage="T"),
    "Plasma":        dict(pos=["CD38", "CD138"], neg=["CD20", "CD3"], lineage="Plasma"),
    "NK":            dict(pos=["CD56", "CD57"], neg=["CD3", "CD20", "Cytokeratin", "MUC1"], lineage="NK"),
    # myeloid (innate)
    "Macrophage":    dict(pos=["CD68", "CD163", "CD206"], neg=["CD3", "CD20"], lineage="Myeloid"),
    "DC":            dict(pos=["CD11c", "HLADR"], neg=["CD3", "CD20", "CD68"], lineage="Myeloid"),
    "Granulocyte":   dict(pos=["CD15", "CD11b", "MMP9", "Arginase1"], neg=["CD3", "CD20"], lineage="Myeloid"),
    "pDC":           dict(pos=["CD123", "CD303"], neg=["CD3", "CD20"], lineage="Myeloid"),   # absent from this tonsil panel, dropped automatically
    # vascular / stromal
    "Endothelial":   dict(pos=["CD31", "CD34"], neg=["CD45"], lineage="Endothelial"),
    "PDPN stroma":   dict(pos=["Podoplanin", "Vimentin"], neg=["CD45", "Cytokeratin"], lineage="Stroma"),
    "Fibroblast":    dict(pos=["Vimentin", "CD73"], neg=["CD45", "Cytokeratin", "Podoplanin"], lineage="Stroma"),
    "FDC":           dict(pos=["CD21"], neg=["CD3", "CD20"], lineage="Stroma"),
    "Smooth muscle": dict(pos=["aSMA"], neg=["CD45"], lineage="Stroma"),
    "Nerve":         dict(pos=["Nestin", "CD56"], k=2, neg=["CD45", "Cytokeratin"], lineage="Nerve"),
    # epithelial
    "Squamous epi":  dict(pos=["p63", "Cytokeratin"], neg=["CD45"], lineage="Epithelial"),
    "Glandular epi": dict(pos=["CK7", "MUC1", "Bcatenin", "Cytokeratin"], neg=["CD45", "p63"], lineage="Epithelial"),
    "Secretory epi": dict(pos=["MUC5AC", "MUC6"], neg=["CD45"], lineage="Epithelial"),
    "Paneth":        dict(pos=["aDefensin5"], neg=["CD45"], lineage="Epithelial"),
    "Neuroendocrine": dict(pos=["CHGA"], neg=["CD45"], lineage="Epithelial"),
}


# given labels of the STELLAR tonsil·Barrett's oesophagus table → this module's lineages (given_map of compare_labels)
GIVEN_TO_LINEAGE = {"Innate": "Myeloid", "PDPN": "Stroma", "SmoothMuscle": "Stroma", "Squamous_epithelial": "Epithelial",
                    "Glandular_epi": "Epithelial", "Secretory_epithelial": "Epithelial", "Paneth": "Epithelial"}


def _sets(var_names: Sequence[str], sets: Optional[Dict] = None, verbose: bool = False) -> Dict[str, Dict]:
    """Type definitions keeping only markers present in the panel. Types with no positive marker left are dropped."""
    have = set(map(str, var_names))
    out, dropped = {}, []
    for t, d in (sets or MARKER_SETS).items():
        pos = [m for m in d["pos"] if m in have]
        if not pos:
            dropped.append(t)
            continue
        k = min(d.get("k", int(np.ceil(len(d["pos"]) / 2))), len(pos))
        out[t] = dict(pos=pos, k=k, neg=[m for m in d.get("neg", []) if m in have], lineage=d.get("lineage", t))
    if verbose and dropped:
        print(f"[types] types dropped for having no positive marker in the panel: {dropped}")
    return out


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(a, clip_max: float = 5.0, transform: str = "auto", cofactor: float = 5.0, verbose: bool = True):
    """Keep the original values in a.layers['input'] and replace a.X with values for clustering (in place).

    transform
      'auto'   : if column means are roughly 0 and there are negative values, treat as already z-scored → 'clip', otherwise 'arcsinh'
      'clip'   : clip z-scores at clip_max from above only (so a few extreme values do not drag clusters)
      'arcsinh': raw intensity → arcsinh(x / cofactor) → per-marker z-score → clip_max
    """
    X = np.asarray(a.X, dtype=np.float32)
    if "input" not in a.layers:
        a.layers["input"] = X.copy()
    X = np.asarray(a.layers["input"], dtype=np.float32)
    if transform == "auto":
        transform = "clip" if X.min() < 0 and np.median(np.abs(X.mean(0))) < 0.5 else "arcsinh"
    if transform == "arcsinh":
        X = np.arcsinh(np.clip(X, 0, None) / cofactor)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    elif transform != "clip":
        raise ValueError("transform must be 'auto' / 'clip' / 'arcsinh'")
    a.X = np.clip(X, None, clip_max).astype(np.float32)
    a.uns["preprocess"] = dict(transform=transform, clip_max=clip_max, cofactor=cofactor)
    if verbose:
        print(f"[types] preprocessing: {transform}, clipped above {clip_max} | {a.n_obs:,} cells × {a.n_vars} markers")
    return a


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------
def _leiden(X: np.ndarray, n_pcs: Optional[int], n_neighbors: int, resolution: float, seed: int):
    """PCA → kNN (pynndescent) → Leiden (igraph). (labels, PCA coordinates, kNN indices) — labels 0, 1, 2 … from the largest cluster

    with n_pcs=None, X is used as is without PCA (when it is already an embedding, e.g. Harmony coordinates of scRNA)
    """
    import igraph as ig
    import leidenalg as la
    from pynndescent import NNDescent
    from sklearn.decomposition import PCA

    if n_pcs is None:
        Z = np.asarray(X, dtype=np.float32)
    else:
        n_pcs = min(n_pcs, X.shape[1] - 1, len(X) - 1)
        Z = PCA(n_pcs, random_state=seed).fit_transform(X).astype(np.float32)
    k = min(n_neighbors, len(X) - 1)
    idx, _ = NNDescent(Z, n_neighbors=k, random_state=seed, n_jobs=-1).neighbor_graph
    n = len(Z)
    g = ig.Graph(n=n, edges=np.c_[np.repeat(np.arange(n), k - 1), idx[:, 1:].ravel()], directed=False)
    g.simplify()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution,
                             seed=seed, n_iterations=2)
    lab = np.asarray(part.membership)
    order = pd.Series(lab).value_counts().index
    remap = np.empty(lab.max() + 1, int)
    remap[order] = np.arange(len(order))
    return remap[lab], Z, idx.astype(np.int32)


def _cols(a, markers):
    return [list(map(str, a.var_names)).index(m) for m in markers] if markers else slice(None)


def _ckey(c: str):
    """Sort cluster names ('3' < '3.1' < '12')."""
    return tuple(int(x) if x.isdigit() else 10**9 for x in str(c).split("."))


def cluster(a, n_pcs: int = 20, n_neighbors: int = 15, resolution: float = 2.0, markers: Optional[Sequence[str]] = None,
            seed: int = 0, verbose: bool = True):
    """PCA → kNN (pynndescent) → Leiden (igraph). Fills obs['cluster'], obsm['X_pca'], obsm['knn'].

    markers    : markers used for clustering (None = all). Better to leave out markers unrelated to type, such as nuclear stains
    resolution : larger = finer. Annotation uses cluster means, so a somewhat fine split (1.5–3) preserves rare types
    """
    t0 = time.time()
    X = np.asarray(a.X, dtype=np.float32)[:, _cols(a, markers)]
    lab, Z, idx = _leiden(X, n_pcs, n_neighbors, resolution, seed)
    n_cl = lab.max() + 1
    a.obs["cluster"] = pd.Categorical(lab.astype(str), categories=[str(i) for i in range(n_cl)])
    a.obsm["X_pca"] = Z
    a.obsm["knn"] = idx
    a.uns["cluster"] = dict(n_pcs=Z.shape[1], n_neighbors=n_neighbors, resolution=resolution, seed=seed,
                            markers=list(markers) if markers else None)
    if verbose:
        print(f"[types] {n_cl} clusters (resolution {resolution}, {n_neighbors} neighbours) | {time.time() - t0:.0f}s")
    return a


def refine(a, tab: pd.DataFrame, clusters: Optional[Sequence[str]] = None, resolution: float = 1.0,
           min_cells: int = 300, max_agree: float = 0.6, sets: Optional[Dict] = None, verbose: bool = True,
           **annotate_kw) -> pd.DataFrame:
    """Re-split only clusters that need checking (subclusters '22.0', '22.1' …) and re-annotate everything.

    clusters : clusters to split (None = automatic: 'A|B?', or not Unknown with cell agreement cell_agree < max_agree)
               separates two types mixed in one cluster (e.g. endothelial + plasma cells around vessels)
    min_cells: clusters smaller than this are not split
    returns  : new annotation table (same format as annotate)
    """
    if clusters is None:
        lab = tab["label"].astype(str)
        clusters = tab.index[(lab.str.endswith("?")) | ((lab != "Unknown") & (tab["cell_agree"] < max_agree))]
    clusters = [str(c) for c in clusters if tab.loc[str(c), "n_cells"] >= min_cells]
    if not clusters:
        if verbose:
            print("[types] no clusters to re-split")
        return tab
    p = a.uns.get("cluster", {})
    cl = a.obs["cluster"].astype(str).to_numpy().copy()
    X = np.asarray(a.X, dtype=np.float32)[:, _cols(a, p.get("markers"))]
    for c in clusters:
        ii = np.where(cl == c)[0]
        sub, _, _ = _leiden(X[ii], p.get("n_pcs", 20), p.get("n_neighbors", 15), resolution, p.get("seed", 0))
        cl[ii] = [f"{c}.{s}" for s in sub]
    a.obs["cluster"] = pd.Categorical(cl, categories=sorted(set(cl), key=_ckey))
    a.uns["cluster"]["refined"] = dict(clusters=clusters, resolution=resolution)
    if verbose:
        print(f"[types] re-split {len(clusters)} clusters: {clusters} → {len(set(cl))} in total")
    return annotate(a, sets=sets, verbose=verbose, **annotate_kw)


def cluster_profiles(a, key: str = "cluster") -> pd.DataFrame:
    """Cluster × marker means (on the preprocessed a.X)."""
    X = pd.DataFrame(np.asarray(a.X), columns=list(map(str, a.var_names)), index=a.obs_names)
    return X.groupby(a.obs[key].astype(str).to_numpy()).mean()


# ---------------------------------------------------------------------------
# 3. Annotation
# ---------------------------------------------------------------------------
def _score(P: pd.DataFrame, sets: Dict[str, Dict], neg_weight: float) -> pd.DataFrame:
    S = {}
    for t, d in sets.items():
        v = np.sort(P[d["pos"]].to_numpy(), axis=1)[:, ::-1][:, :d["k"]]
        s = pd.Series(v.mean(1), index=P.index)
        if d["neg"]:
            s = s - neg_weight * P[d["neg"]].clip(lower=0).max(axis=1)
        S[t] = s
    return pd.DataFrame(S)


def annotate(a, sets: Optional[Dict] = None, key: str = "cluster", min_score: float = 0.5, min_margin: float = 0.15,
             neg_weight: float = 0.7, verbose: bool = True) -> pd.DataFrame:
    """Set a type per cluster and fill obs['cell_type'], obs['lineage'], obs['type_margin'], obs['type_agree'].

    label rules
      score < min_score                       → 'Unknown'  (cluster without clear markers)
      1st·2nd gap < min_margin, different lineage → '1st|2nd?' (for a person to check; fix with set_cluster_types)
      1st·2nd gap < min_margin, same lineage  → 1st (e.g. CD4 T vs Tfh — the lineage is certain)
    type_agree : whether the cell's own top score matches the cluster type (to check heterogeneity within the cluster, 0/1)
    returned table: per cluster size, type, score, runner-up, margin, top markers
    """
    sets = _sets(a.var_names, sets, verbose=verbose)
    _keep_given(a, verbose)
    P = cluster_profiles(a, key)
    S = _score(P, sets, neg_weight)
    lin = {t: d["lineage"] for t, d in sets.items()}
    rows = {}
    for c in S.index:
        s = S.loc[c].sort_values(ascending=False)
        t1, t2 = s.index[0], s.index[1]
        margin = s.iloc[0] - s.iloc[1]
        if s.iloc[0] < min_score:
            label = "Unknown"
        elif margin < min_margin and lin[t1] != lin[t2]:
            label = f"{t1}|{t2}?"
        else:
            label = t1
        top = P.loc[c].nlargest(5)
        rows[c] = dict(label=label, score=round(float(s.iloc[0]), 2), second=f"{t2} {s.iloc[1]:.2f}",
                       margin=round(float(margin), 2), top_markers=", ".join(f"{k} {v:.1f}" for k, v in top.items()))
    tab = pd.DataFrame(rows).T
    tab.index.name = key
    tab.insert(0, "n_cells", a.obs[key].astype(str).value_counts().reindex(tab.index).to_numpy())
    tab = tab.loc[sorted(tab.index, key=_ckey)]
    # cluster × type score table. Kept out of DataFrame.attrs because pandas 2.2 compares attrs with == when printing·concatenating and errors
    a.uns["type_scores"] = S.round(2)
    _apply(a, tab, sets, key)
    # per-cell top score (checks heterogeneity within the cluster)
    X = pd.DataFrame(np.asarray(a.X), columns=list(map(str, a.var_names)))
    best = _score(X, sets, neg_weight).to_numpy().argmax(1)
    names = np.array(list(sets))
    a.obs["type_agree"] = (names[best] == a.obs["cell_type"].astype(str).to_numpy()).astype(np.int8)
    tab["cell_agree"] = a.obs.groupby(a.obs[key].astype(str), observed=True)["type_agree"].mean().reindex(tab.index).round(2)
    if verbose:
        vc = a.obs["cell_type"].value_counts()
        unsure = tab.label.str.contains(r"\?|Unknown").sum()
        print(f"[types] {len(vc)} types | {unsure} clusters need checking (Unknown / 'A|B?') | "
              f"cell scores agree with cluster type {a.obs['type_agree'].mean():.0%}")
    return tab


def _keep_given(a, verbose: bool = True):
    """Move existing 'cell_type' / 'lineage' columns of the table (given labels) to names with '_given' so they are not overwritten."""
    if "cluster_types" in a.uns:
        return
    for c in ("cell_type", "lineage"):
        if c in a.obs:
            a.obs[f"{c}_given"] = a.obs.pop(c)
            if verbose:
                print(f"[types] existing obs['{c}'] → moved to obs['{c}_given'] (for comparison)")


def _apply(a, tab: pd.DataFrame, sets: Dict, key: str):
    lin = {t: d["lineage"] for t, d in sets.items()}
    lab = a.obs[key].astype(str).map(tab["label"].astype(str))
    a.obs["cell_type"] = pd.Categorical(lab)
    a.obs["lineage"] = pd.Categorical(lab.map(lambda x: lin.get(x, "Unknown" if x == "Unknown" else "Uncertain")))
    a.obs["type_margin"] = a.obs[key].astype(str).map(tab["margin"].astype(float)).astype(np.float32)
    a.uns["cluster_types"] = tab["label"].astype(str).to_dict()


def set_cluster_types(a, tab: pd.DataFrame, fixes: Dict[str, str], sets: Optional[Dict] = None,
                      key: str = "cluster") -> pd.DataFrame:
    """Fix cluster types directly (e.g. {'17': 'NK', '21': 'Unknown'}). Returns the fixed table and updates obs too."""
    sets = _sets(a.var_names, sets)
    tab = tab.copy()
    for c, t in fixes.items():
        if str(c) not in tab.index:
            raise KeyError(f"No cluster '{c}'")
        if t not in sets and t != "Unknown":
            print(f"[types] warning: '{t}' is not a name in MARKER_SETS — lineage becomes 'Uncertain'")
        tab.loc[str(c), "label"] = t
    _apply(a, tab, sets, key)
    return tab


# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------
DEFAULT_EXCLUSIVE = [("CD3", "CD20"), ("CD3", "CD68"), ("CD20", "CD68"), ("CD45", "Cytokeratin"),
                     ("CD8", "CD4"), ("CD31", "CD45")]


def spillover_flags(a, pairs: Sequence[Tuple[str, str]] = DEFAULT_EXCLUSIVE, thr: float = 1.0,
                    verbose: bool = True) -> pd.DataFrame:
    """Cells where both markers of a pair that cannot co-occur in one cell are ≥ thr (z) → obs['spill'] (0/1).

    Mostly spillover from neighbouring cells or segmentation errors. Only flagged, not removed, so you can choose whether to exclude them from mapping·statistics.
    returned table: per pair, co-positive fraction and the chance co-positive fraction expected from each being positive alone
    """
    names = list(map(str, a.var_names))
    X = np.asarray(a.X)
    flag = np.zeros(a.n_obs, bool)
    rows = []
    for m1, m2 in pairs:
        if m1 not in names or m2 not in names:
            continue
        p1, p2 = X[:, names.index(m1)] >= thr, X[:, names.index(m2)] >= thr
        both = p1 & p2
        flag |= both
        rows.append(dict(pair=f"{m1}+{m2}", both=round(both.mean(), 4), expected=round(p1.mean() * p2.mean(), 4)))
    a.obs["spill"] = flag.astype(np.int8)
    tab = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
    if verbose:
        print(f"[types] cells co-positive for exclusive markers: {flag.sum():,} ({flag.mean():.1%}) → obs['spill']")
    return tab


def compare_labels(a, given: str, key: str = "lineage", given_map: Optional[Dict[str, str]] = None,
                   verbose: bool = True) -> Dict[str, object]:
    """Compare with given labels (given column). given_map renames given labels to our lineage names to match them.

    returns: ari, nmi, confusion (rows = given, columns = ours, row fractions), recall (agreement per given label)
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    g = a.obs[given].astype(str)
    if given_map:
        g = g.map(lambda x: given_map.get(x, x))
    o = a.obs[key].astype(str)
    conf = pd.crosstab(g, o)
    rec = pd.Series({lab: (conf.loc[lab, lab] / conf.loc[lab].sum()) if lab in conf.columns else 0.0
                     for lab in conf.index}).round(3)
    out = dict(ari=adjusted_rand_score(g, o), nmi=normalized_mutual_info_score(g, o),
               confusion=conf.div(conf.sum(1), axis=0).round(3), counts=conf, recall=rec)
    if verbose:
        print(f"[types] compared with '{given}' ({key}): ARI {out['ari']:.3f}, NMI {out['nmi']:.3f}")
        print("        agreement per label: " + ", ".join(f"{k} {v:.0%}" for k, v in rec.sort_values().items()))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _palette(names: Sequence[str]) -> Dict[str, tuple]:
    import matplotlib.pyplot as plt

    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    pal = {n: base[i % len(base)] for i, n in enumerate(sorted(names))}
    for n in names:
        if n == "Unknown" or n.endswith("?"):
            pal[n] = (0.8, 0.8, 0.8)
    return pal


def plot_profiles(a, tab: pd.DataFrame, key: str = "cluster", vmax: float = 3.0, figsize=None):
    """Cluster × marker mean heatmap (rows = clusters, sorted by type)."""
    import matplotlib.pyplot as plt

    P = cluster_profiles(a, key)
    order = tab.sort_values(["label", "n_cells"], ascending=[True, False]).index
    P = P.loc[order]
    fig, ax = plt.subplots(figsize=figsize or (0.28 * P.shape[1] + 3, 0.28 * P.shape[0] + 1.5))
    im = ax.imshow(P.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(P.shape[1]), P.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(P.shape[0]), [f"{c}: {tab.loc[c, 'label']} ({tab.loc[c, 'n_cells']:,})" for c in order],
                  fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, label="mean z")
    ax.set_title("cluster mean marker profile")
    fig.tight_layout()
    return fig


def plot_types(a, key: str = "cell_type", types: Optional[Sequence[str]] = None, max_points: int = 200_000,
               s: float = 0.5, figsize=(9, 9), seed: int = 0):
    """Type colours at cell positions. With types, only those types are coloured, the rest grey."""
    import matplotlib.pyplot as plt

    xy = a.obsm["spatial"]
    lab = a.obs[key].astype(str).to_numpy()
    idx = np.arange(a.n_obs)
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    names = sorted(set(lab))
    pal = _palette(names)
    fig, ax = plt.subplots(figsize=figsize)
    show = names if types is None else [t for t in names if t in types]
    if types is not None:
        rest = idx[~np.isin(lab[idx], show)]
        ax.scatter(xy[rest, 0], xy[rest, 1], s=s, c=[(0.88, 0.88, 0.88)], linewidths=0, rasterized=True)
    for t in show:
        ii = idx[lab[idx] == t]
        ax.scatter(xy[ii, 0], xy[ii, 1], s=s, color=pal[t], label=f"{t} ({(lab == t).sum():,})", linewidths=0,
                   rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{key} ({a.uns.get('spatial_unit', '')})")
    ax.legend(markerscale=12, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    return fig


def plot_composition(a, key: str = "cell_type", by: Optional[str] = None, figsize=(8, 4)):
    """Type fraction bars (stacked per column value with by)."""
    import matplotlib.pyplot as plt

    lab = a.obs[key].astype(str)
    if by is None:
        vc = lab.value_counts()
        fig, ax = plt.subplots(figsize=figsize)
        pal = _palette(vc.index)
        ax.bar(range(len(vc)), vc.to_numpy() / vc.sum(), color=[pal[t] for t in vc.index])
        ax.set_xticks(range(len(vc)), vc.index, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("fraction")
    else:
        tab = pd.crosstab(a.obs[by].astype(str), lab, normalize="index")
        pal = _palette(tab.columns)
        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(tab))
        for t in tab.columns:
            ax.bar(tab.index, tab[t], bottom=bottom, color=pal[t], label=t)
            bottom += tab[t].to_numpy()
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        ax.set_ylabel("fraction")
    fig.tight_layout()
    return fig


def plot_confusion(cmp: Dict[str, object], figsize=None):
    """Row-fraction confusion table from compare_labels."""
    import matplotlib.pyplot as plt

    C = cmp["confusion"]
    fig, ax = plt.subplots(figsize=figsize or (0.5 * C.shape[1] + 3, 0.4 * C.shape[0] + 1.5))
    im = ax.imshow(C.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(C.shape[1]), C.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(C.shape[0]), C.index, fontsize=8)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C.iat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_xlabel("this pipeline")
    ax.set_ylabel("given label")
    ax.set_title(f"ARI {cmp['ari']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig
