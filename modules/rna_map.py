"""
rna_map.py
==========

Module that maps the expression of scRNA-seq cells prepared with rna_prep onto CODEX cells (position + markers + type) 'per type'
(for Google Colab, replaces rna_link.py).

Inputs
----
* CODEX : codex_types output (obs['cell_type'], layers['input'], obs['spill'])
* scRNA : rna_prep output (layers['lognorm'], obs['codex_type'] — same type names as CODEX, var['highly_variable'])

Design
----
scRNA cells are dissociated and have no position. For each CODEX cell, the k scRNA cells 'most similar in the protein panel
within the same type' are found and their expression is attached as a weighted mean.

1. Linked features : PROTEIN_TO_GENE (protein → gene, summed if several genes) → linked_features
2. Mapping plan    : mapping_plan — per type, CODEX cell count, scRNA cell count, whether mapped
               types absent or rare in scRNA are not mapped (falling back to lineage only with allow_lineage=True)
3. Mapping         : map_cells(method='knn') — kNN within type in a space where linked features are rank-normalised separately in each dataset
               method='scmodal' — kNN within the same type in the scMODAL shared embedding (scRNA uses linked features + HVGs)
               result CellMap: sparse weights W (CODEX cells × scRNA cells) → impute(genes) / transfer(labels)
4. Validation      : holdout_check — leave some linked markers out, map, and see how well mapped RNA predicts the held-out proteins
               (compared with a random mapping that shuffles matches within the same type)

Limitation: differences between cells in the mapped RNA reflect reality only as far as the panel proteins explain them.
      Interpret per-niche RNA comparisons only for types where holdout_check is clearly better than random.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["PROTEIN_TO_GENE", "check_rna", "linked_features", "mapping_plan", "map_cells", "CellMap", "holdout_check",
           "plot_holdout"]

# ---------------------------------------------------------------------------
# protein (CODEX marker name) → gene symbols. Expression is summed if several (complexes·isoforms)
# ---------------------------------------------------------------------------
PROTEIN_TO_GENE: Dict[str, List[str]] = {
    "CD3": ["CD3E", "CD3D", "CD3G"], "CD4": ["CD4"], "CD8": ["CD8A", "CD8B"], "FoxP3": ["FOXP3"],
    "CD25": ["IL2RA"], "PD1": ["PDCD1"], "PDL1": ["CD274"], "CD20": ["MS4A1"], "CD21": ["CR2"],
    "CD38": ["CD38"], "CD138": ["SDC1"], "CD45": ["PTPRC"], "HLADR": ["HLA-DRA", "HLA-DRB1"],
    "BCL2": ["BCL2"], "CD56": ["NCAM1"], "CD57": ["B3GAT1"], "CD68": ["CD68"], "CD163": ["CD163"],
    "CD206": ["MRC1"], "CD11b": ["ITGAM"], "CD11c": ["ITGAX"], "CD15": ["FUT4"], "MMP9": ["MMP9"],
    "Arginase1": ["ARG1"], "IDO": ["IDO1"], "CD36": ["CD36"], "COX2": ["PTGS2"], "Annexin A1": ["ANXA1"],
    "CD31": ["PECAM1"], "CD34": ["CD34"], "Podoplanin": ["PDPN"], "Vimentin": ["VIM"], "CD73": ["NT5E"],
    "aSMA": ["ACTA2"], "Nestin": ["NES"], "Cytokeratin": ["KRT8", "KRT18", "KRT19", "KRT5", "KRT14"],
    "CK7": ["KRT7"], "p63": ["TP63"], "MUC1": ["MUC1"], "MUC5AC": ["MUC5AC"], "MUC6": ["MUC6"],
    "Bcatenin": ["CTNNB1"], "aDefensin5": ["DEFA5"], "CHGA": ["CHGA"],
}

def _lineage_of() -> Dict[str, str]:
    try:
        import codex_types as cty
        return {t: d.get("lineage", t) for t, d in cty.MARKER_SETS.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 1–2. Input checks, linked features
# ---------------------------------------------------------------------------
def check_rna(rna, rna_key: str = "codex_type", need_hvg: bool = False) -> None:
    """Check that this is rna_prep output (layers['lognorm'], obs[rna_key], var['highly_variable'] for scMODAL)."""
    miss = []
    if "lognorm" not in rna.layers:
        miss.append("layers['lognorm'] (rna_prep.normalize)")
    if rna_key not in rna.obs:
        miss.append(f"obs['{rna_key}'] (rna_prep.annotate)")
    if need_hvg and "highly_variable" not in rna.var:
        miss.append("var['highly_variable'] (rna_prep.hvg)")
    if miss:
        raise ValueError("scRNA was not prepared with rna_prep. Missing: " + ", ".join(miss))


def linked_features(codex_markers: Sequence[str], rna_genes: Sequence[str],
                    table: Optional[Dict[str, List[str]]] = None, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Per marker, the corresponding genes and those present in scRNA. used = at least one corresponding gene and not excluded."""
    table = table or PROTEIN_TO_GENE
    genes = set(map(str, rna_genes))
    rows = []
    for m in map(str, codex_markers):
        g = table.get(m, [])
        found = [x for x in g if x in genes]
        rows.append(dict(marker=m, genes=",".join(g), found=",".join(found), used=bool(found) and m not in exclude))
    return pd.DataFrame(rows).set_index("marker")


def _rna_feature_matrix(rna, links: pd.DataFrame) -> np.ndarray:
    """scRNA cells × linked markers: sum of lognorm of corresponding genes (dense float32)."""
    L = rna.layers["lognorm"]
    vi = {g: i for i, g in enumerate(map(str, rna.var_names))}
    used = links.index[links.used]
    out = np.zeros((rna.n_obs, len(used)), np.float32)
    for j, m in enumerate(used):
        cols = [vi[g] for g in links.loc[m, "found"].split(",")]
        out[:, j] = np.asarray(L[:, cols].sum(1)).ravel()
    return out


def _codex_feature_matrix(codex, links: pd.DataFrame, layer: str = "input") -> np.ndarray:
    X = codex.layers[layer] if layer in codex.layers else codex.X
    names = list(map(str, codex.var_names))
    return np.asarray(X, dtype=np.float32)[:, [names.index(m) for m in links.index[links.used]]]


# ---------------------------------------------------------------------------
# 3. Mapping plan
# ---------------------------------------------------------------------------
def mapping_plan(codex, rna, codex_key: str = "cell_type", rna_key: str = "codex_type", min_rna: int = 30,
                 allow_lineage: bool = False, lineage: Optional[Dict[str, str]] = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Which scRNA cell group each CODEX type maps to.

    unit      : 'type'    — ≥ min_rna scRNA cells of the same type
                'lineage' — (only with allow_lineage=True) no such type, but ≥ min_rna scRNA cells of the same lineage
                'none'    — not mapped (Unknown, 'A|B?', types absent or rare in scRNA)
    allow_lineage is off by default: e.g. if scRNA has no neutrophils and we fall back to the lineage (Myeloid), CODEX granulocytes
    would receive monocyte expression. Turn it on only when types within a lineage are close to each other (e.g. Tfh → T).
    """
    check_rna(rna, rna_key)
    lineage = lineage or _lineage_of()
    ct = codex.obs[codex_key].astype(str).value_counts()
    rt = rna.obs[rna_key].astype(str).value_counts()
    r_lin = rna.obs[rna_key].astype(str).map(lambda x: lineage.get(x, x)).value_counts()
    rows = []
    for t, n in ct.items():
        lin = lineage.get(t)
        if t == "Unknown" or t.endswith("?"):                     # cells without a settled type are not mapped
            unit, target, n_r = "none", "", 0
        elif t in rt and rt[t] >= min_rna:
            unit, target, n_r = "type", t, rt[t]
        elif allow_lineage and lin and lin in r_lin and r_lin[lin] >= min_rna:
            unit, target, n_r = "lineage", lin, r_lin[lin]
        else:
            unit, target, n_r = "none", "", int(rt.get(t, 0))
        rows.append(dict(codex_type=t, lineage=lin or "", n_codex=int(n), unit=unit, rna_group=target, n_rna=int(n_r)))
    plan = pd.DataFrame(rows).set_index("codex_type")
    extra = sorted(t for t in set(rt.index) - set(ct.index) if t != "Unknown" and not t.endswith("?"))
    plan.attrs["rna_only_types"] = extra
    if verbose:
        frac = plan.groupby("unit")["n_codex"].sum() / plan.n_codex.sum()
        print("[map] mapping units of CODEX cells: " + ", ".join(f"{k} {v:.0%}" for k, v in frac.items()))
        if extra:
            print(f"[map] types only in scRNA (no CODEX counterpart): {extra}")
    return plan


# ---------------------------------------------------------------------------
# 4. Mapping
# ---------------------------------------------------------------------------
def _rank_norm(X: np.ndarray) -> np.ndarray:
    """Per column, ranks → normal scores (removes scale·distribution differences between the two measurements). Ties get average ranks."""
    from scipy.stats import norm, rankdata

    n = len(X)
    out = np.empty_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        r = rankdata(X[:, j])
        out[:, j] = norm.ppf((r - 0.5) / n) if np.ptp(X[:, j]) > 0 else 0.0
    return out


@dataclass
class CellMap:
    """Sparse weights CODEX cells × scRNA cells (rows sum to 1, rows of unmapped cells are empty)."""
    W: sp.csr_matrix
    codex_names: np.ndarray
    rna_names: np.ndarray
    group: pd.Series                       # mapping unit name per CODEX cell ('' = not mapped)
    dist: pd.Series                        # mean neighbour distance per CODEX cell (mapping confidence: smaller = more similar)
    stats: pd.DataFrame                    # per-group summary
    params: Dict = field(default_factory=dict)

    @property
    def mapped(self) -> np.ndarray:
        return np.asarray(self.W.getnnz(axis=1) > 0)

    def impute(self, rna, genes: Sequence[str], layer: str = "lognorm") -> pd.DataFrame:
        """CODEX cells × genes (weighted mean lognorm of neighbouring scRNA cells). NaN for unmapped cells."""
        vi = {g: i for i, g in enumerate(map(str, rna.var_names))}
        miss = [g for g in genes if g not in vi]
        genes = [g for g in genes if g in vi]
        if miss:
            print(f"[map] {len(miss)} genes not in scRNA excluded: {miss[:10]}")
        X = rna.layers[layer] if layer in rna.layers else rna.X
        Y = self.W @ sp.csr_matrix(X[:, [vi[g] for g in genes]])
        out = pd.DataFrame(np.asarray(Y.todense(), dtype=np.float32), index=self.codex_names, columns=genes)
        out.loc[~self.mapped] = np.nan
        return out

    def transfer(self, labels: pd.Series) -> pd.DataFrame:
        """scRNA labels (e.g. fine subtypes) → per-CODEX-cell label probabilities (columns = labels)."""
        lab = pd.Categorical(labels.reindex(self.rna_names).astype(str))
        onehot = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab.codes)),
                               shape=(len(lab), len(lab.categories)))
        P = np.asarray((self.W @ onehot).todense())
        return pd.DataFrame(P, index=self.codex_names, columns=lab.categories)

    def save(self, out_dir: str, prefix: str = "") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        p = {k: os.path.join(out_dir, f"{prefix}{k}") for k in ("cellmap_W.npz", "cellmap_cells.csv.gz",
                                                                 "cellmap_rna_cells.csv.gz", "cellmap_stats.csv",
                                                                 "cellmap_params.json")}
        sp.save_npz(p["cellmap_W.npz"], self.W)
        pd.DataFrame(dict(group=self.group, dist=self.dist), index=pd.Index(self.codex_names, name="cell")) \
            .to_csv(p["cellmap_cells.csv.gz"])
        pd.Series(self.rna_names, name="rna_cell").to_csv(p["cellmap_rna_cells.csv.gz"], index=False)
        self.stats.to_csv(p["cellmap_stats.csv"])
        with open(p["cellmap_params.json"], "w") as f:
            json.dump(self.params, f, indent=1, default=str)
        return p

    @classmethod
    def load(cls, out_dir: str, prefix: str = "") -> "CellMap":
        j = lambda k: os.path.join(out_dir, f"{prefix}{k}")
        cells = pd.read_csv(j("cellmap_cells.csv.gz"), index_col=0, keep_default_na=False)
        with open(j("cellmap_params.json")) as f:
            params = json.load(f)
        return cls(W=sp.load_npz(j("cellmap_W.npz")).tocsr(), codex_names=cells.index.to_numpy(),
                   rna_names=pd.read_csv(j("cellmap_rna_cells.csv.gz"))["rna_cell"].astype(str).to_numpy(),
                   group=cells["group"].astype(str), dist=pd.to_numeric(cells["dist"], errors="coerce"),
                   stats=pd.read_csv(j("cellmap_stats.csv"), index_col=0), params=params)


def _scmodal_latent(codex, rna, links, steps: int, seed: int, model_path: str, verbose: bool):
    """scMODAL shared embedding. Install: git clone the repository and add it to sys.path (no conda environment needed).

    A = scRNA : [linked features (sum of corresponding gene lognorm) | remaining HVGs]  — standardised per column, dense
    B = CODEX : [linked markers | remaining markers]                                   — standardised per column
    shared_gene_num = number of linked features (scMODAL finds MNN matches between the first n columns)
    """
    import anndata as ad
    import scmodal

    check_rna(rna, need_hvg=True)
    used = list(links.index[links.used])
    linked_genes = {g for m in used for g in links.loc[m, "found"].split(",")}
    R_link = _rna_feature_matrix(rna, links)
    hv = [i for i, g in enumerate(map(str, rna.var_names))
          if rna.var["highly_variable"].iat[i] and g not in linked_genes]
    R_hvg = rna.layers["lognorm"][:, hv].toarray().astype(np.float32)
    names = list(map(str, codex.var_names))
    other = [m for m in names if m not in used]
    Xc = np.asarray(codex.layers["input"] if "input" in codex.layers else codex.X, dtype=np.float32)
    C = np.hstack([_codex_feature_matrix(codex, links), Xc[:, [names.index(m) for m in other]]])
    zs = lambda M: np.clip((M - M.mean(0)) / (M.std(0) + 1e-6), -10, 10).astype(np.float32)
    A = ad.AnnData(np.hstack([zs(R_link), zs(R_hvg)]))
    B = ad.AnnData(zs(C))
    if verbose:
        print(f"[map] scMODAL input: scRNA {A.shape} (linked {len(used)} + HVG {len(hv)}), CODEX {B.shape}")
    model = scmodal.model.Model(training_steps=steps, seed=seed, model_path=model_path)
    model.preprocess(A, B, shared_gene_num=len(used))
    model.train()
    model.eval()
    lat = np.asarray(model.latent, dtype=np.float32)
    return lat[: rna.n_obs], lat[rna.n_obs:]


def map_cells(codex, rna, plan: pd.DataFrame, links: pd.DataFrame, codex_key: str = "cell_type",
              rna_key: str = "codex_type", k: int = 20, method: str = "knn", exclude_codex: Optional[str] = "spill",
              lineage: Optional[Dict[str, str]] = None, scmodal_steps: int = 10000,
              scmodal_dir: str = "scmodal_models", seed: int = 0, verbose: bool = True) -> CellMap:
    """Per mapping_plan unit (type / lineage), CODEX cell → the k most similar scRNA cells.

    method
      'knn'     : within the unit, linked features rank-normalised separately in each dataset → Euclidean kNN
      'scmodal' : train scMODAL once on all cells for a shared embedding, then per-unit kNN within it
    weights   : exp(−d² / 2σ²), σ = median distance to the k-th neighbour (per unit) → rows sum to 1
    exclude_codex : cells with 1 in this 0/1 obs column (default 'spill' — co-positive exclusive markers) are not mapped
    """
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time()
    check_rna(rna, rna_key)
    lineage = lineage or _lineage_of()
    used = links.index[links.used]
    if len(used) < 5:
        raise ValueError(f"Only {len(used)} linked features (≥ 5 recommended). Check the linked_features output")
    if method == "knn":
        C_all = _codex_feature_matrix(codex, links)
        R_all = _rna_feature_matrix(rna, links)
    elif method == "scmodal":
        R_all, C_all = _scmodal_latent(codex, rna, links, scmodal_steps, seed, scmodal_dir, verbose)
    else:
        raise ValueError("method must be 'knn' or 'scmodal'")
    ctype = codex.obs[codex_key].astype(str).to_numpy()
    rtype = rna.obs[rna_key].astype(str).to_numpy()
    rlin = np.array([lineage.get(x, x) for x in rtype])
    skip = (codex.obs[exclude_codex].to_numpy() > 0) if exclude_codex and exclude_codex in codex.obs else \
        np.zeros(codex.n_obs, bool)
    rows, cols, vals = [], [], []
    group = np.full(codex.n_obs, "", dtype=object)
    dist = np.full(codex.n_obs, np.nan, np.float32)
    stats = []
    for t, r in plan.iterrows():
        if r.unit == "none":
            continue
        ci = np.where((ctype == t) & ~skip)[0]
        ri = np.where((rtype == r.rna_group) if r.unit == "type" else (rlin == r.rna_group))[0]
        if not len(ci) or not len(ri):
            continue
        C, R = C_all[ci], R_all[ri]
        if method == "knn":
            C, R = _rank_norm(C), _rank_norm(R)
        kk = min(k, len(ri))
        d, j = NearestNeighbors(n_neighbors=kk).fit(R).kneighbors(C)
        sig = np.median(d[:, -1]) + 1e-6
        w = np.exp(-(d ** 2) / (2 * sig ** 2))
        w /= w.sum(1, keepdims=True)
        rows.append(np.repeat(ci, kk))
        cols.append(ri[j].ravel())
        vals.append(w.ravel())
        group[ci] = f"{r.unit}:{r.rna_group}"
        dist[ci] = d.mean(1)
        used_r = np.unique(ri[j])
        stats.append(dict(codex_type=t, unit=r.unit, rna_group=r.rna_group, n_codex=len(ci), n_rna=len(ri),
                          rna_used=round(len(used_r) / len(ri), 3), mean_dist=round(float(d.mean()), 3),
                          eff_neighbors=round(float((1 / (w ** 2).sum(1)).mean()), 1)))
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(codex.n_obs, rna.n_obs), dtype=np.float32) if rows else \
        sp.csr_matrix((codex.n_obs, rna.n_obs), dtype=np.float32)
    st = pd.DataFrame(stats).set_index("codex_type") if stats else pd.DataFrame()
    cm = CellMap(W=W, codex_names=codex.obs_names.to_numpy(), rna_names=rna.obs_names.to_numpy().astype(str),
                 group=pd.Series(group, index=codex.obs_names), dist=pd.Series(dist, index=codex.obs_names),
                 stats=st, params=dict(method=method, k=k, codex_key=codex_key, rna_key=rna_key,
                                       features=list(used), exclude_codex=exclude_codex, seed=seed))
    if verbose:
        print(f"[map] {method}: {cm.mapped.sum():,} / {codex.n_obs:,} CODEX cells mapped ({cm.mapped.mean():.0%}), "
              f"{len(used)} linked features, k = {k} | {time.time() - t0:.0f}s")
    return cm


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------
def holdout_check(codex, rna, plan: pd.DataFrame, links: pd.DataFrame, holdout: Optional[Sequence[str]] = None,
                  n_folds: int = 4, n_null: int = 5, seed: int = 0, verbose: bool = True, **map_kw) -> pd.DataFrame:
    """Leave out linked markers by fold and map → Spearman correlation of the mapped RNA of held-out markers with CODEX protein (within types).

    null : mapping rows of CODEX cells shuffled randomly within the same type (a mapping that only matches type means).
           If r − null is near 0, 'within-type cell differences' of that marker are not carried over by the mapping.
    returns: marker × type table (r, null, gain, n) — only cells with ≥ 50 cells of the same type
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    used = list(links.index[links.used])
    holdout = list(holdout) if holdout else used
    folds = np.array_split(rng.permutation(holdout), max(1, min(n_folds, len(holdout))))
    ctype = codex.obs[map_kw.get("codex_key", "cell_type")].astype(str).to_numpy()
    names = list(map(str, codex.var_names))
    Xc = np.asarray(codex.layers["input"] if "input" in codex.layers else codex.X, dtype=np.float32)
    rows = []
    for fold in folds:
        lk = links.copy()
        lk.loc[list(fold), "used"] = False
        cm = map_cells(codex, rna, plan, lk, verbose=False, seed=seed, **map_kw)
        gl = {m: links.loc[m, "found"].split(",") for m in fold}
        imp = cm.impute(rna, sorted({g for v in gl.values() for g in v}))
        for m in fold:
            pred = imp[gl[m]].sum(1).to_numpy()
            prot = Xc[:, names.index(m)]
            for t in np.unique(ctype[cm.mapped]):
                ii = np.where((ctype == t) & cm.mapped)[0]
                if len(ii) < 50 or np.ptp(prot[ii]) == 0 or np.ptp(pred[ii]) == 0:
                    continue
                r = spearmanr(pred[ii], prot[ii])[0]
                nul = np.nanmean([spearmanr(pred[rng.permutation(ii)], prot[ii])[0] for _ in range(n_null)])
                rows.append(dict(marker=m, codex_type=t, n=len(ii), r=r, null=nul, gain=r - nul))
    res = pd.DataFrame(rows)
    if verbose and len(res):
        by_m = res.groupby("marker")["gain"].median().sort_values(ascending=False)
        print(f"[check] {res.marker.nunique()} held-out markers, median within-type correlation r = {res.r.median():.3f} "
              f"(random {res.null.median():.3f}) | cells with gain > 0.1 {(res.gain > 0.1).mean():.0%}")
        print("        top gain per marker: " + ", ".join(f"{k} {v:.2f}" for k, v in by_m.head(8).items()))
    return res


def plot_holdout(res: pd.DataFrame, figsize=None):
    """holdout_check result: marker × type gain heatmap."""
    import matplotlib.pyplot as plt

    M = res.pivot_table(index="marker", columns="codex_type", values="gain")
    M = M.loc[M.median(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=figsize or (0.45 * M.shape[1] + 3, 0.3 * M.shape[0] + 1.5))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="r − shuffled r (within type)")
    ax.set_title("held-out marker: gain over type-mean mapping")
    fig.tight_layout()
    return fig
