"""
maxfuse_map.py
==============

Matches scRNA cells ↔ CODEX cells at roughly 1:1 with MaxFuse (Chen et al., Nat Biotechnol 2023),
① transfers scRNA labels to CODEX cells (label transfer) and ② attaches a tissue position to each scRNA cell through its matches (for Google Colab).

Inputs (outputs of rna_prep / codex_types)
----------------------------------
* scRNA : layers['lognorm'], var['highly_variable'], obs['codex_type'] (or another label column)
* CODEX : layers['input'] (marker values), obsm['spatial'] (x, y), obs['cell_type'] (for comparison)

The 4 MaxFuse inputs (same setup as the MaxFuse tonsil tutorial)
---------------------------------------------------
* shared_arr1 / shared_arr2 : linked features. One column per protein ↔ gene pair (several columns if one protein ↔ several genes),
                              scRNA lognorm / CODEX values standardised per column. Columns with low variation in either dataset are dropped
* active_arr1               : standardised scRNA HVG lognorm (structure between cells = used for graph smoothing)
* active_arr2               : all CODEX marker values
Steps: batching → per-dataset kNN graphs → initial matches (shared features) → refine matches with CCA → drop bad matches → propagate to the remaining cells

Result MaxFuseResult
------------------
* matching       : list of (rna, codex, distance) matches. order=(2, 1) — every CODEX cell not filtered out has at least one match
* transfer_labels: per CODEX cell, distance-weighted vote of matched scRNA labels → obs['mf_label'], obs['mf_conf'], agreement table with existing types
* rna_locations  : per scRNA cell, positions of its matched CODEX cells (distance-weighted centre, position of the closest match, spread, number of matches)
* pairs          : per match, CODEX position · both labels · whether they agree (long table)
* to_cellmap     : converts to rna_map.CellMap → impute(genes) attaches expression to CODEX positions

Notes
----
* MaxFuse's graph step calls scanpy.pp.neighbors, which breaks with recent scipy·anndata
  (csr_array.getnnz / .A1). It is swapped for a function doing the same thing (kNN → UMAP fuzzy graph) with pynndescent + umap.
* MaxFuse's Leiden iterates until convergence and is very slow (85 % of total time). By default it is changed to 2 iterations (fast_leiden).
* One scRNA cell can be matched with several CODEX cells (tonsil: 170 k CODEX vs 37 k scRNA). The 'position' of such a cell is
  several places — if spread in rna_locations is large, look at the match list (pairs) rather than the centre coordinates.
* MaxFuse forces matches even for CODEX cells of types absent from scRNA (granulocytes·stroma·epithelium in this tonsil scRNA).
  Filter with the agreement table of transfer_labels and the consistent column of pairs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["DEFAULT_PARAMS", "patch_maxfuse", "build_inputs", "run_maxfuse", "MaxFuseResult", "plot_label_confusion",
           "plot_rna_locations"]

# values from the MaxFuse tonsil CODEX + scRNA tutorial. Except batching_scheme is 'cyclic' (tutorial·default is 'pairwise' = all batch pairs)
# — pairwise grows with the number of batch pairs and took 42 min on CPU for 30 k CODEX × 37 k scRNA
DEFAULT_PARAMS = dict(
    batches=dict(max_outward_size=8000, matching_ratio=4, metacell_size=2, batching_scheme="cyclic"),
    graphs=dict(n_neighbors1=15, n_neighbors2=15, svd_components1=40, svd_components2=15, resolution1=2,
                resolution2=2, resolution_tol=0.1),
    initial=dict(wt1=0.3, wt2=0.3, svd_components1=25, svd_components2=20),
    refine=dict(wt1=0.3, wt2=0.3, svd_components1=40, svd_components2=None, cca_components=25, n_iters=1),
    filter_pivot=0.5,
    propagate=dict(svd_components1=40, svd_components2=None, wt1=0.7, wt2=0.7),
    filter_propagated=0.3,
)


# ---------------------------------------------------------------------------
# replacing MaxFuse graph functions
# ---------------------------------------------------------------------------
def _construct_graph(arr, randomized_svd=False, svd_runs=1, svd_components=None, n_neighbors=15,
                     metric="correlation", verbose=False):
    """Same result as maxfuse.graph.construct_graph (edges of the UMAP fuzzy kNN graph) without scanpy."""
    from maxfuse import utils as mfu
    from pynndescent import NNDescent
    from umap.umap_ import fuzzy_simplicial_set

    arr = mfu.drop_zero_variability_columns(arr_lst=[arr, arr])[0]
    arr = mfu.svd_embedding(arr=arr, n_components=svd_components, randomized=randomized_svd, n_runs=svd_runs)
    arr = np.asarray(arr, dtype=np.float32)
    k = min(n_neighbors, len(arr) - 1)
    idx, dist = NNDescent(arr, n_neighbors=k, metric=metric, random_state=0, n_jobs=-1).neighbor_graph
    g, _, _ = fuzzy_simplicial_set(arr, k, np.random.RandomState(0), metric, knn_indices=idx, knn_dists=dist)
    g = sp.coo_matrix(g)
    return g.row, g.col, g.data


LEIDEN_ITERATIONS = 2


def _leiden_clustering(n, edges, resolution=1, n_runs=1, seed=None, verbose=False):
    """Same as maxfuse.graph.leiden_clustering but with Leiden iterations limited to LEIDEN_ITERATIONS (the original runs to convergence = −1).

    MaxFuse calls Leiden dozens of times per batch while bisecting the resolution to match cluster counts of the two datasets.
    In local measurements this was 85 % of total time. 2 iterations is the same as scanpy's default.
    """
    import igraph as ig
    import leidenalg

    g = ig.Graph(directed=True)
    g.add_vertices(n)
    g.add_edges(list(zip(edges[0], edges[1])))
    kw = dict(n_iterations=LEIDEN_ITERATIONS, seed=seed if n_runs == 1 else None, resolution_parameter=resolution)
    if len(edges) > 2:
        kw["weights"] = np.asarray(edges[2], dtype=np.float64)
    best, best_q = None, -np.inf
    for _ in range(n_runs):
        part = leidenalg.find_partition(graph=g, partition_type=leidenalg.RBConfigurationVertexPartition, **kw)
        if part.modularity > best_q:
            best_q, best = part.modularity, np.array(part.membership)
    return best


def patch_maxfuse(fast_leiden: bool = True):
    """Swap maxfuse.graph's construct_graph (depends on scanpy) and leiden_clustering (slow) (once after import)."""
    import maxfuse as mf

    mf.graph.construct_graph = _construct_graph
    if fast_leiden:
        mf.graph.leiden_clustering = _leiden_clustering
    return mf


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _zs(M: np.ndarray, clip: float = 10.0) -> np.ndarray:
    return np.clip((M - M.mean(0)) / (M.std(0) + 1e-6), -clip, clip).astype(np.float32)


def build_inputs(codex, rna, links: pd.DataFrame, min_std_rna: float = 0.1, min_std_codex: float = 0.1,
                 codex_layer: str = "input", verbose: bool = True) -> Dict:
    """The 4 MaxFuse inputs and the protein–gene pairs used.

    links : output of rna_map.linked_features (marker → found genes)
    Pairs with low variation (scRNA lognorm SD < min_std_rna or CODEX < min_std_codex) are dropped — near-zero genes
    only add noise to matching.
    """
    L = rna.layers["lognorm"]
    vi = {g: i for i, g in enumerate(map(str, rna.var_names))}
    names = list(map(str, codex.var_names))
    Xc = np.asarray(codex.layers[codex_layer] if codex_layer in codex.layers else codex.X, dtype=np.float32)
    pairs = [(g, m) for m in links.index[links.used] for g in links.loc[m, "found"].split(",") if g in vi]
    R = L[:, [vi[g] for g, _ in pairs]].toarray().astype(np.float32)
    C = Xc[:, [names.index(m) for _, m in pairs]]
    keep = (R.std(0) >= min_std_rna) & (C.std(0) >= min_std_codex)
    pairs = [p for p, k in zip(pairs, keep) if k]
    hv = np.where(rna.var["highly_variable"].to_numpy())[0]
    out = dict(shared1=_zs(R[:, keep]), shared2=_zs(C[:, keep]),
               active1=_zs(L[:, hv].toarray().astype(np.float32)), active2=_zs(Xc),
               pairs=pd.DataFrame(pairs, columns=["gene", "marker"]))
    if verbose:
        print(f"[maxfuse] {keep.sum()} shared feature pairs ({int((~keep).sum())} low-variation pairs excluded) | "
              f"active: scRNA {out['active1'].shape}, CODEX {out['active2'].shape}")
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_maxfuse(codex, rna, links: pd.DataFrame, params: Optional[Dict] = None, rna_key: str = "codex_type",
                codex_mask: Optional[np.ndarray] = None, fast_leiden: bool = True, seed: int = 0,
                verbose: bool = True) -> "MaxFuseResult":
    """All MaxFuse steps → MaxFuseResult. params has the same shape as DEFAULT_PARAMS (partial is fine).

    codex_mask : CODEX cells to find matches for (bool, length = number of CODEX cells). None = all.
                 e.g. only CODEX cells of types present in scRNA — prevents absent types (granulocytes etc.) being forced onto B cells
                 and saves time. But label transfer then leans on CODEX types, so use None for label validation.
    """
    mf = patch_maxfuse(fast_leiden)
    P = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_PARAMS.items()}
    for k, v in (params or {}).items():
        P[k] = {**P[k], **v} if isinstance(P.get(k), dict) else v
    t0 = time.time()
    sub = np.arange(codex.n_obs) if codex_mask is None else np.where(np.asarray(codex_mask, bool))[0]
    full_codex = codex
    codex = codex[sub] if codex_mask is not None else codex
    inp = build_inputs(codex, rna, links, verbose=verbose)
    n_shared = inp["shared1"].shape[1]
    for k in ("svd_components1", "svd_components2"):        # shared-feature SVD must be smaller than the number of features (when markers are held out)
        if P["initial"].get(k) is not None and P["initial"][k] >= n_shared:
            P["initial"][k] = max(1, n_shared - 1)
    np.random.seed(seed)
    fusor = mf.model.Fusor(shared_arr1=inp["shared1"], shared_arr2=inp["shared2"], active_arr1=inp["active1"],
                           active_arr2=inp["active2"], labels1=None, labels2=None)
    fusor.split_into_batches(**P["batches"], seed=seed, verbose=verbose)
    fusor.construct_graphs(**P["graphs"], leiden_seed=seed, verbose=verbose)
    fusor.find_initial_pivots(**P["initial"], verbose=verbose)
    fusor.refine_pivots(**P["refine"], verbose=verbose)
    fusor.filter_bad_matches(target="pivot", filter_prop=P["filter_pivot"], verbose=verbose)
    pivot = fusor.get_matching(order=(2, 1), target="pivot")
    fusor.propagate(**P["propagate"], verbose=verbose)
    fusor.filter_bad_matches(target="propagated", filter_prop=P["filter_propagated"], verbose=verbose)
    full = fusor.get_matching(order=(2, 1), target="full_data")
    m = pd.DataFrame(dict(rna=np.asarray(full[0], np.int64), codex=np.asarray(full[1], np.int64),
                          dist=np.asarray(full[2], np.float32)))
    piv = set(zip(np.asarray(pivot[0]).tolist(), np.asarray(pivot[1]).tolist()))
    m["pivot"] = [(r, c) in piv for r, c in zip(m.rna, m.codex)]
    m["codex"] = sub[m["codex"].to_numpy()]                    # subset index → full CODEX index
    codex = full_codex
    res = MaxFuseResult(matching=m, codex_names=codex.obs_names.to_numpy().astype(str),
                        rna_names=rna.obs_names.to_numpy().astype(str), pairs_used=inp["pairs"],
                        params=dict(P, rna_key=rna_key, seed=seed, fast_leiden=fast_leiden,
                                    n_codex_used=int(len(sub)), time_s=round(time.time() - t0)))
    if verbose:
        nc = m.codex.nunique()
        print(f"[maxfuse] {len(m):,} matches | {nc:,} / {codex.n_obs:,} CODEX cells ({nc / codex.n_obs:.0%}) matched, "
              f"{m.rna.nunique():,} / {rna.n_obs:,} scRNA cells used | {time.time() - t0:.0f}s")
    return res


@dataclass
class MaxFuseResult:
    matching: pd.DataFrame                 # rna, codex (integer positions), dist (smaller is better), pivot
    codex_names: np.ndarray
    rna_names: np.ndarray
    pairs_used: pd.DataFrame
    params: Dict = field(default_factory=dict)

    # --- weights: among matches of the same CODEX cell, larger for smaller distance ---
    def _weights(self) -> np.ndarray:
        d = self.matching["dist"].to_numpy(np.float64)
        s = np.median(d) + 1e-9
        return np.exp(-d / s)

    def transfer_labels(self, codex, rna, rna_key: Optional[str] = None, codex_key: str = "cell_type",
                        lineage: bool = True, verbose: bool = True) -> Dict:
        """Per CODEX cell, distance-weighted vote of matched scRNA labels → codex.obs['mf_label'], ['mf_conf'] ('unmatched' if no match).

        returns: agree (type-level agreement among matched cells), agree_lineage, confusion (rows = CODEX types, columns = transferred labels)
        """
        rna_key = rna_key or self.params.get("rna_key", "codex_type")
        m = self.matching
        lab_r = rna.obs[rna_key].astype(str).to_numpy()[m.rna.to_numpy()]
        df = pd.DataFrame(dict(codex=m.codex.to_numpy(), lab=lab_r, w=self._weights()))
        votes = df.groupby(["codex", "lab"])["w"].sum().reset_index()
        tot = votes.groupby("codex")["w"].transform("sum")
        votes["share"] = votes["w"] / tot
        best = votes.sort_values("share", ascending=False).drop_duplicates("codex").set_index("codex")
        lab = np.full(codex.n_obs, "unmatched", dtype=object)
        conf = np.zeros(codex.n_obs, np.float32)
        lab[best.index.to_numpy()] = best["lab"].to_numpy()
        conf[best.index.to_numpy()] = best["share"].to_numpy()
        codex.obs["mf_label"] = pd.Categorical(lab)
        codex.obs["mf_conf"] = conf
        out = {}
        if codex_key in codex.obs:
            ok = lab != "unmatched"
            orig = codex.obs[codex_key].astype(str).to_numpy()
            out["agree"] = float((orig[ok] == lab[ok]).mean())
            if lineage:
                try:
                    import codex_types as cty
                    lin = {t: d.get("lineage", t) for t, d in cty.MARKER_SETS.items()}
                    lo = np.array([lin.get(x, x) for x in orig])
                    ln = np.array([lin.get(x, x) for x in lab])
                    out["agree_lineage"] = float((lo[ok] == ln[ok]).mean())
                    codex.obs["mf_lineage"] = pd.Categorical(ln)
                except ImportError:
                    pass
            conf_m = pd.crosstab(pd.Series(orig, name=codex_key), pd.Series(lab, name="mf_label"))
            out["confusion"] = conf_m.div(conf_m.sum(1), axis=0).round(3)
            out["counts"] = conf_m
            if verbose:
                print(f"[transfer] labels for {ok.sum():,} CODEX cells | agreement with '{codex_key}': type {out['agree']:.1%}"
                      + (f", lineage {out['agree_lineage']:.1%}" if "agree_lineage" in out else ""))
        return out

    def pairs(self, codex, rna, rna_key: Optional[str] = None, codex_key: str = "cell_type") -> pd.DataFrame:
        """One row per match: CODEX cell · position · type, scRNA cell · label, distance, consistent (both types equal)."""
        rna_key = rna_key or self.params.get("rna_key", "codex_type")
        m = self.matching
        xy = np.asarray(codex.obsm["spatial"])[m.codex.to_numpy()]
        out = pd.DataFrame(dict(
            codex_cell=self.codex_names[m.codex.to_numpy()], rna_cell=self.rna_names[m.rna.to_numpy()],
            x=xy[:, 0], y=xy[:, 1], dist=m["dist"].to_numpy(), pivot=m["pivot"].to_numpy(),
            codex_type=codex.obs[codex_key].astype(str).to_numpy()[m.codex.to_numpy()] if codex_key in codex.obs else "",
            rna_type=rna.obs[rna_key].astype(str).to_numpy()[m.rna.to_numpy()]))
        out["consistent"] = out["codex_type"] == out["rna_type"]
        return out

    def rna_locations(self, codex, rna, consistent_only: bool = False, codex_key: str = "cell_type",
                      rna_key: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
        """Position per scRNA cell: distance-weighted centre of matched CODEX cells (x, y), position of the closest match (best_x, best_y),
        spread (weighted mean distance from the centre to the matches, in coordinate units), number of matches, modal CODEX type of the matches.

        consistent_only=True uses only matches whose two types agree (so mismatched matches do not drag the position).
        scRNA cells without matches are NaN. Results also go into 'loc_x', 'loc_y', 'loc_spread', 'loc_n' of rna.obs.
        """
        p = self.pairs(codex, rna, rna_key, codex_key)
        p["w"] = self._weights()
        if consistent_only:
            p = p[p["consistent"]]
        g = p.groupby("rna_cell")
        sw = g["w"].sum()
        cx = (p["x"] * p["w"]).groupby(p["rna_cell"]).sum() / sw
        cy = (p["y"] * p["w"]).groupby(p["rna_cell"]).sum() / sw
        p = p.join(pd.DataFrame(dict(cx=cx, cy=cy)), on="rna_cell")
        p["r"] = np.hypot(p["x"] - p["cx"], p["y"] - p["cy"])
        spread = (p["r"] * p["w"]).groupby(p["rna_cell"]).sum() / sw
        best = p.sort_values("dist").drop_duplicates("rna_cell").set_index("rna_cell")
        loc = pd.DataFrame(dict(x=cx, y=cy, best_x=best["x"], best_y=best["y"], spread=spread, n_matches=g.size(),
                                codex_type_mode=g["codex_type"].agg(lambda s: s.value_counts().index[0]),
                                consistent_frac=g["consistent"].mean()))
        loc = loc.reindex(self.rna_names)
        loc.index.name = "rna_cell"
        for c_new, c in (("loc_x", "x"), ("loc_y", "y"), ("loc_spread", "spread"), ("loc_n", "n_matches")):
            rna.obs[c_new] = loc[c].to_numpy()
        if verbose:
            has = loc["n_matches"].notna()
            print(f"[location] positions for {has.sum():,} / {len(loc):,} scRNA cells | median matches "
                  f"{loc.loc[has, 'n_matches'].median():.0f}, median spread {loc.loc[has, 'spread'].median():.0f} "
                  f"(coordinate units{', same-type matches only' if consistent_only else ''})")
        return loc

    def to_cellmap(self, codex, rna, consistent_only: bool = False, codex_key: str = "cell_type",
                   rna_key: Optional[str] = None):
        """rna_map.CellMap (rows = CODEX cells, distance weights summing to 1 per row) → impute(genes) gives expression at CODEX positions."""
        import rna_map as rm

        m = self.matching.copy()
        m["w"] = self._weights()
        if consistent_only:
            p = self.pairs(codex, rna, rna_key, codex_key)
            m = m[p["consistent"].to_numpy()]
        W = sp.csr_matrix((m["w"].to_numpy(np.float32), (m.codex.to_numpy(), m.rna.to_numpy())),
                          shape=(codex.n_obs, rna.n_obs))
        rs = np.asarray(W.sum(1)).ravel()
        rs[rs == 0] = 1
        W = sp.diags(1 / rs) @ W
        mapped = np.asarray(W.getnnz(axis=1) > 0)
        d = m.groupby("codex")["dist"].mean()
        dist = pd.Series(np.nan, index=codex.obs_names)
        dist.iloc[d.index.to_numpy()] = d.to_numpy()
        grp = pd.Series(np.where(mapped, "maxfuse", ""), index=codex.obs_names)
        return rm.CellMap(W=sp.csr_matrix(W, dtype=np.float32), codex_names=self.codex_names,
                          rna_names=self.rna_names, group=grp, dist=dist,
                          stats=pd.DataFrame(dict(n_codex=[int(mapped.sum())], n_rna=[int(m.rna.nunique())]),
                                             index=["maxfuse"]),
                          params=dict(method="maxfuse", consistent_only=consistent_only))

    def save(self, out_dir: str, prefix: str = "") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        p = {k: os.path.join(out_dir, f"{prefix}{k}") for k in ("maxfuse_matching.csv.gz", "maxfuse_cells.npz",
                                                                 "maxfuse_pairs_used.csv", "maxfuse_params.json")}
        self.matching.to_csv(p["maxfuse_matching.csv.gz"], index=False)
        np.savez_compressed(p["maxfuse_cells.npz"], codex=self.codex_names, rna=self.rna_names)
        self.pairs_used.to_csv(p["maxfuse_pairs_used.csv"], index=False)
        with open(p["maxfuse_params.json"], "w") as f:
            json.dump(self.params, f, indent=1, default=str)
        return p

    @classmethod
    def load(cls, out_dir: str, prefix: str = "") -> "MaxFuseResult":
        j = lambda k: os.path.join(out_dir, f"{prefix}{k}")
        z = np.load(j("maxfuse_cells.npz"), allow_pickle=False)
        with open(j("maxfuse_params.json")) as f:
            params = json.load(f)
        return cls(matching=pd.read_csv(j("maxfuse_matching.csv.gz")), codex_names=z["codex"], rna_names=z["rna"],
                   pairs_used=pd.read_csv(j("maxfuse_pairs_used.csv")), params=params)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_label_confusion(out: Dict, figsize=None):
    """Row-fraction confusion table of transfer_labels (rows = CODEX types, columns = scRNA labels transferred by MaxFuse)."""
    import matplotlib.pyplot as plt

    C = out["confusion"]
    fig, ax = plt.subplots(figsize=figsize or (0.45 * C.shape[1] + 3, 0.35 * C.shape[0] + 1.5))
    im = ax.imshow(C.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(C.shape[1]), C.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(C.shape[0]), C.index, fontsize=8)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C.iat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_xlabel("label transferred from scRNA (MaxFuse)")
    ax.set_ylabel("CODEX cell type")
    ax.set_title(f"agreement {out.get('agree', float('nan')):.1%}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def plot_rna_locations(codex, loc: pd.DataFrame, rna, color: str = "codex_type", use: str = "best",
                       max_points: int = 100_000, s: float = 2.0, figsize=(8, 8), seed: int = 0):
    """Plot scRNA cells at their attached positions (grey = all CODEX cells). use='best' closest-match position / 'centroid' centre."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    xy = np.asarray(codex.obsm["spatial"])
    ii = rng.choice(len(xy), min(max_points, len(xy)), replace=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[ii, 0], xy[ii, 1], s=0.2, c=[(0.85, 0.85, 0.85)], linewidths=0, rasterized=True)
    cx, cy = ("best_x", "best_y") if use == "best" else ("x", "y")
    ok = loc[cx].notna().to_numpy()
    lab = rna.obs[color].astype(str).reindex(loc.index).to_numpy()
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(sorted(set(lab[ok]))):
        m = ok & (lab == t)
        ax.scatter(loc[cx].to_numpy()[m], loc[cy].to_numpy()[m], s=s, color=cmap(i % 20), label=f"{t} ({m.sum():,})",
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"scRNA cells placed at matched CODEX location ({use})")
    ax.legend(markerscale=5, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return fig
