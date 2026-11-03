"""
niche_remap.py
==============

Type-mean mapping (evidence) → within-type remapping by signals from adjacent types (hypothesis) → judged with CODEX protein.

Steps
----
1. mean_impute        : per CODEX cell, mean expression of scRNA cells of the same type (evidence result, no within-type differences)
2. neighbor_composition / adjacency_enrichment
                      : per CODEX cell, type composition of its k spatial neighbours; adjacency frequency of type pairs vs a label-shuffled expectation
3. load_nichenet      : NicheNet v2 prior (ligand → target regulatory potential matrix, ligand–receptor table)
   receiver_geneset   : differential genes of finer substates of the receiver type's scRNA (genes of interest, independent of space)
   ligand_activity    : ranks ligands by how well they explain the genes of interest (NicheNet AUPR − base rate, Pearson)
4. Remapping hypothesis (one per receiver type r, sender type s pair)
     CODEX side : closeness to s = negative mean distance to the 3 nearest s cells (rank)
     scRNA side : 'nichenet'   — downstream target gene score of active ligands expressed by s (per cell)
                'similarity' — negative distance to the nearest s cells in Harmony space
                               (space without s marker genes, top doublets excluded)
5. remap              : candidates = the n_cand closest scRNA cells in protein (linked feature) space
                        weights = protein distance kernel × Π hypothesis kernels exp(−(q_codex − p_rna)² / 2σ²) → balanced cell use (Sinkhorn)
                        σ = ∞ switches the hypothesis off, giving the protein-only placement (baseline)
6. Judgement          : remap with validation proteins (signal responses: PD-L1, IDO, HLA-DR, COX2, CD25 …) removed from protein features
                        → does the spatial correlation (neighbour mean) between the gene's remapped value and actual protein rise above the baseline
                        → compared with a null that shuffles the CODEX-side hypothesis (multiple comparisons corrected with a max statistic)
   Route decision : nichenet passes → 'nichenet' | fails·not possible → try similarity → passes 'similarity' ('similarity_weak' if
               nichenet failed validation first) | all fail → 'none'
               if the sender type is absent from scRNA, neither route can be computed → 'none' (sender_absent)
7. Final remap : per-type remapping using only passing hypotheses together → CellMap (same impute / transfer as rna_map)

Interpretation: the remapped map is a 'hypothesis map'. Only (receiver type, sender type, ligand) combinations that passed judgement have data support.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["VALIDATION_PROTEINS", "NICHENET_FILES", "mean_impute", "neighbor_composition", "adjacency_enrichment",
           "load_nichenet", "receiver_geneset", "ligand_activity", "run_niche_remap", "NicheRemapResult",
           "plot_adjacency", "plot_routes", "plot_remap"]

# panel proteins that change in response to signals (for judgement). e.g. IFN-γ → PD-L1·IDO·HLA-DR, IL-2 → CD25, TNF/IL-1 → COX2
VALIDATION_PROTEINS = ["PDL1", "IDO", "HLADR", "COX2", "CD25", "PD1", "CD38", "BCL2"]

NICHENET_URL = "https://zenodo.org/records/7074291/files/{}?download=1"
NICHENET_FILES = {"ligand_target": "ligand_target_matrix_nsga2r_final.rds",
                  "lr_network": "lr_network_human_21122021.rds"}

# cell-cycle genes (removed from target scores — they move a lot regardless of signalling)
CELL_CYCLE = set("""MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 HELLS RFC2 RPA2 NASP
RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN
POLA1 CHAF1B BRIP1 E2F8 HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A
SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1
NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA
H2AFZ STMN1 TUBA1B TUBB HIST1H4C""".split())
# pairs for which the similarity route is not used even across lineages (differentiation relationships with similar transcriptomes)
RELATED_PAIRS = {frozenset(("B", "Plasma"))}


def _lineage_of() -> Dict[str, str]:
    try:
        import codex_types as cty
        return {t: d.get("lineage", t) for t, d in cty.MARKER_SETS.items()}
    except Exception:
        return {}


def _markers() -> Dict[str, List[str]]:
    try:
        import rna_prep as rp
        return rp.RNA_MARKERS
    except Exception:
        return {}


def _excluded_genes(names: Sequence[str]) -> np.ndarray:
    """Type marker · mitochondrial · ribosomal · cell-cycle genes (removed from targets·genes of interest)."""
    mk = {g for v in _markers().values() for g in v}
    s = pd.Index(names).astype(str)
    return np.asarray(s.isin(mk) | s.isin(CELL_CYCLE) | s.str.match(r"^(MT-|RP[SL]\d|HB[AB])"))


# ---------------------------------------------------------------------------
# 1. Type-mean mapping (evidence)
# ---------------------------------------------------------------------------
def mean_impute(codex, rna, plan: pd.DataFrame, genes: Sequence[str], codex_key: str = "cell_type",
                rna_key: str = "codex_type", layer: str = "lognorm") -> pd.DataFrame:
    """CODEX cell × gene: mean of scRNA cells of the same type. NaN for cells whose plan unit is not 'type'."""
    vi = {g: i for i, g in enumerate(map(str, rna.var_names))}
    genes = [g for g in genes if g in vi]
    X = rna.layers[layer] if layer in rna.layers else rna.X
    rt = rna.obs[rna_key].astype(str).to_numpy()
    ct = codex.obs[codex_key].astype(str).to_numpy()
    out = np.full((codex.n_obs, len(genes)), np.nan, np.float32)
    cols = [vi[g] for g in genes]
    for t, r in plan.iterrows():
        if r.unit != "type":
            continue
        m = np.asarray(X[rt == r.rna_group][:, cols].mean(0)).ravel()
        out[ct == t] = m
    return pd.DataFrame(out, index=codex.obs_names, columns=genes)


# ---------------------------------------------------------------------------
# 2. Spatial neighbours
# ---------------------------------------------------------------------------
def _spatial_knn(xy: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    return idx[:, 1:]


def neighbor_composition(codex, key: str = "cell_type", k: int = 10) -> pd.DataFrame:
    """CODEX cell × type: fraction of that type among k spatial neighbours (excluding self). Neighbours stored in obsm['spatial_knn']."""
    idx = _spatial_knn(np.asarray(codex.obsm["spatial"], dtype=np.float64), k)
    codex.obsm["spatial_knn"] = idx
    lab = pd.Categorical(codex.obs[key].astype(str))
    codes = lab.codes
    M = np.zeros((codex.n_obs, len(lab.categories)), np.float32)
    for j in range(k):
        M[np.arange(codex.n_obs), codes[idx[:, j]]] += 1
    return pd.DataFrame(M / k, index=codex.obs_names, columns=lab.categories)


def adjacency_enrichment(codex, key: str = "cell_type", k: int = 10, n_perm: int = 50, seed: int = 0,
                         verbose: bool = True) -> pd.DataFrame:
    """receiver type × sender type: mean fraction of the sender type among receiver cells' neighbours (observed) vs with shuffled labels.

    returns (long table): receiver, sender, obs, exp, ratio (obs/exp), z
    """
    idx = codex.obsm.get("spatial_knn")
    if idx is None or idx.shape[1] != k:
        idx = _spatial_knn(np.asarray(codex.obsm["spatial"], dtype=np.float64), k)
        codex.obsm["spatial_knn"] = idx
    lab = pd.Categorical(codex.obs[key].astype(str))
    codes, cats = lab.codes.astype(np.int64), list(lab.categories)
    nt = len(cats)
    n_r = np.bincount(codes, minlength=nt).astype(np.float64)

    def counts(c):
        C = np.bincount((c[:, None] * nt + c[idx]).ravel(), minlength=nt * nt).reshape(nt, nt)
        return C / (n_r[:, None] * k)

    obs = counts(codes)
    rng = np.random.default_rng(seed)
    P = np.stack([counts(rng.permutation(codes)) for _ in range(n_perm)])
    mu, sd = P.mean(0), P.std(0) + 1e-9
    rows = [dict(receiver=cats[a], sender=cats[b], obs=obs[a, b], exp=mu[a, b],
                 ratio=obs[a, b] / max(mu[a, b], 1e-9), z=(obs[a, b] - mu[a, b]) / sd[a, b])
            for a in range(nt) for b in range(nt)]
    res = pd.DataFrame(rows)
    if verbose:
        top = res[(res.receiver != res.sender)].sort_values("z", ascending=False).head(8)
        print("[adj] top adjacent type pairs (receiver ← sender, obs/exp): " +
              ", ".join(f"{a}←{b} {r:.1f}" for a, b, r in zip(top.receiver, top.sender, top.ratio)))
    return res


# ---------------------------------------------------------------------------
# 3. NicheNet
# ---------------------------------------------------------------------------
@dataclass
class NicheNetPrior:
    ligand_target: pd.DataFrame            # target gene × ligand (regulatory potential)
    lr: pd.DataFrame                       # columns ligand, receptor


def load_nichenet(resource_dir: str, download: bool = True, verbose: bool = True) -> NicheNetPrior:
    """NicheNet v2 (Zenodo 7074291) human resources. If missing, the rds files are downloaded to resource_dir (about 260 MB). Needs pyreadr."""
    import urllib.request

    import pyreadr

    os.makedirs(resource_dir, exist_ok=True)
    paths = {}
    for k, f in NICHENET_FILES.items():
        p = os.path.join(resource_dir, f)
        if not os.path.exists(p):
            if not download:
                raise FileNotFoundError(p)
            if verbose:
                print(f"[nichenet] downloading: {f}")
            urllib.request.urlretrieve(NICHENET_URL.format(f), p + ".part")
            os.replace(p + ".part", p)
        paths[k] = p
    lt = next(iter(pyreadr.read_r(paths["ligand_target"]).values())).astype(np.float32)
    lt.columns = lt.columns.astype(str)
    lt.index = lt.index.astype(str)
    lr = next(iter(pyreadr.read_r(paths["lr_network"]).values()))
    lr = lr.rename(columns={"from": "ligand", "to": "receptor"})[["ligand", "receptor"]].drop_duplicates()
    if verbose:
        print(f"[nichenet] {lt.shape[0]:,} targets × {lt.shape[1]:,} ligands, {len(lr):,} ligand–receptor pairs")
    return NicheNetPrior(ligand_target=lt, lr=lr)


def _type_stats(rna, idx: np.ndarray, layer: str = "lognorm") -> Tuple[np.ndarray, np.ndarray]:
    X = rna.layers[layer][idx]
    return np.asarray(X.mean(0)).ravel(), np.asarray((X > 0).mean(0)).ravel()


def receiver_geneset(rna, idx: np.ndarray, rep: str = "X_integrated", resolution: float = 0.5,
                     n_per_state: int = 60, min_frac: float = 0.1, min_lfc: float = 0.25, seed: int = 0,
                     verbose: bool = False) -> Tuple[List[str], pd.Series]:
    """Re-split receiver-type cells (idx) in the integrated embedding into substates, take genes high in each vs the rest → union.

    Type markers·mitochondrial·ribosomal·cell-cycle genes are removed (so type labels or the cycle are not repeated).
    returns: list of genes of interest, per-cell substate
    """
    import codex_types as cty

    lab, _, _ = cty._leiden(np.asarray(rna.obsm[rep])[idx], None, 15, resolution, seed)
    names = rna.var_names.astype(str)
    bad = _excluded_genes(names)
    X = rna.layers["lognorm"][idx]
    genes = set()
    for s in np.unique(lab):
        a, b = lab == s, lab != s
        if a.sum() < 30 or b.sum() < 30:
            continue
        ma, mb = np.asarray(X[a].mean(0)).ravel(), np.asarray(X[b].mean(0)).ravel()
        fa = np.asarray((X[a] > 0).mean(0)).ravel()
        d = np.where((fa >= min_frac) & ~bad, ma - mb, -np.inf)
        top = np.argsort(d)[::-1][:n_per_state]
        genes |= {names[i] for i in top if d[i] >= min_lfc}
    if verbose:
        print(f"    {lab.max() + 1} substates → {len(genes)} genes of interest")
    return sorted(genes), pd.Series(lab, index=rna.obs_names[idx])


def ligand_activity(prior: NicheNetPrior, geneset: Sequence[str], background: Sequence[str],
                    ligands: Sequence[str]) -> pd.DataFrame:
    """Per ligand, how well its regulatory potential over background genes separates the genes of interest.

    aupr_corrected = AUPR − fraction of genes of interest (NicheNet v2 default ranking), pearson = correlation of potential with 0/1
    """
    from sklearn.metrics import average_precision_score

    lt = prior.ligand_target
    bg = [g for g in background if g in lt.index]
    ligs = [l for l in ligands if l in lt.columns]
    if not bg or not ligs:
        return pd.DataFrame(columns=["aupr_corrected", "pearson"])
    y = np.isin(bg, list(geneset)).astype(np.float32)
    if y.sum() < 5:
        return pd.DataFrame(columns=["aupr_corrected", "pearson"])
    M = lt.loc[bg, ligs].to_numpy(np.float32)
    yc = y - y.mean()
    Mc = M - M.mean(0)
    pear = (yc @ Mc) / (np.linalg.norm(yc) * np.linalg.norm(Mc, axis=0) + 1e-12)
    aupr = np.array([average_precision_score(y, M[:, j]) for j in range(M.shape[1])]) - y.mean()
    return pd.DataFrame(dict(aupr_corrected=aupr, pearson=pear), index=ligs) \
        .sort_values("aupr_corrected", ascending=False)


def _zcols(X: sp.spmatrix, cols: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Dense z of cells × chosen genes (per group (sample) mean removed to drop donor·sample differences → divided by the overall SD)."""
    D = np.asarray(X[:, cols].todense(), dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        D[m] -= D[m].mean(0)
    return D / (D.std(0) + 1e-6)


def _module_score(X: sp.spmatrix, mean_all: np.ndarray, genes_idx: np.ndarray, groups: np.ndarray,
                  rng, n_bins: int = 25, n_ctrl: int = 5) -> np.ndarray:
    """Mean z of target genes − mean z of control genes with similar expression levels (Seurat AddModuleScore approach)."""
    ok = np.where(mean_all > 0)[0]
    bins = pd.qcut(pd.Series(mean_all[ok]).rank(method="first"), n_bins, labels=False).to_numpy()
    bin_of = dict(zip(ok, bins))
    pool = {b: ok[bins == b] for b in range(n_bins)}
    gs = set(genes_idx.tolist())
    ctrl = []
    for g in genes_idx:
        cand = np.setdiff1d(pool[bin_of[g]], list(gs))
        ctrl.extend(rng.choice(cand, min(n_ctrl, len(cand)), replace=False))
    ctrl = np.unique(ctrl)
    return _zcols(X, genes_idx, groups).mean(1) - _zcols(X, ctrl, groups).mean(1)


# ---------------------------------------------------------------------------
# 4. Similarity score (alternative route)
# ---------------------------------------------------------------------------
def _similarity_embedding(rna, drop_genes: Sequence[str], n_pcs: int = 30, batch_key: str = "sample",
                          seed: int = 0) -> np.ndarray:
    """PCA on HVGs minus drop_genes (sender-type markers) → Harmony. So ambient RNA contamination does not create neighbours."""
    from sklearn.decomposition import PCA

    hv = rna.var["highly_variable"].to_numpy() & ~rna.var_names.isin(list(drop_genes))
    L = rna.layers["lognorm"][:, np.where(hv)[0]].toarray()
    L = np.clip((L - L.mean(0)) / (L.std(0) + 1e-6), -10, 10).astype(np.float32)
    Z = PCA(n_pcs, random_state=seed, svd_solver="randomized").fit_transform(L).astype(np.float32)
    if batch_key in rna.obs and rna.obs[batch_key].nunique() > 1:
        import harmonypy

        h = harmonypy.run_harmony(Z, rna.obs[[batch_key]].astype(str), batch_key, random_state=seed, verbose=False)
        H = np.asarray(h.Z_corr, dtype=np.float32)
        Z = H if H.shape[0] == rna.n_obs else H.T
    return Z


def _similarity_score(Z: np.ndarray, rtype: np.ndarray, r_idx: np.ndarray, s: str, k: int = 10) -> np.ndarray:
    """Per receiver-type cell, negative mean distance to the k nearest cells of sender type s (larger = more similar to s).

    'Fraction of s among kNN neighbours' is 0 for most cells, giving tied ranks so the hypothesis cannot work; distance is used instead.
    """
    from sklearn.neighbors import NearestNeighbors

    s_idx = np.where(rtype == s)[0]
    d, _ = NearestNeighbors(n_neighbors=min(k, len(s_idx))).fit(Z[s_idx]).kneighbors(Z[r_idx])
    return -d.mean(1)


# ---------------------------------------------------------------------------
# 5. Remapping kernel
# ---------------------------------------------------------------------------
def _rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return ((rankdata(x) - 0.5) / len(x)).astype(np.float32)


@dataclass
class _Cand:
    """Candidates for one receiver type: for each CODEX cell i, the n_cand closest scRNA cells in protein space."""
    ci: np.ndarray            # CODEX cell indices (global)
    ri: np.ndarray            # scRNA candidate pool (global)
    nb: np.ndarray            # (len(ci), n_cand) positions within the pool
    kp: np.ndarray            # protein kernel exp(−d² / 2h²)


def _candidates(C: np.ndarray, R: np.ndarray, ci, ri, n_cand: int, k_ref: int = 20) -> _Cand:
    from sklearn.neighbors import NearestNeighbors

    import rna_map as rm

    C, R = rm._rank_norm(C), rm._rank_norm(R)
    kk = min(n_cand, len(ri))
    d, nb = NearestNeighbors(n_neighbors=kk).fit(R).kneighbors(C)
    h = np.median(d[:, min(k_ref, kk) - 1]) + 1e-6
    return _Cand(ci=ci, ri=ri, nb=nb, kp=np.exp(-(d ** 2) / (2 * h ** 2)).astype(np.float32))


def _weights(cd: _Cand, q_codex: Optional[np.ndarray], p_rna: Optional[np.ndarray], sigma: float,
             n_iter: int = 10) -> sp.csr_matrix:
    """(len(ci) × len(ri)) sparse weights. q_codex (len(ci), m), p_rna (len(ri), m) are [0, 1] ranks.

    Balanced cell use: column scales adjusted iteratively so column sums approach len(ci)/len(ri) (relaxed Sinkhorn) → rows sum to 1.
    """
    w = cd.kp.copy()
    if q_codex is not None and np.isfinite(sigma):
        for j in range(q_codex.shape[1]):
            w *= np.exp(-((q_codex[:, j][:, None] - p_rna[:, j][cd.nb]) ** 2) / (2 * sigma ** 2))
    w += 1e-12
    n, m = w.shape[0], len(cd.ri)
    cols = cd.nb.ravel()
    target = n / m
    b = np.ones(m, np.float32)
    for _ in range(n_iter):
        ww = w * b[cd.nb]
        ww /= ww.sum(1, keepdims=True)
        cs = np.bincount(cols, weights=ww.ravel(), minlength=m)
        b *= np.clip(np.sqrt(target / np.maximum(cs, 1e-9)), 0.5, 2.0).astype(np.float32)
        b = np.clip(b, 1e-3, 1e3)
    ww = w * b[cd.nb]
    ww /= ww.sum(1, keepdims=True)
    return sp.csr_matrix((ww.ravel(), (np.repeat(np.arange(n), cd.nb.shape[1]), cols)), shape=(n, m))


def _prune(W: sp.csr_matrix, keep: int) -> sp.csr_matrix:
    """Keep only the top keep weights per row and renormalise rows to 1."""
    W = W.tocsr()
    rows, cols, vals = [], [], []
    for i in range(W.shape[0]):
        s, e = W.indptr[i], W.indptr[i + 1]
        v, c = W.data[s:e], W.indices[s:e]
        if len(v) > keep:
            o = np.argpartition(v, -keep)[-keep:]
            v, c = v[o], c[o]
        rows.append(np.full(len(v), i))
        cols.append(c)
        vals.append(v / v.sum())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=W.shape)


# ---------------------------------------------------------------------------
# 6. Judgement
# ---------------------------------------------------------------------------
def _spearman_cols(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    rb = rankdata(b)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    out = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        ra = rankdata(A[:, j])
        out[j] = ((ra - ra.mean()) / (ra.std() + 1e-12) * rb).mean()
    return out


def _validate(cd: _Cand, G: np.ndarray, P_obs: np.ndarray, S: sp.csr_matrix, q: np.ndarray, p: np.ndarray,
              sigma: float, n_null: int, rng) -> Dict:
    """G: pool scRNA × validation genes, P_obs: receiver CODEX cells × validation proteins, S: same-type spatial neighbour-mean operator.

    score = Spearman between the neighbour-averaged remapped gene values and the neighbour-averaged protein (per protein).
    gain = hypothesis − baseline (σ=∞). null = CODEX-side hypothesis shuffled among receiver cells. p values from the max statistic.
    """
    Ps = S @ P_obs

    def score(W):
        return np.array([_spearman_cols(S @ (W @ G[:, [j]]), Ps[:, j])[0] for j in range(G.shape[1])])

    base = score(_weights(cd, None, None, np.inf))
    obs = score(_weights(cd, q, p, sigma)) - base
    null = np.array([score(_weights(cd, q[rng.permutation(len(q))], p, sigma)) - base for _ in range(n_null)])
    pval = (1 + (null.max(1) >= obs.max()).sum()) / (1 + n_null)
    return dict(base=base, gain=obs, null_mean=null.mean(0), null_max=null.max(1), p=pval)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class NicheRemapResult:
    adjacency: pd.DataFrame                        # type-pair adjacency (long table)
    routes: pd.DataFrame                           # per receiver·sender pair, routes tried and the verdict
    validation: pd.DataFrame                       # pair × route × validation protein: base, gain, null_mean
    activities: Dict[str, pd.DataFrame]            # receiver type → ligand activity table
    remap: "object"                                # rna_map.CellMap (remapped with passing hypotheses — hypothesis map)
    baseline: "object"                             # rna_map.CellMap (σ=∞, protein only, same candidates·balancing)
    params: Dict = field(default_factory=dict)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.adjacency.to_csv(os.path.join(out_dir, "adjacency.csv"), index=False)
        self.routes.to_csv(os.path.join(out_dir, "routes.csv"), index=False)
        self.validation.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
        act = {k: v for k, v in self.activities.items() if len(v)}
        if act:
            pd.concat(act, names=["receiver", "ligand"]) \
                .to_csv(os.path.join(out_dir, "ligand_activity.csv"))
        self.remap.save(out_dir, prefix="remap_")
        self.baseline.save(out_dir, prefix="baseline_")
        with open(os.path.join(out_dir, "niche_params.json"), "w") as f:
            json.dump(self.params, f, indent=1, default=str)

    @classmethod
    def load(cls, out_dir: str) -> "NicheRemapResult":
        import rna_map as rm

        j = lambda f: os.path.join(out_dir, f)
        act = {}
        if os.path.exists(j("ligand_activity.csv")):
            a = pd.read_csv(j("ligand_activity.csv"))
            act = {k: g.drop(columns="receiver").set_index("ligand") for k, g in a.groupby("receiver")}
        with open(j("niche_params.json")) as f:
            params = json.load(f)
        return cls(adjacency=pd.read_csv(j("adjacency.csv")), routes=pd.read_csv(j("routes.csv")),
                   validation=pd.read_csv(j("validation.csv")), activities=act,
                   remap=rm.CellMap.load(out_dir, prefix="remap_"),
                   baseline=rm.CellMap.load(out_dir, prefix="baseline_"), params=params)


# ---------------------------------------------------------------------------
# 7. Full run
# ---------------------------------------------------------------------------
def run_niche_remap(codex, rna, plan: pd.DataFrame, links: pd.DataFrame, prior: Optional[NicheNetPrior],
                    codex_key: str = "cell_type", rna_key: str = "codex_type",
                    validation_proteins: Sequence[str] = VALIDATION_PROTEINS,
                    k_spatial: int = 10, min_ratio: float = 1.2, min_z: float = 2.0, max_senders: int = 6,
                    min_expr_frac: float = 0.1, n_ligands: int = 5, n_target_pool: int = 250,
                    min_targets: int = 10, n_cand: int = 200, keep: int = 30, sigma: float = 0.15,
                    n_null: int = 20, alpha: float = 0.05, min_gain: float = 0.02, max_validate: int = 20000,
                    doublet_q: float = 0.9, exclude_codex: Optional[str] = "spill", seed: int = 0,
                    verbose: bool = True) -> NicheRemapResult:
    """On top of the mean mapping, try·judge NicheNet → (if that fails) the similarity route per receiver·sender pair, and remap
    within types with the passing hypotheses. With prior=None, NicheNet is skipped and only the similarity route is used.

    sigma        : hypothesis kernel width (rank units). Smaller follows the hypothesis more strongly
    n_cand, keep : number of protein candidates, number of neighbours kept in the final weights
    max_validate : cap on receiver-type CODEX cells used for judgement (random sample)
    doublet_q    : per type, scRNA cells at or above this doublet-score quantile are excluded from remapping candidates
    """
    import rna_map as rm

    t0 = time.time()
    rng = np.random.default_rng(seed)
    lineage = _lineage_of()
    markers = _markers()
    ctype = codex.obs[codex_key].astype(str).to_numpy()
    rtype = rna.obs[rna_key].astype(str).to_numpy()
    names_c = list(map(str, codex.var_names))
    Xc = np.asarray(codex.layers["input"] if "input" in codex.layers else codex.X, dtype=np.float32)
    skip = (codex.obs[exclude_codex].to_numpy() > 0) if exclude_codex and exclude_codex in codex.obs else \
        np.zeros(codex.n_obs, bool)
    L = rna.layers["lognorm"]
    gnames = rna.var_names.astype(str)
    vi = {g: i for i, g in enumerate(gnames)}
    samples = rna.obs["sample"].astype(str).to_numpy() if "sample" in rna.obs else np.zeros(rna.n_obs, str)

    # --- spatial neighbours, adjacent pairs
    adj = adjacency_enrichment(codex, codex_key, k_spatial, seed=seed, verbose=verbose)
    mapped_types = [t for t, r in plan.iterrows() if r.unit == "type"]
    in_rna = set(mapped_types)

    # --- judging proteins (linked ones only) / link table without them for protein features
    vprot = [m for m in validation_proteins if m in links.index and links.loc[m, "used"]]
    links_v = links.copy()
    links_v.loc[vprot, "used"] = False
    C_feat_v = rm._codex_feature_matrix(codex, links_v)
    R_feat_v = rm._rna_feature_matrix(rna, links_v)
    C_feat = rm._codex_feature_matrix(codex, links)
    R_feat = rm._rna_feature_matrix(rna, links)
    G_v = np.column_stack([np.asarray(L[:, [vi[g] for g in links.loc[m, "found"].split(",")]].sum(1)).ravel()
                           for m in vprot]).astype(np.float32) if vprot else np.zeros((rna.n_obs, 0), np.float32)
    P_v = Xc[:, [names_c.index(m) for m in vprot]]
    if not vprot:
        raise ValueError("No judging proteins. Set validation_proteins to linked markers of the panel")
    if verbose:
        print(f"[remap] judging proteins: {vprot}")

    # --- sender-type ligand expression (scRNA type mean / expressing fraction)
    tstats = {t: _type_stats(rna, np.where(rtype == plan.loc[t, "rna_group"])[0]) for t in mapped_types}

    sim_cache: Dict[str, np.ndarray] = {}
    routes, vrows, activities = [], [], {}
    final_q, final_p = {}, {}
    W_parts, W_base_parts = [], []
    stats_rows = []

    for r in mapped_types:
        ci_all = np.where((ctype == r) & ~skip)[0]
        ri = np.where(rtype == plan.loc[r, "rna_group"])[0]
        if "doublet_score" in rna.obs and len(ri) > 50:
            ds = rna.obs["doublet_score"].to_numpy()[ri]
            ri = ri[ds < np.quantile(ds, doublet_q)]
        if len(ci_all) < 50 or len(ri) < 30:
            continue
        if verbose:
            print(f"[remap] {r}: CODEX {len(ci_all):,}, scRNA candidates {len(ri):,}")

        # sender-type candidates
        a = adj[(adj.receiver == r) & (adj.sender != r) & (adj.ratio >= min_ratio) & (adj.z >= min_z)]
        a = a[~a.sender.str.endswith("?") & (a.sender != "Unknown")].sort_values("z", ascending=False)
        senders = list(a.sender.head(max_senders))

        # candidates for judgement (features without validation proteins) — once per receiver type
        ci_v = ci_all if len(ci_all) <= max_validate else np.sort(rng.choice(ci_all, max_validate, replace=False))
        cd_v = _candidates(C_feat_v[ci_v], R_feat_v[ri], ci_v, ri, n_cand)
        S_v = _same_type_smoother(codex, ci_v)
        G_r, P_r = G_v[ri], P_v[ci_v]

        # NicheNet: receiver-type genes of interest · ligand activity (once)
        act, targets_of = None, {}
        if prior is not None and senders:
            mean_r, frac_r = _type_stats(rna, ri)
            gs, _ = receiver_geneset(rna, ri, seed=seed)
            bg = [g for g, f in zip(gnames, frac_r) if f >= 0.05]
            rec_ok = set(gnames[frac_r >= min_expr_frac])
            lr_ok = prior.lr[prior.lr.receptor.isin(rec_ok)]
            cand_l = set()
            for s in senders:
                if s in in_rna:
                    _, fs = tstats[s]
                    cand_l |= {l for l in lr_ok.ligand.unique() if l in vi and fs[vi[l]] >= min_expr_frac}
            act = ligand_activity(prior, gs, bg, sorted(cand_l))
            activities[r] = act
            bad = _excluded_genes(gnames)
            bgset = set(bg)
            gsset = set(gs)
            for l in act.index[:max(n_ligands * 4, 20)]:
                col = prior.ligand_target[l]
                top = col.nlargest(n_target_pool).index
                targets_of[l] = [g for g in top if g in bgset and g in gsset and not bad[vi[g]]]
            if verbose:
                print(f"    {len(gs)} genes of interest, {len(cand_l)} candidate ligands, top activity: " +
                      ", ".join(f"{l} {v:.3f}" for l, v in act.aupr_corrected.head(5).items()))

        q_codex_all = {s: _rank01(_proximity(codex, ci_all, ctype == s)) for s in senders}
        pos_v = np.searchsorted(ci_all, ci_v)

        for s in senders:
            row = dict(receiver=r, sender=s, adj_ratio=float(a.set_index("sender").loc[s, "ratio"]),
                       adj_z=float(a.set_index("sender").loc[s, "z"]), route="none", reason="", ligands="",
                       n_targets=0, tried="", gain=np.nan, p=np.nan, best_protein="")
            if s not in in_rna:
                row["reason"] = "sender_absent (type not in scRNA)"
                routes.append(row)
                continue
            q = q_codex_all[s]
            tried = []
            nn_failed_validation = False

            # --- NicheNet route
            p_nn = None
            if act is not None and len(act):
                ms, fs = tstats[s]
                ligs = [l for l in act.index if act.loc[l, "aupr_corrected"] > 0 and l in vi
                        and fs[vi[l]] >= min_expr_frac and len(targets_of.get(l, [])) >= min_targets][:n_ligands]
                if ligs:
                    sc = []
                    wts = []
                    for l in ligs:
                        gi = np.array([vi[g] for g in targets_of[l]])
                        sc.append(_module_score(L[ri], np.asarray(L[ri].mean(0)).ravel(), gi, samples[ri], rng))
                        wts.append(act.loc[l, "aupr_corrected"] * ms[vi[l]])
                    wts = np.asarray(wts) / np.sum(wts)
                    p_nn = _rank01(np.average(np.vstack(sc), axis=0, weights=wts))
                    row["ligands"] = ",".join(ligs)
                    row["n_targets"] = int(np.median([len(targets_of[l]) for l in ligs]))
                else:
                    row["reason"] = f"nichenet: no active ligand with ≥ {min_targets} targets"
            elif prior is None:
                row["reason"] = "nichenet: no prior resources"
            if p_nn is not None:
                v = _validate(cd_v, G_r, P_r, S_v, q[pos_v][:, None], p_nn[:, None], sigma, n_null, rng)
                tried.append("nichenet")
                _vrows(vrows, r, s, "nichenet", vprot, v)
                ok = v["p"] <= alpha and v["gain"].max() >= min_gain
                row.update(gain=float(v["gain"].max()), p=float(v["p"]),
                           best_protein=vprot[int(np.argmax(v["gain"]))] if vprot else "")
                if ok:
                    row["route"] = "nichenet"
                    final_q.setdefault(r, []).append(q)
                    final_p.setdefault(r, []).append(p_nn)
                    row["tried"] = ",".join(tried)
                    routes.append(row)
                    continue
                nn_failed_validation = True
                row["reason"] = "nichenet: failed judgement"

            # --- similarity route
            fam = lineage.get(r, r) == lineage.get(s, s) or frozenset((r, s)) in RELATED_PAIRS
            if fam:
                row["reason"] += " | similarity: not used (same lineage·differentiation relationship)"
            else:
                if s not in sim_cache:
                    sim_cache[s] = _similarity_embedding(rna, markers.get(s, []), seed=seed)
                p_sim = _rank01(_similarity_score(sim_cache[s], rtype, ri, plan.loc[s, "rna_group"]))
                v = _validate(cd_v, G_r, P_r, S_v, q[pos_v][:, None], p_sim[:, None], sigma, n_null, rng)
                tried.append("similarity")
                _vrows(vrows, r, s, "similarity", vprot, v)
                ok = v["p"] <= alpha and v["gain"].max() >= min_gain
                if ok:
                    row["route"] = "similarity_weak" if nn_failed_validation else "similarity"
                    row.update(gain=float(v["gain"].max()), p=float(v["p"]),
                               best_protein=vprot[int(np.argmax(v["gain"]))] if vprot else "")
                    final_q.setdefault(r, []).append(q)
                    final_p.setdefault(r, []).append(p_sim)
                else:
                    row["reason"] += " | similarity: failed judgement"
                    if not np.isfinite(row["gain"]):
                        row.update(gain=float(v["gain"].max()), p=float(v["p"]))
            row["reason"] = row["reason"].strip(" |")
            row["tried"] = ",".join(tried)
            routes.append(row)

        # --- final remapping (all linked features) · baseline
        cd = _candidates(C_feat[ci_all], R_feat[ri], ci_all, ri, n_cand)
        Wb = _prune(_weights(cd, None, None, np.inf), keep)
        if r in final_q:
            Wr = _prune(_weights(cd, np.column_stack(final_q[r]), np.column_stack(final_p[r]), sigma), keep)
        else:
            Wr = Wb
        W_parts.append((ci_all, ri, Wr))
        W_base_parts.append((ci_all, ri, Wb))
        stats_rows.append(dict(codex_type=r, n_codex=len(ci_all), n_rna=len(ri), n_priors=len(final_q.get(r, [])),
                               senders=",".join(senders)))

    def assemble(parts, tag):
        rows_, cols_, vals_ = [], [], []
        group = np.full(codex.n_obs, "", dtype=object)
        for ci, ri, W in parts:
            W = W.tocoo()
            rows_.append(ci[W.row])
            cols_.append(ri[W.col])
            vals_.append(W.data)
            group[ci] = f"{tag}:{ctype[ci[0]]}"
        Wfull = sp.csr_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))),
                              shape=(codex.n_obs, rna.n_obs), dtype=np.float32) if rows_ else \
            sp.csr_matrix((codex.n_obs, rna.n_obs), dtype=np.float32)
        return rm.CellMap(W=Wfull, codex_names=codex.obs_names.to_numpy(),
                          rna_names=rna.obs_names.to_numpy().astype(str),
                          group=pd.Series(group, index=codex.obs_names),
                          dist=pd.Series(np.nan, index=codex.obs_names, dtype=np.float32),
                          stats=pd.DataFrame(stats_rows).set_index("codex_type") if stats_rows else pd.DataFrame(),
                          params=dict(method=f"niche_remap:{tag}", codex_key=codex_key, rna_key=rna_key))

    params = dict(k_spatial=k_spatial, min_ratio=min_ratio, min_z=min_z, max_senders=max_senders,
                  min_expr_frac=min_expr_frac, n_ligands=n_ligands, n_target_pool=n_target_pool,
                  min_targets=min_targets, n_cand=n_cand, keep=keep, sigma=sigma, n_null=n_null, alpha=alpha,
                  min_gain=min_gain, max_validate=max_validate, doublet_q=doublet_q, seed=seed,
                  validation_proteins=vprot, nichenet=prior is not None)
    res = NicheRemapResult(adjacency=adj, routes=pd.DataFrame(routes), validation=pd.DataFrame(vrows),
                           activities=activities, remap=assemble(W_parts, "remap"),
                           baseline=assemble(W_base_parts, "baseline"), params=params)
    if verbose and len(res.routes):
        print(f"[remap] routes: " + ", ".join(f"{k} {v}" for k, v in res.routes.route.value_counts().items()) +
              f" | {time.time() - t0:.0f}s")
    return res


def _proximity(codex, ci: np.ndarray, sender_mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Per CODEX cell (ci), negative mean distance to the k nearest sender-type cells (larger = closer).

    The fraction among k neighbours is mostly 0, giving tied ranks, so the continuous distance is used.
    """
    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(codex.obsm["spatial"], dtype=np.float64)
    s_idx = np.where(sender_mask)[0]
    d, _ = NearestNeighbors(n_neighbors=min(k, len(s_idx))).fit(xy[s_idx]).kneighbors(xy[ci])
    return -d.mean(1)


def _same_type_smoother(codex, ci: np.ndarray, k: int = 10) -> sp.csr_matrix:
    """Mean operator over k spatial neighbours + self among cells of the same type (ci) (len(ci) × len(ci))."""
    xy = np.asarray(codex.obsm["spatial"], dtype=np.float64)[ci]
    kk = min(k, len(ci) - 1)
    idx = np.column_stack([np.arange(len(ci)), _spatial_knn(xy, kk)])
    n = len(ci)
    return sp.csr_matrix((np.full(idx.size, 1.0 / idx.shape[1], np.float32),
                          (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())), shape=(n, n))


def _vrows(out: List, r: str, s: str, route: str, prots: Sequence[str], v: Dict) -> None:
    for j, m in enumerate(prots):
        out.append(dict(receiver=r, sender=s, route=route, protein=m, base=float(v["base"][j]),
                        gain=float(v["gain"][j]), null_mean=float(v["null_mean"][j]), p_max=float(v["p"])))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_adjacency(adj: pd.DataFrame, types: Optional[Sequence[str]] = None, figsize=None):
    """Receiver × sender type log2(obs/exp) heatmap."""
    import matplotlib.pyplot as plt

    M = adj.pivot(index="receiver", columns="sender", values="ratio")
    if types is not None:
        M = M.reindex(index=[t for t in types if t in M.index], columns=[t for t in types if t in M.columns])
    M = np.log2(M.clip(lower=1 / 16))
    fig, ax = plt.subplots(figsize=figsize or (0.4 * M.shape[1] + 3, 0.35 * M.shape[0] + 1.5))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    ax.set_xlabel("neighbour (sender)")
    ax.set_ylabel("cell (receiver)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="log2 observed / shuffled")
    fig.tight_layout()
    return fig


def plot_routes(routes: pd.DataFrame, figsize=None):
    """Chosen route and gain per receiver × sender type pair."""
    import matplotlib.pyplot as plt

    col = {"nichenet": "#2b7bba", "similarity": "#3a9e5f", "similarity_weak": "#9ccf8f", "none": "#d9d9d9"}
    R = sorted(routes.receiver.unique())
    S = sorted(routes.sender.unique())
    fig, ax = plt.subplots(figsize=figsize or (0.55 * len(S) + 3, 0.45 * len(R) + 1.5))
    for _, x in routes.iterrows():
        i, j = R.index(x.receiver), S.index(x.sender)
        ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=col.get(x.route, "#d9d9d9")))
        if np.isfinite(x.gain):
            ax.text(j, i, f"{x.gain:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, len(S) - 0.5)
    ax.set_ylim(len(R) - 0.5, -0.5)
    ax.set_xticks(range(len(S)), S, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(R)), R, fontsize=8)
    ax.set_xlabel("sender")
    ax.set_ylabel("receiver")
    for k, c in col.items():
        ax.scatter([], [], marker="s", color=c, label=k)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("route per pair (text = best validation gain)")
    fig.tight_layout()
    return fig


def plot_remap(codex, values: pd.Series, title: str = "", region: Optional[Tuple[float, float, float, float]] = None,
               s: float = 0.5, figsize=(6, 6)):
    """Plot values at CODEX positions (e.g. remapped − baseline gene values). region = (x0, x1, y0, y1)."""
    import matplotlib.pyplot as plt

    xy = np.asarray(codex.obsm["spatial"])
    v = values.reindex(codex.obs_names).to_numpy(dtype=float)
    m = np.ones(len(v), bool)
    if region:
        x0, x1, y0, y1 = region
        m = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    lim = np.nanpercentile(np.abs(v[m]), 99) if np.isfinite(v[m]).any() else 1
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[m, 0], xy[m, 1], c=v[m], s=s, cmap=cm, vmin=-lim, vmax=lim, plotnonfinite=True, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    return fig
