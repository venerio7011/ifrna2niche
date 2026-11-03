"""
rna_prep.py
===========

scRNA-seq raw (Cell Ranger raw_feature_bc_matrix) → cleaned AnnData for mapping (for Google Colab). Uses RNA only.

Steps
----
1. prepare_samples : takes only scRNA samples from a folder of per-sample archives / a file list / a GEO RAW tar / an already extracted folder
2. read_raw        : reads per-sample matrices — gene expression (Gene Expression) only; other features (antibodies etc.) are dropped
3. call_cells      : separates cells / empty droplets at the knee of the barcode UMI rank curve (same idea as DropletUtils barcodeRanks)
4. qc              : per-sample MAD criteria (low UMI·gene counts, high mitochondrial %) + absolute floors
5. doublets        : per-sample Scrublet-style score — how much cells mix in kNN with simulated doublets made by adding two real cells
                     calls the expected 10x doublet count from the top scores. Doublets of the same type cannot be caught in principle
6. normalize       : layers['counts'] (original UMI), layers['lognorm'] = log1p(CP10k), X = lognorm
7. hvg             : per-sample variance z by mean bin → genes commonly high across samples (batch-aware)
8. integrate       : standardised HVGs → PCA → Harmony (corrects differences between samples; only the embedding is corrected, expression values stay as is)
9. cluster/annotate: Leiden on the integrated embedding → RNA_MARKERS scores on cluster-mean expression → the same type names as codex_types
                     (obs['codex_type'], obs['lineage']). 'A|B?' clusters are also checked for being doublet clumps

rna_map reads the resulting h5ad as is (layers['lognorm'], var['highly_variable'], obsm['X_integrated'], obs['codex_type']).
"""

from __future__ import annotations

import glob
import gzip
import os
import re
import tarfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["RNA_MARKERS", "RNA_NEGATIVE", "RNA_REQUIRED", "refine", "prepare_samples", "read_raw", "call_cells", "load_samples", "qc", "doublets", "filter_cells",
           "normalize", "hvg", "integrate", "mixing_score", "cluster", "cluster_markers", "annotate",
           "set_cluster_types", "embed_2d", "plot_barcode_ranks", "plot_qc", "plot_doublets", "plot_embedding",
           "plot_marker_dots"]

# scRNA type markers (same names as codex_types.MARKER_SETS → both datasets use the same type vocabulary)
RNA_MARKERS: Dict[str, List[str]] = {
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5"],
    "CD4 T": ["CD3E", "CD3D", "CD4", "IL7R", "CCR7", "TCF7"],
    "CD8 T": ["CD3E", "CD8A", "CD8B", "GZMK", "CCL5", "NKG7"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],
    "Tfh": ["PDCD1", "CXCL13", "CXCR5", "ICOS", "TOX2", "IL21"],
    "Plasma": ["SDC1", "JCHAIN", "MZB1", "XBP1", "PRDM1", "IGHG1"],
    "NK": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRF1", "PRF1"],
    "Macrophage": ["CD68", "CD163", "MRC1", "C1QA", "C1QB", "APOE", "CST3", "AIF1"],
    "DC": ["ITGAX", "CLEC9A", "XCR1", "CD1C", "FCER1A", "LAMP3"],
    "pDC": ["LILRA4", "IL3RA", "CLEC4C", "IRF7", "TCF4", "GZMB"],
    "Granulocyte": ["FCGR3B", "CSF3R", "S100A8", "S100A9", "CXCR2", "MMP9"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EMCN"],
    "PDPN stroma": ["PDPN", "CCL19", "CCL21", "PDGFRA", "LUM"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "NT5E"],
    "FDC": ["CR2", "FDCSP", "CLU", "VCAM1", "SERPINE2"],
    "Smooth muscle": ["ACTA2", "MYH11", "TAGLN", "DES", "CNN1"],
    "Nerve": ["NES", "S100B", "PLP1", "NCAM1"],
    "Squamous epi": ["TP63", "KRT5", "KRT14", "KRT13", "KRT6A"],
    "Glandular epi": ["KRT7", "KRT8", "KRT18", "MUC1", "EPCAM"],
    "Secretory epi": ["MUC5AC", "MUC6", "TFF1", "TFF2"],
    "Paneth": ["DEFA5", "DEFA6", "LYZ", "REG3A"],
    "Neuroendocrine": ["CHGA", "CHGB", "SYP"],
}
# markers that must be absent (0.7 × their max is subtracted from the score) — separates type pairs sharing many markers
RNA_NEGATIVE: Dict[str, List[str]] = {
    "NK": ["CD3E", "CD3D", "CD3G"],
    "CD4 T": ["CD8A", "CD8B"],
    "CD8 T": ["CD4"],
    "B": ["CD3E", "JCHAIN"],
    "Plasma": ["MS4A1"],
    "pDC": ["MS4A1", "CD3E"],
    "Granulocyte": ["CST3", "HLA-DRA"],          # separates from monocytes·macrophages (both low in neutrophils)
}
# markers that must be expressed: if the fraction of expressing cells in a cluster is below min_frac, that type's score is −1 (defining markers)
RNA_REQUIRED: Dict[str, List[str]] = {
    "Treg": ["FOXP3"],
    "Tfh": ["PDCD1"],
    "pDC": ["LILRA4"],
    "Plasma": ["JCHAIN"],
}
_MATRIX_FILES = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")


def _lineage_of() -> Dict[str, str]:
    try:
        import codex_types as cty
        return {t: d.get("lineage", t) for t, d in cty.MARKER_SETS.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 1–3. Extracting samples, reading, calling cells
# ---------------------------------------------------------------------------
_ARCHIVE_EXT = (".tar.gz", ".tgz", ".tar")


def _sample_name(fname: str) -> str:
    """'GSM5051493_Tonsil_1a_scRNA_seq_raw_..._mtx.tsv.tar.gz' → 'Tonsil_1a'. If the rule does not match, the file name without extension."""
    base = os.path.basename(fname)
    m = re.match(r"GSM\d+_(.+?)_scRNA", base)
    return m.group(1) if m else re.sub(r"(\.tar)?\.gz$|\.tgz$|\.tar$", "", base)


def prepare_samples(source, out_dir: str, pattern: str = "scRNA", verbose: bool = True) -> Dict[str, str]:
    """Sample name → raw matrix folder (barcodes / features / matrix.mtx.gz).

    source : one of
             - a folder of per-sample archives (*.tar.gz / *.tgz / *.tar)  ← when only the scRNA files were downloaded from GEO
             - a list of such file paths
             - a GEO RAW tar (with per-sample *.tar.gz inside)
             - the parent folder of already extracted matrix folders
    pattern: only samples whose file name contains this (excludes other files in the same folder such as ATAC fragments, CODEX csv)
    out_dir: where to put extracted matrices. Not extracted again if already present
    """
    os.makedirs(out_dir, exist_ok=True)
    out = {}

    def _done(name):
        dest = os.path.join(out_dir, name)
        return dest if all(os.path.exists(os.path.join(dest, f)) for f in _MATRIX_FILES) else None

    def _from_inner(fileobj, name, label):
        import shutil

        dest = os.path.join(out_dir, name)
        tmp = dest + ".part"                                    # renamed only after full extraction, so an interrupted folder is not mistaken for a cache
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            with tarfile.open(fileobj=fileobj, mode="r:*") as t:          # gzip or not
                for m in t.getmembers():
                    b = os.path.basename(m.name)
                    if m.isfile() and b in _MATRIX_FILES:      # drop paths and keep file names only → safely flattened
                        with t.extractfile(m) as src, open(os.path.join(tmp, b), "wb") as dst:
                            shutil.copyfileobj(src, dst)
        except (tarfile.TarError, EOFError, OSError) as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"Cannot extract {label} (incomplete upload or corrupted): {e}") from e
        miss = [f for f in _MATRIX_FILES if not os.path.exists(os.path.join(tmp, f))]
        if miss:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"{miss} missing in {label} (check that it is a Cell Ranger raw matrix archive)")
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
        return dest

    def _from_files(files):
        for f in files:
            name = _sample_name(f)
            if name in out:
                raise ValueError(f"Duplicate sample name: {name} ({f})")
            if _done(name):
                out[name] = _done(name)
                continue
            if verbose:
                print(f"[prep] extracting: {os.path.basename(f)} ({os.path.getsize(f) / 1e6:.0f} MB)")
            with open(f, "rb") as fh:
                out[name] = _from_inner(fh, name, os.path.basename(f))

    t0 = time.time()
    if isinstance(source, (list, tuple)):
        _from_files([str(f) for f in source])
    elif os.path.isfile(source) and tarfile.is_tarfile(source) and pattern not in os.path.basename(source):
        with tarfile.open(source) as outer:                      # GEO RAW tar
            for m in outer.getmembers():
                if m.isfile() and pattern in m.name and m.name.endswith(_ARCHIVE_EXT):
                    name = _sample_name(m.name)
                    if _done(name):
                        out[name] = _done(name)
                        continue
                    if verbose:
                        print(f"[prep] extracting: {m.name} ({m.size / 1e6:.0f} MB)")
                    out[name] = _from_inner(outer.extractfile(m), name, m.name)
    elif os.path.isfile(source):                                 # a single sample archive
        _from_files([source])
    elif os.path.isdir(source):
        files = sorted(f for f in glob.glob(os.path.join(source, "*"))
                       if os.path.isfile(f) and pattern in os.path.basename(f) and f.endswith(_ARCHIVE_EXT))
        _from_files(files)
        if not out:                                              # an already extracted folder
            for mtx in sorted(glob.glob(os.path.join(source, "**", "matrix.mtx.gz"), recursive=True)):
                d = os.path.dirname(mtx)
                out[os.path.basename(d.rstrip("/")) if "raw_feature" not in d else d.split(os.sep)[-3]] = d
    else:
        raise FileNotFoundError(source)
    if not out:
        raise FileNotFoundError(f"No sample with '{pattern}' in its name found: {source}")
    if verbose:
        print(f"[prep] {len(out)} samples: {list(out)} ({time.time() - t0:.0f}s)")
    return out


def read_raw(mtx_dir: str, sample: str, min_umi: int = 10, verbose: bool = True):
    """raw matrix → AnnData (barcode × gene, counts float32). Gene Expression only; barcodes with UMI < min_umi are dropped.

    Barcode names = 'sample:barcode'. obs['total_umi'] is for the cell-calling curve.
    """
    import anndata as ad

    t0 = time.time()
    feats = pd.read_csv(os.path.join(mtx_dir, "features.tsv.gz"), sep="\t", header=None)
    bcs = pd.read_csv(os.path.join(mtx_dir, "barcodes.tsv.gz"), sep="\t", header=None)[0].astype(str).to_numpy()
    with gzip.open(os.path.join(mtx_dir, "matrix.mtx.gz"), "rt") as fh:
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        nr, nc, _ = map(int, line.split())
        coo = pd.read_csv(fh, sep=" ", header=None, dtype={0: np.int32, 1: np.int32, 2: np.float32}).to_numpy()
    M = sp.csr_matrix((coo[:, 2].astype(np.float32), (coo[:, 1].astype(np.int64) - 1, coo[:, 0].astype(np.int64) - 1)),
                      shape=(nc, nr))
    del coo
    ge = (feats[2] == "Gene Expression").to_numpy() if feats.shape[1] > 2 else np.ones(len(feats), bool)
    M = M[:, np.where(ge)[0]]
    tot = np.asarray(M.sum(1)).ravel()
    keep = tot >= min_umi
    M = M[keep]
    var = pd.DataFrame({"gene_id": feats.loc[ge, 0].astype(str).to_numpy()},
                       index=pd.Index(feats.loc[ge, 1].astype(str).to_numpy(), name="gene"))
    a = ad.AnnData(M, obs=pd.DataFrame({"sample": sample, "barcode": bcs[keep], "total_umi": tot[keep]},
                                       index=[f"{sample}:{b}" for b in bcs[keep]]), var=var)
    a.var_names_make_unique()
    if verbose:
        print(f"[prep] {sample}: {keep.sum():,} of {nc:,} barcodes with UMI ≥ {min_umi}, {ge.sum():,} genes "
              f"({int((~ge).sum())} other features dropped) | {time.time() - t0:.0f}s")
    return a


def call_cells(a, lower: int = 100, method: str = "inflection", min_cells: int = 100, verbose: bool = True) -> Dict:
    """Find the cell boundary on the UMI rank curve (log rank vs log UMI). Fills obs['is_cell'] and returns curve information.

    inflection : the point where the curve drops most steeply (middle of the cell → empty-droplet cliff). Default
    knee       : the bend just before the cliff (more conservative, may miss small cells)
    lower      : barcodes with fewer UMI than this are left out of the curve (certain empty droplets)
    """
    tot = a.obs["total_umi"].to_numpy()
    t = np.sort(tot[tot >= lower])[::-1]
    if len(t) < min_cells * 2:
        raise ValueError(f"Only {len(t)} barcodes with UMI ≥ {lower}")
    u, cnt = np.unique(t[::-1], return_counts=True)            # ascending unique values
    u, cnt = u[::-1], cnt[::-1]
    rank = np.cumsum(cnt) - (cnt - 1) / 2                       # ties get average ranks
    x, y = np.log10(rank), np.log10(u)
    # resample x evenly and smooth before differentiating
    xs = np.linspace(x.min(), x.max(), 400)
    ys = np.interp(xs, x, y)
    w = 9
    ys = np.convolve(np.pad(ys, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    d1 = np.gradient(ys, xs)
    d2 = np.gradient(d1, xs)
    ok = xs >= np.log10(min_cells)
    i_inf = np.where(ok)[0][np.argmin(d1[ok])]
    curv = d2 / (1 + d1 ** 2) ** 1.5
    seg = np.where(ok & (xs < xs[i_inf]))[0]
    i_knee = seg[np.argmin(curv[seg])] if len(seg) else i_inf
    thr_inf, thr_knee = 10 ** ys[i_inf], 10 ** ys[i_knee]
    thr = thr_inf if method == "inflection" else thr_knee
    a.obs["is_cell"] = tot >= thr
    info = dict(sample=str(a.obs["sample"].iloc[0]), method=method, threshold=float(thr), knee=float(thr_knee),
                inflection=float(thr_inf), n_cells=int(a.obs["is_cell"].sum()), rank=rank, umi=u)
    if verbose:
        print(f"[cells] {info['sample']}: {info['n_cells']:,} cells (UMI ≥ {thr:.0f}, {method}; "
              f"knee {thr_knee:.0f}, inflection {thr_inf:.0f})")
    return info


def load_samples(samples: Dict[str, str], method: str = "inflection", lower: int = 100,
                 batch_map: Optional[Dict[str, str]] = None, verbose: bool = True):
    """Per sample read_raw → call_cells → keep cells only and merge. (AnnData, list of curve information)

    batch_map : sample → higher grouping such as donor (obs['donor']). e.g. {'Tonsil_1a': 'donor1', 'Tonsil_1b': 'donor1'}
    """
    import anndata as ad

    parts, curves = [], []
    for s, d in samples.items():
        raw = read_raw(d, s, verbose=verbose)
        info = call_cells(raw, lower=lower, method=method, verbose=verbose)
        curves.append(info)
        parts.append(raw[raw.obs["is_cell"].to_numpy()].copy())
        del raw
    a = ad.concat(parts, join="inner", merge="same")
    a.obs["sample"] = a.obs["sample"].astype("category")
    a.obs["donor"] = a.obs["sample"].astype(str).map(lambda x: (batch_map or {}).get(x, x)).astype("category")
    a.obs = a.obs.drop(columns=["is_cell"])
    if verbose:
        print(f"[prep] merged: {a.n_obs:,} cells × {a.n_vars:,} genes | per sample " +
              ", ".join(f"{k} {v:,}" for k, v in a.obs["sample"].value_counts(sort=False).items()))
    return a, curves


# ---------------------------------------------------------------------------
# 4. QC
# ---------------------------------------------------------------------------
def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def qc(a, sample_key: str = "sample", nmads: float = 3.0, min_umi: int = 500, min_genes: int = 250,
       max_mt_abs: float = 25.0, min_cells_gene: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Fill QC metrics and pass flags (obs['qc_pass'], obs['qc_fail'] = failure reason). Cells are not removed.

    Per sample: log UMI · log gene count < median − nmads·MAD → low,
              mitochondrial % (logit) > median + nmads·MAD or > max_mt_abs → high
    Absolute floors: UMI < min_umi, genes < min_genes.  Genes with var['n_cells'] < min_cells_gene get var['keep']=False
    returns: per-sample cell counts and exclusions by reason
    """
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    names = a.var_names.str.upper()
    mt = np.asarray(names.str.startswith("MT-"))
    ribo = np.asarray(names.str.match(r"^RP[SL]\d"))
    umi = np.asarray(X.sum(1)).ravel()
    a.obs["n_umi"] = umi
    a.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
    a.obs["pct_mt"] = 100 * np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(umi, 1)
    a.obs["pct_ribo"] = 100 * np.asarray(X[:, ribo].sum(1)).ravel() / np.maximum(umi, 1)
    a.var["n_cells"] = np.asarray((X > 0).sum(0)).ravel()
    a.var["keep"] = a.var["n_cells"] >= min_cells_gene
    reason = np.full(a.n_obs, "", dtype=object)
    rows = []
    for s, idx in a.obs.groupby(sample_key, observed=True).indices.items():
        o = a.obs.iloc[idx]
        lu, lg = np.log10(o["n_umi"] + 1), np.log10(o["n_genes"] + 1)
        p = np.clip(o["pct_mt"].to_numpy() / 100, 1e-4, 1 - 1e-4)
        lm = np.log(p / (1 - p))
        r = dict(
            low_umi=(lu < np.median(lu) - nmads * _mad(lu)) | (o["n_umi"] < min_umi),
            low_genes=(lg < np.median(lg) - nmads * _mad(lg)) | (o["n_genes"] < min_genes),
            high_mt=(lm > np.median(lm) + nmads * _mad(lm)) | (o["pct_mt"] > max_mt_abs),
        )
        row = dict(sample=s, n=len(idx))
        for k, v in r.items():
            v = np.asarray(v)
            row[k] = int(v.sum())
            for i in idx[v]:
                reason[i] = f"{reason[i]},{k}" if reason[i] else k
        rows.append(row)
    a.obs["qc_fail"] = pd.Categorical(reason)
    a.obs["qc_pass"] = reason == ""
    tab = pd.DataFrame(rows).set_index("sample")
    tab["pass"] = a.obs.groupby(sample_key, observed=True)["qc_pass"].sum()
    if verbose:
        print(f"[qc] passed {a.obs['qc_pass'].sum():,} / {a.n_obs:,} ({a.obs['qc_pass'].mean():.1%}) | "
              f"median UMI {np.median(umi):.0f}, genes {a.obs['n_genes'].median():.0f}, MT {a.obs['pct_mt'].median():.1f}% | "
              f"{(~a.var['keep']).sum():,} genes in fewer than {min_cells_gene} cells flagged")
    return tab


# ---------------------------------------------------------------------------
# 5. doublet
# ---------------------------------------------------------------------------
def _lognorm(X: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1
    Y = sp.diags((target / lib).astype(np.float32)) @ X
    Y = sp.csr_matrix(Y, dtype=np.float32)
    Y.data = np.log1p(Y.data)
    return Y


def _mean_var(Y) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(Y.mean(0)).ravel()
    m2 = np.asarray(Y.multiply(Y).mean(0)).ravel() if sp.issparse(Y) else (Y ** 2).mean(0)
    return m, np.maximum(m2 - m ** 2, 0)


def _dispersion_z(Y, n_bins: int = 20) -> np.ndarray:
    """z of log variance/mean per mean bin (Seurat 'mean.var.plot' approach). −inf for unexpressed genes."""
    m, v = _mean_var(Y)
    out = np.full(len(m), -np.inf)
    ok = m > 1e-3
    disp = np.log(v[ok] / m[ok] + 1e-12)
    bins = pd.qcut(m[ok], n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame(dict(d=disp, b=bins))
    g = df.groupby("b")["d"]
    z = (df["d"] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)
    out[ok] = z.to_numpy()
    return out


def _otsu(x: np.ndarray, bins: int = 100) -> float:
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def doublets(a, sample_key: str = "sample", only_pass: bool = True, sim_ratio: float = 2.0, n_hvg: int = 2000,
             n_pcs: int = 30, k: Optional[int] = None, mode: str = "expected", rate_per_1000: float = 0.008,
             seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Per-sample Scrublet-style doublet score → obs['doublet_score'], obs['doublet'] (0/1).

    Simulated doublets = UMI of two real cells added together (observed cells × sim_ratio of them).
    They are placed in the PCA space of observed cells, and the fraction of simulated doublets among kNN neighbours is the score.
    mode
      'expected' : calls the expected 10x doublet count (rate_per_1000 × cells/1000 × cells) from the top scores.
                   Cells below the Otsu threshold of simulated-doublet scores are not called (same idea as DoubletFinder). Default
      'otsu'     : uses only the Otsu threshold of the simulated-doublet score distribution. In samples dominated by one type (e.g. tonsil with 86 % B cells)
                   B+B simulated doublets cannot be told from single cells, the threshold drops, and 15–25 % get over-called
    The expected count uses the cell count right after cell calling (including QC failures) — the 10x doublet rate depends on the number of cells loaded.
    Doublets of the same type are poorly caught in principle. Remaining doublets are checked again as 'A|B?' clusters in annotate.
    only_pass : compute only for QC-passing cells (others NaN)
    returns   : per-sample cell count, threshold, doublet count / fraction, expected fraction
    """
    from pynndescent import NNDescent
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    score = np.full(a.n_obs, np.nan, np.float32)
    call = np.zeros(a.n_obs, np.int8)
    rows, a.uns["doublet_sim"] = [], {}
    base = a.obs["qc_pass"].to_numpy() if (only_pass and "qc_pass" in a.obs) else np.ones(a.n_obs, bool)
    for s, idx in a.obs.groupby(sample_key, observed=True).indices.items():
        t0 = time.time()
        n_loaded = len(idx)
        idx = idx[base[idx]]
        n = len(idx)
        if n < 200:
            continue
        C = X[idx]
        ns = int(n * sim_ratio)
        p1, p2 = rng.integers(0, n, ns), rng.integers(0, n, ns)
        S = C[p1] + C[p2]
        Yo, Ys = _lognorm(C), _lognorm(S)
        g = np.argsort(_dispersion_z(Yo))[::-1][:n_hvg]
        Do, Ds = Yo[:, g].toarray(), Ys[:, g].toarray()
        mu, sd = Do.mean(0), Do.std(0) + 1e-6
        Do, Ds = (Do - mu) / sd, (Ds - mu) / sd
        pca = PCA(min(n_pcs, n - 1), random_state=seed).fit(Do)
        Z = np.vstack([pca.transform(Do), pca.transform(Ds)]).astype(np.float32)
        kk = k or max(5, int(round(0.5 * np.sqrt(n))))
        kk_adj = int(round(kk * (1 + sim_ratio)))
        nn, _ = NNDescent(Z, n_neighbors=kk_adj + 1, random_state=seed, n_jobs=-1).neighbor_graph
        frac = (nn[:, 1:] >= n).mean(1)
        so, ss = frac[:n], frac[n:]
        otsu = _otsu(ss)
        exp_rate = rate_per_1000 * n_loaded / 1000
        if mode == "expected":
            n_exp = int(round(exp_rate * n))
            thr = max(otsu, float(np.sort(so)[::-1][n_exp - 1]) if n_exp > 0 else np.inf)
            hit = so >= thr if n_exp > 0 else np.zeros(n, bool)
        elif mode == "otsu":
            thr, hit = otsu, so > otsu
        else:
            raise ValueError("mode must be 'expected' or 'otsu'")
        score[idx] = so
        call[idx] = hit.astype(np.int8)
        a.uns["doublet_sim"][str(s)] = dict(sim=ss.astype(np.float32), threshold=thr)
        rows.append(dict(sample=s, n=n, otsu=round(otsu, 3), threshold=round(float(thr), 3), doublets=int(hit.sum()),
                         rate=round(float(hit.mean()), 4), expected=round(exp_rate, 4),
                         otsu_rate=round(float((so > otsu).mean()), 4), time_s=round(time.time() - t0, 1)))
    a.obs["doublet_score"] = score
    a.obs["doublet"] = call
    tab = pd.DataFrame(rows).set_index("sample")
    if verbose:
        print(f"[doublet] {mode}: {int(call.sum()):,} ({call.sum() / max(base.sum(), 1):.1%}) | per-sample rate " +
              ", ".join(f"{k} {r.rate:.1%}(expected {r.expected:.1%})" for k, r in tab.iterrows()))
    return tab


def filter_cells(a, drop_doublets: bool = True, verbose: bool = True):
    """Copy without QC failures and doublets (genes: var['keep'] only)."""
    keep = a.obs["qc_pass"].to_numpy().copy()
    if drop_doublets and "doublet" in a.obs:
        keep &= a.obs["doublet"].to_numpy() == 0
    genes = a.var["keep"].to_numpy() if "keep" in a.var else np.ones(a.n_vars, bool)
    b = a[keep][:, genes].copy()
    if verbose:
        print(f"[prep] remaining cells {b.n_obs:,} / {a.n_obs:,}, genes {b.n_vars:,}")
    return b


# ---------------------------------------------------------------------------
# 6–8. Normalisation, HVG, integration
# ---------------------------------------------------------------------------
def normalize(a, target_sum: float = 1e4, verbose: bool = True):
    """layers['counts'] = original UMI, layers['lognorm'] = log1p(CP10k), X = lognorm (in place)."""
    C = sp.csr_matrix(a.layers["counts"] if "counts" in a.layers else a.X, dtype=np.float32)
    a.layers["counts"] = C
    a.layers["lognorm"] = _lognorm(C, target_sum)
    a.X = a.layers["lognorm"].copy()
    if verbose:
        print(f"[norm] counts → CP{target_sum:.0e} → log1p (layers['lognorm'])")
    return a


def hvg(a, n_top: int = 3000, batch_key: Optional[str] = "sample", exclude: str = r"^(MT-|RP[SL]\d|HB[AB])",
        verbose: bool = True) -> pd.DataFrame:
    """Per batch, variance z by mean bin → n_top genes ordered by number of batches in which they are in the top n_top, ties by mean z.

    exclude : genes reflecting state·contamination more than cell type, such as mitochondrial·ribosomal·haemoglobin genes, are removed
    fills var['highly_variable'], var['hvg_z'], var['hvg_nbatches']
    """
    L = a.layers["lognorm"]
    groups = a.obs.groupby(batch_key, observed=True).indices if batch_key else {"all": np.arange(a.n_obs)}
    Z, top = [], []
    for _, idx in groups.items():
        z = _dispersion_z(L[idx])
        Z.append(z)
        t = np.zeros(a.n_vars, bool)
        t[np.argsort(z)[::-1][:n_top]] = True
        top.append(t)
    Z = np.vstack(Z)
    zm = np.where(np.isfinite(Z), Z, np.nan)
    with np.errstate(all="ignore"):
        mean_z = np.nanmean(zm, 0)
    nb = np.vstack(top).sum(0)
    bad = np.asarray(a.var_names.str.upper().str.match(exclude)) if exclude else np.zeros(a.n_vars, bool)
    order = np.lexsort((-np.nan_to_num(mean_z, nan=-1e9), -nb))
    order = [i for i in order if not bad[i] and np.isfinite(mean_z[i])][:n_top]
    a.var["highly_variable"] = False
    a.var.iloc[order, a.var.columns.get_loc("highly_variable")] = True
    a.var["hvg_z"] = mean_z.astype(np.float32)
    a.var["hvg_nbatches"] = nb
    if verbose:
        sel = a.var["highly_variable"]
        print(f"[hvg] {sel.sum():,} | genes in the top in all {len(groups)} batches {(nb[sel.to_numpy()] == len(groups)).sum():,}")
    return a.var.loc[a.var["highly_variable"], ["hvg_z", "hvg_nbatches"]].sort_values("hvg_z", ascending=False)


def _scaled_hvg(a, max_value: float = 10.0) -> np.ndarray:
    L = a.layers["lognorm"][:, np.where(a.var["highly_variable"].to_numpy())[0]].toarray()
    L = (L - L.mean(0)) / (L.std(0) + 1e-6)
    return np.clip(L, -max_value, max_value).astype(np.float32)


def integrate(a, batch_key: str = "sample", n_pcs: int = 30, method: str = "harmony", seed: int = 0,
              verbose: bool = True):
    """Standardised HVGs → PCA (obsm['X_pca']) → Harmony correction (obsm['X_integrated']). Expression values (layers) are unchanged.

    With method='none', X_pca is used as X_integrated without correction (single sample, or for comparison).
    """
    from sklearn.decomposition import PCA

    t0 = time.time()
    Z = PCA(n_pcs, random_state=seed, svd_solver="randomized").fit_transform(_scaled_hvg(a)).astype(np.float32)
    a.obsm["X_pca"] = Z
    if method == "harmony" and a.obs[batch_key].nunique() > 1:
        import harmonypy

        h = harmonypy.run_harmony(Z, a.obs[[batch_key]].astype(str), batch_key, random_state=seed, verbose=False)
        H = np.asarray(h.Z_corr, dtype=np.float32)
        a.obsm["X_integrated"] = H if H.shape[0] == a.n_obs else H.T     # orientation differs between harmonypy versions
    elif method in ("harmony", "none"):
        a.obsm["X_integrated"] = Z
    else:
        raise ValueError("method must be 'harmony' or 'none'")
    a.uns["integrate"] = dict(method=method, batch_key=batch_key, n_pcs=n_pcs)
    if verbose:
        print(f"[integrate] PCA {n_pcs} → {method} | {time.time() - t0:.0f}s")
    return a


def mixing_score(a, rep: str, batch_key: str = "sample", k: int = 30, n_sample: int = 5000, seed: int = 0) -> float:
    """Fraction of other-batch cells among kNN neighbours ÷ the fraction expected if randomly mixed (1 = fully mixed, 0 = not mixed at all)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    b = a.obs[batch_key].astype(str).to_numpy()
    q = rng.choice(a.n_obs, min(n_sample, a.n_obs), replace=False)
    _, nn = NearestNeighbors(n_neighbors=k + 1).fit(a.obsm[rep]).kneighbors(a.obsm[rep][q])
    obs = (b[nn[:, 1:]] != b[q, None]).mean()
    p = pd.Series(b).value_counts(normalize=True)
    exp = 1 - (p ** 2).sum()
    return float(obs / exp) if exp > 0 else float("nan")


# ---------------------------------------------------------------------------
# 9. Clusters, types
# ---------------------------------------------------------------------------
def cluster(a, rep: str = "X_integrated", n_neighbors: int = 15, resolution: float = 1.0, seed: int = 0,
            verbose: bool = True):
    """kNN → Leiden on the integrated embedding. obs['cluster'] (0, 1, 2 … from the largest cluster)."""
    import codex_types as cty

    t0 = time.time()
    lab, _, idx = cty._leiden(a.obsm[rep], None, n_neighbors, resolution, seed)
    a.obs["cluster"] = pd.Categorical(lab.astype(str), categories=[str(i) for i in range(lab.max() + 1)])
    a.obsm["knn"] = idx
    a.uns["cluster"] = dict(rep=rep, n_neighbors=n_neighbors, resolution=resolution, seed=seed)
    if verbose:
        print(f"[cluster] {lab.max() + 1} clusters (resolution {resolution}) | {time.time() - t0:.0f}s")
    return a


def refine(a, tab: pd.DataFrame, clusters: Sequence[str], resolution: float = 1.0, key: str = "cluster",
           verbose: bool = True, **annotate_kw) -> pd.DataFrame:
    """Re-split only chosen clusters in the integrated embedding ('6.0', '6.1' …) and re-annotate everything.

    Use when a rare type is buried in a large cluster (e.g. Treg within a CD4 T cluster — FOXP3 cells are only part of the cluster
    and do not pass the RNA_REQUIRED threshold). Pick candidate clusters with cluster_markers / plot_marker_dots.
    """
    import codex_types as cty

    p = a.uns.get("cluster", {})
    cl = a.obs[key].astype(str).to_numpy().copy()
    rep = np.asarray(a.obsm[p.get("rep", "X_integrated")])
    for c in map(str, clusters):
        ii = np.where(cl == c)[0]
        if len(ii) < 50:
            continue
        sub, _, _ = cty._leiden(rep[ii], None, p.get("n_neighbors", 15), resolution, p.get("seed", 0))
        cl[ii] = [f"{c}.{s_}" for s_ in sub]
    a.obs[key] = pd.Categorical(cl, categories=sorted(set(cl), key=cty._ckey))
    if verbose:
        print(f"[cluster] re-split clusters {list(map(str, clusters))} → {len(set(cl))} in total")
    return annotate(a, key=key, verbose=verbose, **annotate_kw)


def _group_means(a, key: str, genes: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster × gene mean lognorm and fraction of expressing cells."""
    L = a.layers["lognorm"]
    cols = np.arange(a.n_vars) if genes is None else np.array([a.var_names.get_loc(g) for g in genes])
    grp = a.obs[key].astype(str).to_numpy()
    cats = sorted(set(grp), key=lambda c: tuple(int(x) if x.isdigit() else 10**9 for x in c.split(".")))
    G = sp.csr_matrix((np.ones(len(grp)), (pd.Categorical(grp, categories=cats).codes, np.arange(len(grp)))),
                      shape=(len(cats), len(grp)))
    n = np.asarray(G.sum(1)).ravel()[:, None]
    Lc = L[:, cols]
    mean = np.asarray((G @ Lc).todense()) / n
    frac = np.asarray((G @ (Lc > 0).astype(np.float32)).todense()) / n
    names = a.var_names[cols]
    return pd.DataFrame(mean, index=cats, columns=names), pd.DataFrame(frac, index=cats, columns=names)


def cluster_markers(a, key: str = "cluster", n: int = 8, min_frac: float = 0.25) -> pd.DataFrame:
    """Per cluster, top genes by mean lognorm difference vs the rest (fraction of expressing cells ≥ min_frac)."""
    M, F = _group_means(a, key)
    sizes = a.obs[key].astype(str).value_counts()
    tot = (M.mul(sizes.reindex(M.index), axis=0)).sum(0)
    out = {}
    for c in M.index:
        rest = (tot - M.loc[c] * sizes[c]) / max(a.n_obs - sizes[c], 1)
        d = (M.loc[c] - rest)[F.loc[c] >= min_frac].nlargest(n)
        out[c] = ", ".join(f"{g} {v:.1f}" for g, v in d.items())
    return pd.Series(out, name="top_genes")


def annotate(a, key: str = "cluster", markers: Optional[Dict[str, List[str]]] = None,
             negatives: Optional[Dict[str, List[str]]] = None, required: Optional[Dict[str, List[str]]] = None,
             min_score: float = 0.5, min_margin: float = 0.3,
             min_frac: float = 0.2, neg_weight: float = 0.7, verbose: bool = True) -> pd.DataFrame:
    """Per cluster RNA_MARKERS scores → obs['codex_type'], obs['lineage'].

    score = mean of the top half of the type's markers after converting cluster-mean lognorm to z across clusters.
           But markers expressed in fewer than min_frac of the cluster's cells are set to z = −1. For types almost absent from the sample
           (e.g. smooth muscle in a B-dominated tonsil), between-cluster variance of their markers is small, so noise alone inflates z.
    score < min_score → 'Unknown',  1st·2nd gap < min_margin with different lineages → 'A|B?'
    negatives  : per type, markers that must be absent (RNA_NEGATIVE). neg_weight × the max z of those markers (positive only) is subtracted from the score
    required   : per type, markers that must be present (RNA_REQUIRED, e.g. Treg ← FOXP3). Score −1 if the expressing fraction < min_frac
    high_doublet column: cluster-mean doublet score above the 90th percentile of all cell scores — candidate remaining doublet clumps
    (in scRNA, 'A|B?' is often a clump of doublets of two different types)
    """
    markers = markers or RNA_MARKERS
    negatives = RNA_NEGATIVE if negatives is None else negatives
    required = RNA_REQUIRED if required is None else required
    lin = _lineage_of()
    genes = sorted({g for gs in list(markers.values()) + list(negatives.values()) + list(required.values())
                    for g in gs if g in a.var_names})
    M, F = _group_means(a, key, genes)
    Z = ((M - M.mean()) / (M.std() + 1e-6)).where(F >= min_frac, -1.0)
    S = {}
    for t, gs in markers.items():
        g = [x for x in gs if x in Z.columns]
        if g:
            # markers missing from the data (filtered genes) are filled as unexpressed (−1) to keep k based on the full marker count
            V = np.hstack([Z[g].to_numpy(), np.full((len(Z), len(gs) - len(g)), -1.0)])
            kk = int(np.ceil(len(gs) / 2))
            S[t] = pd.Series(np.sort(V, 1)[:, ::-1][:, :kk].mean(1), index=Z.index)
            ng = [x for x in negatives.get(t, []) if x in Z.columns]
            if ng:
                S[t] = S[t] - neg_weight * Z[ng].clip(lower=0).max(axis=1)
            rq = required.get(t, [])
            if rq:
                ok = pd.Series(True, index=Z.index)
                for g_ in rq:
                    ok &= (F[g_] >= min_frac) if g_ in F.columns else False
                S[t] = S[t].where(ok, -1.0)
    S = pd.DataFrame(S)
    rows = {}
    for c in S.index:
        s = S.loc[c].sort_values(ascending=False)
        t1, t2, mg = s.index[0], s.index[1], s.iloc[0] - s.iloc[1]
        if s.iloc[0] < min_score:
            label = "Unknown"
        elif mg < min_margin and lin.get(t1, t1) != lin.get(t2, t2):
            label = f"{t1}|{t2}?"
        else:
            label = t1
        rows[c] = dict(label=label, score=round(float(s.iloc[0]), 2), second=f"{t2} {s.iloc[1]:.2f}",
                       margin=round(float(mg), 2))
    tab = pd.DataFrame(rows).T
    tab.index.name = key
    tab.insert(0, "n_cells", a.obs[key].astype(str).value_counts().reindex(tab.index).to_numpy())
    if "doublet_score" in a.obs:
        tab["doublet_score"] = a.obs.groupby(a.obs[key].astype(str), observed=True)["doublet_score"].mean() \
            .reindex(tab.index).round(3)
        tab["high_doublet"] = tab["doublet_score"] > np.nanquantile(a.obs["doublet_score"], 0.9)
    tab["top_genes"] = cluster_markers(a, key).reindex(tab.index)
    # cluster × type score table. Kept out of DataFrame.attrs because pandas 2.2 compares attrs with == when printing·concatenating and errors
    a.uns["type_scores"] = S.round(2)
    _apply(a, tab, key)
    if verbose:
        vc = a.obs["codex_type"].value_counts()
        print(f"[annotate] " + ", ".join(f"{k} {v:,}" for k, v in vc.items()))
    return tab


def _apply(a, tab: pd.DataFrame, key: str):
    lin = _lineage_of()
    lab = a.obs[key].astype(str).map(tab["label"].astype(str))
    a.obs["codex_type"] = pd.Categorical(lab)
    a.obs["lineage"] = pd.Categorical(lab.map(lambda x: lin.get(x, "Unknown" if x == "Unknown" else "Uncertain")))
    a.uns["cluster_types"] = tab["label"].astype(str).to_dict()


def set_cluster_types(a, tab: pd.DataFrame, fixes: Dict[str, str], key: str = "cluster") -> pd.DataFrame:
    """Fix cluster types directly (e.g. {'7': 'Plasma', '12': 'Unknown'}). Use 'Unknown' to leave it out of mapping."""
    tab = tab.copy()
    for c, t in fixes.items():
        if str(c) not in tab.index:
            raise KeyError(f"No cluster '{c}'")
        tab.loc[str(c), "label"] = t
    _apply(a, tab, key)
    return tab


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def embed_2d(a, rep: str = "X_integrated", seed: int = 0, key_added: str = "X_umap"):
    """Two-dimensional coordinates for plotting (UMAP if umap-learn is available, otherwise PCA 1–2)."""
    try:
        import umap
        a.obsm[key_added] = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(a.obsm[rep])
    except ImportError:
        a.obsm[key_added] = np.asarray(a.obsm[rep])[:, :2]
    return a


def plot_barcode_ranks(curves: Sequence[Dict], ncols: int = 3):
    import matplotlib.pyplot as plt

    nr = int(np.ceil(len(curves) / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(4 * ncols, 3.2 * nr), squeeze=False)
    for ax, c in zip(axes.ravel(), curves):
        ax.loglog(c["rank"], c["umi"], lw=1, color="k")
        ax.axhline(c["threshold"], color="r", lw=0.8, label=f"cells {c['n_cells']:,} (≥ {c['threshold']:.0f})")
        ax.axhline(c["knee"], color="b", lw=0.6, ls="--", label=f"knee {c['knee']:.0f}")
        ax.set_title(c["sample"], fontsize=9)
        ax.set_xlabel("barcode rank")
        ax.set_ylabel("UMI")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(curves):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_qc(a, sample_key: str = "sample"):
    import matplotlib.pyplot as plt

    cols = [("n_umi", True), ("n_genes", True), ("pct_mt", False)]
    samples = list(a.obs[sample_key].astype(str).unique())
    fig, axes = plt.subplots(1, 3, figsize=(4.5 * 3, 3.5))
    for ax, (c, log) in zip(axes, cols):
        data = [a.obs.loc[a.obs[sample_key].astype(str) == s, c].to_numpy() for s in samples]
        if log:
            data = [np.log10(d + 1) for d in data]
        ax.violinplot(data, showmedians=True)
        fail = [a.obs.loc[(a.obs[sample_key].astype(str) == s) & ~a.obs["qc_pass"], c].to_numpy() for s in samples]
        for i, f in enumerate(fail):
            f = np.log10(f + 1) if log else f
            ax.scatter(np.full(len(f), i + 1) + np.random.uniform(-0.15, 0.15, len(f)), f, s=1, c="r", alpha=0.3)
        ax.set_xticks(range(1, len(samples) + 1), samples, rotation=45, ha="right", fontsize=8)
        ax.set_title(("log10 " if log else "") + c + " (red = QC fail)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_doublets(a):
    import matplotlib.pyplot as plt

    sims = a.uns.get("doublet_sim", {})
    fig, axes = plt.subplots(1, len(sims), figsize=(3.2 * len(sims), 2.8), squeeze=False)
    for ax, (s, d) in zip(axes.ravel(), sims.items()):
        o = a.obs.loc[a.obs["sample"].astype(str) == s, "doublet_score"].dropna()
        ax.hist(np.asarray(d["sim"]), bins=50, density=True, alpha=0.5, label="simulated")
        ax.hist(o, bins=50, density=True, alpha=0.5, label="observed")
        ax.axvline(float(d["threshold"]), color="r", lw=0.8)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("doublet score")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_embedding(a, color: str, basis: str = "X_umap", s: float = 1.0, figsize=(6, 5)):
    import matplotlib.pyplot as plt

    xy = a.obsm[basis]
    v = a.obs[color]
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(v.dtype, pd.CategoricalDtype) or v.dtype == object:
        v = v.astype(str)
        cats = sorted(v.unique())
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(cats):
            m = (v == c).to_numpy()
            col = (0.8, 0.8, 0.8) if c == "Unknown" or c.endswith("?") else cmap(i % 20)
            ax.scatter(xy[m, 0], xy[m, 1], s=s, color=col, label=f"{c} ({m.sum():,})", linewidths=0, rasterized=True)
        ax.legend(markerscale=6, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=v.to_numpy(), cmap="viridis", linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.04)
    ax.set_title(color)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_marker_dots(a, key: str = "cluster", markers: Optional[Dict[str, List[str]]] = None,
                     types: Optional[Sequence[str]] = None, figsize=None):
    """Cluster × type marker dot plot (dot size = fraction of expressing cells, colour = mean lognorm scaled 0–1 per column)."""
    import matplotlib.pyplot as plt

    markers = markers or RNA_MARKERS
    types = types or list(markers)
    genes = list(dict.fromkeys(g for t in types for g in markers[t][:3] if g in a.var_names))
    M, F = _group_means(a, key, genes)
    Ms = (M - M.min()) / (M.max() - M.min() + 1e-9)
    fig, ax = plt.subplots(figsize=figsize or (0.28 * len(genes) + 2.5, 0.28 * len(M) + 1.2))
    yy, xx = np.meshgrid(range(len(M)), range(len(genes)), indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=(F.to_numpy().ravel() * 60), c=Ms.to_numpy().ravel(), cmap="Reds",
               edgecolors="none")
    ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=7)
    lab = a.uns.get("cluster_types", {})
    ax.set_yticks(range(len(M)), [f"{c}: {lab.get(c, '')}" for c in M.index], fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
