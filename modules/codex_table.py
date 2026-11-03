"""
codex_table.py
==============

Module that converts a cell-level CODEX table (csv: one row per cell = marker intensities + x, y + annotations) into AnnData (for Google Colab).
It is an already segmented·quantified table rather than an image, so it goes straight to cell typing / niches / scRNA mapping without a Mesmer (segmentation) step.

e.g. STELLAR Dryad data (doi:10.5061/dryad.g4f4qrfrc)
    BE_Tonsil_l3_dryad.csv  : tonsil + Barrett's oesophagus. The CODEX source of the scMODAL tonsil example
    B004_training_dryad.csv : small intestine·colon (annotated),  B0056_unnanotated_dryad.csv : small intestine·colon (unannotated)

Column roles are inferred from names and values (check with describe); if wrong, set them directly via to_anndata arguments.
    marker  : numeric columns that are not coordinates·morphology·IDs → X (cell × marker)
    x, y    : coordinates → obsm['spatial'] (in the table's units; use um_per_px to convert to µm)
    obs     : string / categorical columns (cell type, sample, region …) and morphology columns (area etc.)

Order of use
---------
    import codex_table as ct
    df = ct.read_codex_csv(f"{WORK_DIR}/inputs/BE_Tonsil_l3_dryad.csv", cache_dir=f"{WORK_DIR}/resources")
    ct.describe(df)                                  # check column roles / sample names
    ad = ct.to_anndata(df, subset={"sample_name": "tonsil"})   # sample column name·value as shown by describe
    ad.write_h5ad(f"{WORK_DIR}/outputs/codex/tonsil_codex.h5ad")
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["read_codex_csv", "guess_roles", "describe", "to_anndata"]

_X_NAMES = ["x", "x_um", "xcorr", "x_centroid", "centroid_x", "x:x", "x.x", "pos_x", "cx"]
_Y_NAMES = ["y", "y_um", "ycorr", "y_centroid", "centroid_y", "y:y", "y.y", "pos_y", "cy"]
_MORPH = re.compile(r"^(area|size|volume|perimeter|eccentricity|solidity|extent|major_axis|minor_axis|"
                    r"orientation|z|z:z|tile.*|region.*|fov|cell_?size)$", re.I)
_ID = re.compile(r"^(unnamed: ?\d+|index|cell_?id|cellid|id|label|object_?id)$", re.I)


def read_codex_csv(path: str, cache_dir: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Read the csv. Numeric columns are downcast to float32; with cache_dir, a parquet cache is written so later reads are fast."""
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + ".parquet")
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            df = pd.read_parquet(cache)
            if verbose:
                print(f"[codex] using cache: {cache} ({len(df):,} cells × {df.shape[1]} columns)")
            return df
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == object:
            df[c] = df[c].astype("category")
    if cache:
        try:
            df.to_parquet(cache)
        except Exception as e:          # e.g. no pyarrow → only the cache is skipped
            print(f"[codex] parquet cache failed ({type(e).__name__}) — reading from csv only")
    if verbose:
        print(f"[codex] {os.path.basename(path)}: {len(df):,} cells × {df.shape[1]} columns "
              f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB)")
    return df


def _pick(cols: Sequence[str], names: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    return next((low[n] for n in names if n in low), None)


def guess_roles(df: pd.DataFrame, max_levels: int = 200) -> Dict[str, object]:
    """Infer column roles → dict(x, y, markers, obs, ids).

    Integer columns are treated as categories (region numbers etc.) if they have ≤ max_levels distinct values, and as IDs if all values differ.
    """
    cols = list(df.columns)
    x, y = _pick(cols, _X_NAMES), _pick(cols, _Y_NAMES)
    markers, obs, ids = [], [], []
    for c in cols:
        if c in (x, y):
            continue
        s = df[c]
        if _ID.match(str(c)):
            ids.append(c)
        elif not pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == bool:
            obs.append(c)
        elif _MORPH.match(str(c)):
            obs.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
            ids.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() <= max_levels:
            obs.append(c)
        else:
            markers.append(c)
    return dict(x=x, y=y, markers=markers, obs=obs, ids=ids)


def describe(df: pd.DataFrame, max_show: int = 12) -> Dict[str, object]:
    """Print the inferred column roles and value distributions of categorical columns (to check before choosing to_anndata arguments)."""
    r = guess_roles(df)
    print(f"{len(df):,} cells")
    print(f"coordinates : x = {r['x']}, y = {r['y']}"
          + ("" if r["x"] and r["y"] else "  ← not found: set with to_anndata(x=..., y=...)"))
    if r["x"] and r["y"]:
        xs, ys = df[r["x"]], df[r["y"]]
        print(f"         range x {xs.min():.1f}–{xs.max():.1f}, y {ys.min():.1f}–{ys.max():.1f}")
    print(f"{len(r['markers'])} markers: {', '.join(map(str, r['markers']))}")
    print(f"ID columns (dropped): {r['ids']}")
    for c in r["obs"]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > max_show * 4:
            print(f"obs  {c}: numeric (median {s.median():.2f})")
        else:
            vc = s.value_counts()
            more = f" … and {len(vc) - max_show} more" if len(vc) > max_show else ""
            print(f"obs  {c}: " + ", ".join(f"{k} ({v:,})" for k, v in vc.head(max_show).items()) + more)
    return r


def to_anndata(df: pd.DataFrame, markers: Optional[Sequence[str]] = None, x: Optional[str] = None,
               y: Optional[str] = None, obs: Optional[Sequence[str]] = None,
               subset: Optional[Dict[str, Union[str, Sequence[str]]]] = None, um_per_px: Optional[float] = None,
               drop_markers: Sequence[str] = ("DAPI", "Hoechst", "blank", "empty"), verbose: bool = True):
    """Table → AnnData (X = cell × marker float32, obs = annotations, obsm['spatial'] = x, y).

    subset     : keep only rows matching {column: value or list of values} (e.g. {"sample_name": "tonsil"})
    um_per_px  : factor to convert pixel coordinates to µm (None keeps the table's units, recorded in uns['spatial_unit'])
    drop_markers: markers without cell-type information, such as nuclear stains·blank channels (excluded if contained in the name, case-insensitive)
    """
    import anndata as ad

    r = guess_roles(df)
    x, y = x or r["x"], y or r["y"]
    if not (x and y):
        raise ValueError(f"Could not find coordinate columns. Look at the column names with describe(df) and set x=, y=. Columns: {list(df.columns)}")
    if subset:
        keep = np.ones(len(df), bool)
        for c, v in subset.items():
            if c not in df.columns:
                raise KeyError(f"No subset column '{c}'. Columns: {list(df.columns)}")
            vals = [v] if isinstance(v, str) or np.isscalar(v) else list(v)
            hit = df[c].isin(vals).to_numpy()
            if not hit.any():
                raise ValueError(f"'{c}' has no value {vals}. Available values: {list(df[c].unique()[:20])}")
            keep &= hit
        df = df.loc[keep]
    markers = list(markers or r["markers"])
    drop = [m for m in markers if any(d.lower() in str(m).lower() for d in drop_markers)]
    markers = [m for m in markers if m not in drop]
    obs_cols = list(obs if obs is not None else r["obs"])
    X = df[markers].to_numpy(np.float32)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        X = np.nan_to_num(X)
    obs_df = df[obs_cols].copy()
    for c in obs_cols:
        s = obs_df[c]
        if not pd.api.types.is_numeric_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.nunique() <= 200):
            obs_df[c] = obs_df[c].astype(str).astype("category")
    # if an original cell ID column exists without duplicates, use it as cell names (so the original table can be matched again)
    idc = next((c for c in r["ids"] if df[c].is_unique), None)
    names = df[idc].astype(str).to_numpy() if idc else np.arange(len(df)).astype(str)
    obs_df.index = pd.Index([f"c{s}" for s in names], name="cell")
    a = ad.AnnData(X=X, obs=obs_df, var=pd.DataFrame(index=pd.Index(map(str, markers), name="marker")))
    xy = df[[x, y]].to_numpy(np.float64)
    a.obsm["spatial"] = xy * um_per_px if um_per_px else xy
    a.uns["spatial_unit"] = "um" if um_per_px else "table units"
    a.uns["source"] = dict(x=x, y=y, subset={k: str(v) for k, v in (subset or {}).items()},
                           dropped_markers=drop, um_per_px=um_per_px)
    if verbose:
        print(f"[codex] AnnData: {a.n_obs:,} cells × {a.n_vars} markers"
              + (f" | dropped markers {drop}" if drop else "") + (f" | {n_nan:,} missing values → 0" if n_nan else "")
              + f" | obs: {obs_cols}")
    return a
