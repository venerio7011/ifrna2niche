# CODEX cell types + scRNA-seq preprocessing → per-type mapping (Colab)

Sets cell types from a CODEX cell table (marker intensities + x, y). Cleans·annotates the scRNA-seq raw matrix from RNA alone, then maps scRNA expression onto CODEX cell positions within the same type.

![CODEX types, scRNA positions and niche remapping routes](docs/img/00_cover.png)

*Human tonsil (STELLAR CODEX, 44 markers + GSE165860 scRNA-seq, 6 samples). **(a)** CODEX cell types set from markers. **(b)** scRNA cells placed at the position of their closest MaxFuse match. **(c)** Niche remapping: for each adjacent (receiver ← sender) pair, the route that passed judgement (blue = NicheNet) and the best gain on held-out signal-response proteins. Remapped expression is a **hypothesis** to test, not a measurement.*

---

## Results at a glance

The figures and numbers below come from a Colab run of the notebook. The Colab run of section 10 used a 30,000-cell CODEX subset (`MF_SUBSAMPLE`); the full 174k-cell MaxFuse result from a local run is shown next to it, marked *local*.

Evidence levels: **A** = measured (CODEX protein), **B** = mapping between datasets (benchmarkable), **C** = within-type remapping (hypothesis).

### ① CODEX cell types — sections 4, 6 · evidence A

<img src="docs/img/01_cluster_profiles.webp" width="420" align="right" alt="Cluster mean marker profiles">

Cells are clustered first (Leiden) and each cluster is named from its mean marker profile (right; rows = clusters, sorted by type). Clusters with a weak score stay `Unknown`, and clusters split between two lineages are marked `A|B?` for a person to check, instead of forcing a label.

![CODEX cell types](docs/img/02_codex_types.png)

![Agreement with given labels](docs/img/03_given_label_agreement.png)

Agreement with the labels published with the dataset, at lineage level (ARI 0.50): epithelium 78%, T 77%, stroma 74%, myeloid 71%, plasma 69%, B 53%. Most disagreement goes to `Unknown` rather than to a wrong lineage. The given labels were also assigned by people, so this checks for large disagreements rather than accuracy.

<br clear="right">

### ② scRNA-seq preprocessing — section 8-5 · evidence A

<img src="docs/img/04_scrna_integration.png" width="420" alt="scRNA embedding coloured by sample">

After cell calling, QC and doublet removal, the 6 tonsil samples are integrated with Harmony. Samples are mixed within every cluster (sample mixing score 0.82 → 0.96, donor 0.57 → 0.90), so clusters reflect cell types rather than samples. Types are named with the same vocabulary as CODEX.

### ③ Does mapping carry information beyond the type mean? — section 9-2 · evidence A ↔ B

<img src="docs/img/05_heldout_gain.png" width="360" align="right" alt="Held-out marker gain">

Each linked marker is held out, scRNA is mapped onto CODEX without it, and the mapped RNA is correlated with the held-out protein within each type. Colour = gain over a mapping that only matches the type mean (rows shuffled within type). Gains are clear for lineage-level markers (CD45, CD20, HLA-DR in B and plasma cells) but close to 0 for most markers; across 42 held-out markers the median within-type r is 0.021 (shuffled 0.000), only 19% of marker × type cells reach gain > 0.1, and the benchmark's median marker gain is **0.024**. Within-type differences in the mapped RNA are therefore mostly not supported, and position-dependent RNA differences are interpreted only where this map is red.

<br clear="right">

### ④ MaxFuse: label transfer and scRNA positions — section 10 · evidence B

![MaxFuse labels on CODEX](docs/img/06_maxfuse_labels.png)

MaxFuse matches scRNA and CODEX cells from the two datasets alone, without CODEX types. Transferred scRNA labels (above) and scRNA cells placed at their closest match (cover b) follow the B-cell follicles and T-cell zones. Types absent from the scRNA (granulocytes, stroma, epithelium) are still forced into matches — mostly B — so labels are filtered with the agreement table, and in the full-tissue run one scRNA cell can match several distant CODEX cells.

### ⑤ Niche remapping and judgement — sections 11, 12 · evidence C

For each adjacent type pair, scRNA cells within the receiver type are re-placed so that cells with a high NicheNet target score for the sender's ligands sit next to the sender (cover c). Each hypothesis is judged by whether held-out signal-response proteins (PD-1, PD-L1, IDO, CD25 …) match the remapped RNA better than when the hypothesis is placed at random.

| Receiver ← sender | Route | Best protein | Max gain | p | BH q |
|---|---|---|---|---|---|
| CD4 T ← CD8 T | nichenet | PD-1 | 0.25 | 0.048 | 0.10 |
| Treg ← CD4 T | nichenet | IDO | 0.19 | 0.048 | 0.10 |
| CD4 T ← Treg | nichenet | PD-1 | 0.16 | 0.048 | 0.10 |
| CD8 T ← plasma | nichenet | PD-1 | 0.11 | 0.048 | 0.10 |
| CD8 T ← CD4 T | nichenet | CD25 | 0.08 | 0.048 | 0.10 |
| 6 more pairs | similarity / nichenet | – | 0.00–0.17 | 0.10–1.00 | ≥ 0.17 |

### ⑥ Benchmark — section 12

| Output | Test | Value | How to read |
|---|---|---|---|
| RNA map | held-out markers, median marker gain (kNN) | 0.024 | above the type-mean mapping, but small |
| niche remapping | passed tests | 5 of 11 | chance expectation 0.55 |
| niche remapping | mean excess over random placement (all 11 tests) | −0.037 | **not better than random placement overall** |
| niche remapping | tests passing BH q ≤ 0.1 | 0 | all passes sit at the p floor 1/(N_NULL+1) = 0.048 |
| niche remapping | baseline map, median ρ of validation proteins | 0.051 | protein-only map barely predicts these proteins |

Five pairs pass where about one would by chance, but averaged over every pair tried, hypothesis-based remapping is no better than random placement, and none survive multiple-testing correction at 20 nulls. The passing pairs (mostly T-cell PD-1/CD25/IDO responses) are candidates to test, and section 11 should be re-run with more nulls (≥ 200) before ranking them.

---

## Files

| File | Role |
|---|---|
| `codex_rna.ipynb` | main notebook (sections 0–12) |
| `codex_table.py` | CODEX csv → AnnData. Automatic column role inference, sample selection, parquet cache |
| `codex_types.py` | CODEX preprocessing → Leiden clusters → marker score annotation → re-splitting mixed clusters → exclusive-marker flags, comparison with given labels |
| `rna_prep.py` | scRNA raw → cell calling → QC → doublets → normalisation → HVG → Harmony → clusters → types (RNA only) |
| `rna_map.py` | protein↔gene links, mapping plan, kNN / scMODAL mapping (`CellMap`), held-out marker validation. Replaces `rna_link.py` |
| `maxfuse_map.py` | MaxFuse cell matching → transfer scRNA labels to CODEX → tissue position per scRNA cell |
| `niche_remap.py` | type-mean mapping → within-type remapping by NicheNet (similarity as fallback) hypotheses → judged with CODEX signal-response proteins |
| `map_benchmark.py` | section 12 benchmark: comparing mapping methods by held-out marker tests (same folds, shared cells, per-marker tests) · niche remapping judgement summary (passes vs chance expectation, gain over random placement across all attempts) |

Drive: `MyDrive/CODEX_RNA/{modules, inputs, outputs, resources}`.

- `modules`: 6 py files
- `inputs`: the CODEX csv and 6 per-sample scRNA archives (`GSM…_Tonsil_1a_scRNA_seq_raw_feature_bc_matrix_mtx.tsv.tar.gz` etc.)
  - Uploading a single GEO RAW tar also works. Then change `RNA_INPUT` in section 8-1 to that file path.
  - Only files with `scRNA` in their name are read. CODEX csv files etc. in the same folder are ignored.
- `resources/nichenet`: NicheNet v2 human resources (about 260 MB downloaded from Zenodo once, the first time section 11 runs)

## Installation

Package list: [`requirements.txt`](requirements.txt). In Colab, section 0 installs it with Colab's own versions pinned:

```
pip install -c pins.txt anndata igraph leidenalg pynndescent harmonypy umap-learn maxfuse pyreadr
```

`pins.txt` lists the numpy·pandas·scipy versions already installed in Colab (created automatically by the second cell of section 0).

- Recent anndata (0.13) requires pandas ≥ 2.3. Without pinning, pip upgrades Colab's pandas 2.2.3 to 3.x and warns about conflicts with `google-colab` and `cudf`.
- With pinned versions, pip picks a matching anndata 0.12.x.
- If pandas was already upgraded in the runtime, use `Runtime → Disconnect and delete runtime` and start again.

scanpy is not used (avoids Colab's scanpy–scipy version conflict). scMODAL (optional) is cloned and added to the import path, with `annoy` installed (section 9-1).

## scRNA preprocessing (section 8, `rna_prep`)

| Step | Method | Result on 6 GSE165860 tonsil samples (Colab) |
|---|---|---|
| Extraction | flattens the 3 matrix files out of each sample archive (folder / file list / RAW tar all work) | 6 per-sample files, same result as the RAW tar |
| Cell calling | inflection point of the UMI rank curve | 6,231–8,377 per sample, 44,482 in total |
| QC | per-sample 3 MAD + floors (UMI 500, genes 250, MT 25%) | 39,899 / 44,482 passed (89.7%); median UMI 4,583, genes 1,781, MT 8.3% |
| Doublets | Scrublet-style score → the expected 10x count from the top | 2,514 (6.3%); 5.2–7.0% per sample vs 5.0–6.7% expected |
| Normalisation | CP10k → log1p (`layers['lognorm']`) | 37,385 cells × 23,338 genes |
| HVG | per-sample variance z → shared top 3,000 | 1,247 genes in the top in all 6 samples |
| Integration | PCA 30 → Harmony | sample mixing 0.82 → 0.96, donor mixing 0.57 → 0.90 |
| Types | Leiden (resolution 1.0, 20 clusters) → `RNA_MARKERS` scores | see below |

On the doublet results:

- 15.3% of called doublets co-express B+T markers, 38× the rate in singlets (0.4%).
- The Otsu threshold used at first called 15–25%. In samples dominated by B cells, B+B simulated doublets cannot be told from singlets. So the default was changed to the expected-count approach.

Type score rules:

- Uses the mean of the top half of between-cluster z.
- Markers expressed in fewer than 20% of cells are set to −1.
- Markers that must be absent (`RNA_NEGATIVE`) are penalised.
- If a required marker (`RNA_REQUIRED`, e.g. Treg ← FOXP3) is missing, the cluster is excluded from that type.

Tonsil result (Colab, no re-splitting needed — Treg came out as its own cluster):

| Type | Cells |
|---|---|
| B | 26,451 |
| CD8 T | 3,063 |
| Tfh | 1,954 |
| CD4 T | 1,487 |
| Plasma cells | 1,369 |
| Treg | 627 |
| Macrophages | 325 |
| pDC | 139 |
| Unknown | 1,970 |

**This scRNA has almost no granulocytes, stromal cells, endothelium or epithelium.** So those CODEX types (about 40% of tonsil cells) are not mapped.

## CODEX cell types (sections 2–6, `codex_types`)

- Cluster → annotate: marker scores are computed on cluster-mean profiles.
- Low scores become `Unknown`; similar 1st·2nd scores become `A|B?`.
- Clusters mixing two types are re-split by `refine`.

Tonsil result:

- 173,968 cells. `Unknown` is 17% (28,921 cells).
- Lineage agreement with the given labels (ARI 0.50): epithelium 78%, T 77%, stroma 74%, myeloid 71%, plasma 69%, B 53%. B disagreement goes mostly to `Unknown` (27%).

## Mapping (section 9, `rna_map`)

1. `mapping_plan`: maps a CODEX type only if there are ≥ 30 scRNA cells of the same type. `Unknown` and `A|B?` are excluded.
   - Lineage fallback (`allow_lineage=True`) is off by default. If on, for example, CODEX granulocytes would receive scRNA monocyte expression.
2. `map_cells`:
   - `knn`: within type, the 42 linked features are rank-normalised separately in each dataset, then kNN (k = 20).
   - `scmodal`: scRNA input = linked features + HVGs, CODEX input = 44 markers. Builds a shared embedding, then kNN within type.
3. `CellMap`: `impute(genes)`, `transfer(labels)`, `save/load`.
4. `holdout_check`: maps with markers left out and compares with matches shuffled within the same type.

**Limitation:** differences between cells in the mapped RNA reflect reality only as far as the panel proteins explain them. In tonsil kNN (42 held-out markers), the median within-type r was 0.021 against 0.000 for shuffled matches, and 19% of marker × type cells had gain > 0.1. The top markers were CD45 0.21, CD20 0.16, HLA-DR 0.13, CD3 0.12, CD38 0.11. Interpret per-niche RNA comparisons only for types·genes with gain clearly above 0.

## MaxFuse label transfer · scRNA positions (section 10, `maxfuse_map`)

Finds scRNA ↔ CODEX cell matches from the two datasets alone, without CODEX types. Uses the values of the MaxFuse tonsil tutorial.

- Steps: shared features + cell graphs → initial matches → refine with CCA → drop bad matches → propagate to the rest
- Shared features: 28 of the 49 protein–gene pairs that vary in both datasets

Changes from stock MaxFuse:

- `scanpy.pp.neighbors`, called by the graph step, breaks with recent scipy/anndata. Replaced by a function doing the same computation with a pynndescent + UMAP fuzzy graph.
- Running Leiden to convergence took 85% of total time. Limiting it to 2 iterations made it 3× faster with nearly the same result (type agreement 36.2% → 35.7%).
- Batching scheme changed to `cyclic`.

Tonsil result:

| | Colab (CODEX 30,000-cell subset) | *local* (all 173,968 CODEX cells, CPU 18 min) |
|---|---|---|
| CODEX cells with a match | 26,612 (89%) | 144,856 (83%) |
| transferred label = CODEX marker type | type 34.9%, lineage 42.8% | type 39%, lineage 46% |
| agreement for types present in scRNA | – | B 92%, Tfh 69%, CD4 T 60%, CD8 T 67%, Treg 47% |
| types absent or rare in scRNA | mostly matched to B | granulocytes·stroma·epithelium·plasma cells mostly matched to B |
| scRNA cells given a position | 8,312 / 37,385; 3,476 using same-type matches only | 17,895 / 37,384; 8,561 using same-type matches only |
| matches per scRNA cell | median 1 | median 3, max 461 |
| spread of matches | median 0 (single match) | median about 2,000 px |

With a subset, most scRNA cells get a single match, so spread is 0; with all CODEX cells, matches multiply and spread widely.

Caveats for interpretation:

- Types absent from scRNA are still forced into matches. Filter with the `consistent` column of the match table or `consistent_only=True`.
- One scRNA cell matches several CODEX cells, and their positions spread widely. For example, one B cell matches several follicles. So the weighted centre (`x, y`) may be where no cell actually is. Look at the closest match (`best_x, best_y`) or the whole match list (`pairs.csv.gz`) for position.
- CODEX plasma cells (CD38⁺) were matched to B (89%) rather than scRNA plasma cells. Shared features that distinguish plasma cells are weak in this panel.

## Niche remapping (section 11, `niche_remap`)

Keeps the type-mean mapping as the evidence result and builds a hypothesis map that remaps scRNA cells within types by signals from adjacent types. Each hypothesis is judged with CODEX protein.

1. **Type-mean mapping** (`mean_impute`): per CODEX cell, the mean of scRNA cells of the same type. No within-type differences.
2. **Adjacent pairs** (`adjacency_enrichment`): fraction of the sender type among 10 spatial neighbours compared with a label-shuffled expectation. Only pairs with ratio ≥ 1.2× and z ≥ 2, at most 6 per receiver type.
3. **NicheNet** (`load_nichenet`, `receiver_geneset`, `ligand_activity`):
   - Genes of interest: differential genes of substates from re-splitting the receiver type's scRNA. Type markers·mitochondrial·ribosomal·cell-cycle genes are removed.
   - Candidate ligands: expressed by ≥ 10% of sender-type cells, with the receptor expressed by the receiver type.
   - Ligand ranking: AUPR − base rate (NicheNet v2 default).
   - Target score: for each top ligand (at most 5), top-250 regulatory potential ∩ genes of interest ∩ genes detected in the receiver type. Used only if ≥ 10. A per-cell module score with per-sample means removed.
4. **Similarity route** (fallback): in HVG (without sender-type markers) → PCA → Harmony space, negative mean distance to the 10 nearest sender-type cells. The top 10% of scRNA cells by doublet score are excluded from candidates. Not used within a lineage (T with T) or for B↔plasma cell pairs.
5. **Remapping**:
   - Candidates: the 200 closest scRNA cells in protein space.
   - Weights: protein kernel × Π exp(−(q − p)² / 2σ²), σ = 0.15. Here q is the rank of how close the CODEX cell is to the sender type (distance to the nearest 3), and p the rank of the scRNA cell score.
   - After balancing cell use (relaxed Sinkhorn), only the top 30 are kept.
   - Whole cell profiles are moved, not per-gene orderings. So correlations between genes are preserved.
6. **Judgement**:
   - Remaps with the judging proteins PD-L1, IDO, HLA-DR, COX2, CD25, PD-1, CD38, BCL2 removed from the protein features.
   - Compares the Spearman between same-type spatial-neighbour-averaged remapped gene values and protein against the baseline (σ = ∞).
   - null: shuffle the CODEX-side hypothesis among receiver cells. p from the max statistic over the 8 proteins.
   - Pass condition: p ≤ 0.05 and gain ≥ 0.02.
7. **Route decision**:
   - If NicheNet passes, `nichenet`.
   - If NicheNet is not possible or fails, similarity is tried. If it passes, `similarity`, or `similarity_weak` if NicheNet failed judgement first.
   - If both fail, `none`.
   - If the sender type is absent from scRNA, `none (sender_absent)`. The similarity route also needs scRNA cells of the sender type.

Tonsil result (Colab, 20 nulls; 11 pairs judged across both routes):

| Receiver ← sender | Route | Most improved judging protein | Max gain | p | Pass |
|---|---|---|---|---|---|
| CD4 T ← CD8 T | nichenet | PD-1 | 0.25 | 0.048 | ✓ |
| Treg ← CD4 T | nichenet | IDO | 0.19 | 0.048 | ✓ |
| CD4 T ← Treg | nichenet | PD-1 | 0.16 | 0.048 | ✓ |
| CD8 T ← plasma cells | nichenet | PD-1 | 0.11 | 0.048 | ✓ |
| CD8 T ← CD4 T | nichenet | CD25 | 0.08 | 0.048 | ✓ |
| Plasma cells ← CD8 T | similarity | PD-1 | 0.08 | 0.095 | |
| CD8 T ← Treg | nichenet | HLA-DR | 0.07 | 0.19 | |
| Plasma cells ← CD8 T | nichenet | COX2 | 0.06 | 0.24 | |
| Tfh ← B | similarity | BCL2 | 0.17 | 0.33 | |
| Treg ← CD8 T | nichenet | PD-L1 | 0.05 | 0.86 | |
| Tfh ← B | nichenet | IDO | 0.00 | 1.00 | |
| other adjacent pairs | none | | | | sender type absent from scRNA (FDC, stromal cells, epithelium, granulocytes, endothelium, NK) |

Caveats for interpretation:

- **Passing judgement means 'the remapping matches the protein spatial pattern better', not that the ligand is the cause.**
  - The passing pairs are mostly T-cell activation readouts (PD-1, CD25, IDO) between neighbouring T subsets, which may reflect T-zone activation states placed together rather than a specific ligand.
  - Check the ligand list of each passing pair (`res.routes['ligands']`): low-specificity ligands (HLA, PTPRC, integrins) are often ranked high.
- **A passing hypothesis can make other protein patterns worse.** Mean excess over random placement is negative for 3 of the 5 passing pairs (−0.04 to −0.06), so the gain is protein-specific. Check per protein in the `validation` table.
- **The null controls for 'random placement'.** It is not compared against 'any spatially structured hypothesis'.
- **The smallest p is 1/(number of nulls + 1).** With 20 nulls that is 0.048, so every passing pair sits at this floor.
  - With 11 tests, BH q ≤ 0.1 needs p ≤ 0.009, which 20 nulls cannot reach (all q = 0.10). Re-run with ≥ 200 nulls before ranking pairs.
  - Averaged over all 11 tests, the gain over random placement is −0.037 (section 12), so remapping is not better than random placement overall.
- **Reproducibility is weak.** An earlier local run (different scRNA clustering; Treg not split out) passed B ← Tfh (BCL2 +0.40), CD4 T ← CD8 T, CD8 T ← plasma cells and Tfh ← B. In the Colab run, B ← Tfh was not judged and Tfh ← B failed. Only CD4 T ← CD8 T (PD-1) and CD8 T ← plasma cells passed in both, so those are the most robust candidates.
- **The most important tonsil signal, FDC → B (CXCL13, BAFF), cannot be computed.** The scRNA has no FDCs. A reference scRNA with stromal cells is needed.

## Validation status

- Ran notebook sections 0–10 locally end to end with the real CODEX csv and the real GSE165860 tar (except the scMODAL cells).
- Section 11 was run locally with the section 0 settings cell and saved section 8 results (20 nulls, about 90 s).
- Re-ran all of sections 0–11 without errors with the same versions as Colab (pandas 2.2.3, anndata 0.12.19, scipy 1.16.3, numpy 2.1.3 — Colab's 2.0.2 has no build for local Python 3.13, so the closest version) (except the scMODAL cells).
  - Section 10 was run with the notebook default of 30 k CODEX cells (9 min).
  - MaxFuse on all 170 k cells was run separately outside the notebook (18 min).
- Not yet tested in Colab.
- scMODAL route:
  - Tested only 300 steps on CPU with the scMODAL code provided by the user. One training run took about 13 min (about 2.6 s per step). The default 10,000 steps would take over 7 hours on CPU, so a GPU is needed.
  - The 300-step result was worse than kNN. CODEX cells collapsed onto only 3% of scRNA B cells, and within-type gain was about 0.
  - There is no fully trained result yet. Until then, `knn` is the default.

## License

The code in this repository is released under the [MIT License](LICENSE). Packages and datasets the pipeline installs or downloads keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
