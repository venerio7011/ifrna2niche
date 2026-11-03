# Third-party software and data

The MIT license in `LICENSE` covers the code in this repository only. The pipeline installs and calls the packages below at run time and downloads the data below; none of them are redistributed here. Each keeps its own license. Licenses were checked when this file was written; confirm them on each project's page before redistributing anything.

## Software

| Package | Used for | License |
|---|---|---|
| MaxFuse | cell matching, section 10 | MIT |
| scMODAL (optional) | shared embedding, section 9-1 | see github.com/gefeiwang/scMODAL |
| harmonypy | scRNA integration, section 8-4 | GPL-3.0 |
| leidenalg | clustering | GPL-3.0 |
| python-igraph | clustering | GPL-2.0-or-later |
| pynndescent, umap-learn | kNN graphs | BSD |
| pyreadr | reading NicheNet .rds files, section 11 | AGPL-3.0 |
| anndata | data handling | BSD 3-Clause |
| numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow | general | BSD-style / Apache 2.0 |

## Data and resources

| Resource | Used for | Terms |
|---|---|---|
| STELLAR CODEX tables (Brbić et al., Nat Methods 2022), Dryad doi:10.5061/dryad.g4f4qrfrc | CODEX input | CC0 (Dryad) |
| GEO GSE165860 tonsil scRNA-seq | scRNA input | public GEO data; cite the original study |
| NicheNet v2 prior model (Zenodo 7074291) | ligand–target priors, section 11 | see the Zenodo record |

## Citing

If you use the pipeline, please also cite the methods it builds on: MaxFuse (Chen et al., Nat Biotechnol 2023), NicheNet (Browaeys et al., Nat Methods 2020), Harmony (Korsunsky et al., Nat Methods 2019), Leiden (Traag et al., Sci Rep 2019), Scrublet (Wolock et al., Cell Syst 2019), STELLAR (Brbić et al., Nat Methods 2022), and scMODAL if used.
