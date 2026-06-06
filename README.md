# BERT Representation Probes

> (The git directory is still named `bert-cluster-stability` for history; clustering is now just sub-view 5.1 of a multi-probe series.)

Probe the **same** cached BERT document-segment representations with three different lenses, layer by layer, and ask why they disagree.

## The three probes

One representation, three rulers of different kinds:

- **5.1 Clustering** (unsupervised) — does topic-aligned structure *self-organize*?
- **5.2 Linear probe** (supervised, logistic regression) — is topic *linearly decodable*?
- **5.3 Fisher** (supervised geometry, no classifier trained) — how *separated* are the label groups (within- vs between-class scatter)?

## Unifying finding

> Topic information can live in **low-variance directions**. Methods that **reweight directions** (PCA whitening, LDA's `S_W⁻¹`, logistic-regression weights) can read it; variance-respecting geometry probes (naive clustering, the Fisher trace ratio) miss it. That is why random-init BERT looks empty to clustering yet sits well above chance to a linear probe — the topic signal is present, just buried in low-variance directions.

The clustering view (5.1) is documented in detail below; see **Supervised Linear Probe (Artifact 5.2 / 5.3)** for the probe views.

## Research Question (5.1 clustering view)

> Do BERT layerwise document-segment representations contain clustering structure that becomes more topic-aligned across layers?

- The object of study is **layerwise clustering structure** in BERT representations.
- KMeans/Lloyd-style methods are used as probes, not as the final research object.
- Subsampling stability is a robustness check, not the main phenomenon.
- The main semantic signal is measured by alignment with 20 Newsgroups topic labels.

## Current Setup

| Item | Value |
|---|---|
| Main model | `bert-base-uncased` |
| Control model | same BERT architecture with random initialization, seed=1 |
| Dataset | 20 Newsgroups, `remove=("headers", "footers", "quotes")` |
| Sample | pilot uses `n_docs=2000`, `sample_seed=42`, `min_chars=30` |
| Input granularity | document segment |
| Token length | first 512 WordPiece tokens |
| Token pooling | mean over non-padding token hidden states |
| Layers | embedding layer 0 + encoder layers 1..12 |
| Cached embedding shape | `(N, 13, 768)` |
| Primary alignment metrics | NMI and purity against 20NG labels |
| Geometry controls | silhouette, Davies-Bouldin, Calinski-Harabasz |
| Representation diagnostics | anisotropy, participation ratio |

## Current Best Recipe

The strongest pilot configuration so far is:

```text
layer12 + whiten100_l2 + spherical KMeans + K=20
```

Current pilot result:

- pretrained BERT L12: NMI around `0.446`, purity around `0.457`
- random-init BERT L12: NMI around `0.059`, purity around `0.121`

Interpretation:

> After PCA whitening and spherical KMeans, BERT L12 contains a topic-aligned clustering structure that is clearly absent from random-init BERT.

## Experiment Map

Each experiment is meant to isolate one variable while keeping the rest fixed.

### 1. Embedding Extraction

Script:

```powershell
.\.venv\Scripts\python.exe experiments\extract_embeddings.py
```

Purpose:

- Run pretrained BERT and random-init BERT.
- Extract all 13 layer representations.
- Cache embeddings so downstream experiments do not rerun BERT.

Outputs:

- `outputs/cache/pretrained_bert-base-uncased_n2000_s42_len512_min30.npz`
- `outputs/cache/random_seed1_bert-base-uncased_n2000_s42_len512_min30.npz`

### 2. Baseline Layer Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\run_pilot_metrics.py --seeds 0 1 2 3 4
.\.venv\Scripts\python.exe experiments\clustering\plot_pilot.py
```

Variable being tested:

```text
layer L = 0..12
```

Fixed choices:

```text
mode = l2
clusterer = Lloyd KMeans
K = 20
```

Main lesson:

- Pretrained BERT shows rising topic alignment in later layers.
- Random-init BERT remains near zero topic alignment.
- Geometry-only compactness metrics do not explain the semantic signal.

Outputs:

- `outputs/tables/clustering/pilot_metrics_l2_seeds.csv`
- `outputs/figures/pilot/pilot_alignment.png`
- `outputs/figures/pilot/pilot_diagnostics.png`
- `outputs/figures/pilot/pilot_geometry_controls.png`

### 3. Transform Sweep

Script:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_transforms.py
.\.venv\Scripts\python.exe experiments\clustering\plot_transform_sweep.py
```

Variable being tested:

```text
representation transform before clustering
```

Transforms compared:

- `raw`
- `l2`
- `centered_l2`
- `pca50_l2`, `pca100_l2`
- `whiten50_l2`, `whiten100_l2`
- `drop_pc1_l2`, `drop_pc3_l2`, `drop_pc5_l2`, `drop_pc10_l2`

Fixed choices:

```text
model = pretrained
layer = 12
K = 20
clusterer = Lloyd KMeans unless otherwise specified
```

Main lesson:

- PCA whitening helps the most.
- Removing top PCs does not help in this pilot.
- Whitening likely helps because it prevents a few high-variance axes from dominating clustering.

Outputs:

- `outputs/tables/clustering/transform_sweep.csv`
- `outputs/figures/transforms/transform_sweep_alignment.png`
- `outputs/figures/transforms/transform_sweep_geometry.png`

### 4. Clusterer Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_transforms.py --transforms l2 centered_l2 pca100_l2 whiten50_l2 whiten100_l2 --clusterers lloyd spherical --output outputs/tables/clustering/clusterer_sweep.csv
.\.venv\Scripts\python.exe experiments\clustering\plot_clusterer_sweep.py
```

Agglomerative control under the current best transform:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_transforms.py --layers 12 --transforms whiten100_l2 --clusterers lloyd spherical agglo_cosine agglo_ward gmm_diag gmm_full --models pretrained --output outputs/tables/clustering/clusterer_sweep_gmm.csv
.\.venv\Scripts\python.exe experiments\clustering\plot_clusterer_sweep.py --csv outputs/tables/clustering/clusterer_sweep_gmm.csv --filename clusterer_sweep_gmm_alignment.png
```

Variable being tested:

```text
clustering backend
```

Clusterers compared:

- `lloyd`: standard sklearn KMeans with `algorithm="lloyd"`, `n_init=20`
- `spherical`: cosine/spherical KMeans implemented in `src/clusterers.py`
- `agglo_cosine`: agglomerative clustering with cosine distance and average linkage
- `agglo_ward`: agglomerative clustering with Euclidean distance and Ward linkage
- `gmm_diag`: Gaussian mixture model with diagonal covariance
- `gmm_full`: Gaussian mixture model with full covariance

Fixed choices:

```text
model = pretrained
layer = 12
K = 20
selected transforms
```

Main lesson:

- Spherical KMeans alone is not a magic fix.
- Spherical KMeans works best after whitening.
- Best combination so far is `whiten100_l2 + spherical`.
- Agglomerative variants recover some topic signal but underperform KMeans-style probes in the current pilot.
- Ward linkage creates some very pure small clusters, but also very large mixed clusters, lowering NMI.
- GMM variants recover clear topic signal and outperform agglomerative clustering, but still do not beat spherical KMeans.

Outputs:

- `outputs/tables/clustering/clusterer_sweep.csv`
- `outputs/tables/clustering/clusterer_sweep_agglo.csv`
- `outputs/tables/clustering/clusterer_sweep_gmm.csv`
- `outputs/figures/transforms/clusterer_sweep_alignment.png`
- `outputs/figures/transforms/clusterer_sweep_agglo_alignment.png`
- `outputs/figures/transforms/clusterer_sweep_gmm_alignment.png`

### 5. Pooling Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_pooling.py
.\.venv\Scripts\python.exe experiments\clustering\plot_pooling_sweep.py
```

Variable being tested:

```text
which cached layer representation is used
```

Pooling variants:

- `layer12`
- `last4_mean`
- `last4_concat`

Fixed choices:

```text
transform = whiten100_l2
clusterer = spherical
K = 20
```

Main lesson:

- `layer12` remains best.
- Averaging or concatenating the last four layers does not improve topic alignment.
- The strongest topic-aligned signal appears concentrated in L12 rather than distributed evenly across the last four layers.

Outputs:

- `outputs/tables/clustering/pooling_sweep.csv`
- `outputs/figures/transforms/pooling_sweep_alignment.png`

### 6. K Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_k.py --models pretrained random
.\.venv\Scripts\python.exe experiments\clustering\plot_k_sweep.py
```

Variable being tested:

```text
number of clusters K
```

Values:

```text
K = 5, 10, 20, 50
```

Fixed choices:

```text
pooling = layer12
transform = whiten100_l2
clusterer = spherical
```

Main lesson:

- `K=10` captures coarse semantic categories.
- `K=20` best matches the 20NG topic label granularity by NMI.
- `K=50` increases purity but lowers NMI, suggesting over-splitting.

Outputs:

- `outputs/tables/clustering/k_sweep.csv`
- `outputs/figures/transforms/k_sweep_alignment.png`

### 7. Best-Recipe Layer Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_transforms.py --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 --transforms whiten100_l2 --clusterers spherical --models pretrained random --output outputs/tables/clustering/layer_sweep_best_recipe.csv
.\.venv\Scripts\python.exe experiments\clustering\plot_best_recipe_layer_sweep.py
```

Variable being tested:

```text
layer L = 0..12
```

Fixed choices:

```text
transform = whiten100_l2
clusterer = spherical
K = 20
```

Main lesson:

- Pretrained BERT has a three-stage pattern:
  - L0: weak lexical/topic signal
  - L1-L5: fast rise
  - L6-L9: plateau
  - L10-L12: second rise, with L12 strongest
- Random-init BERT remains flat near NMI `0.05-0.06`.

Outputs:

- `outputs/tables/clustering/layer_sweep_best_recipe.csv`
- `outputs/figures/transforms/best_recipe_layer_sweep.png`

### 8. Whitening Dimension Sweep

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\sweep_whitening_dims.py
.\.venv\Scripts\python.exe experiments\clustering\plot_whitening_dim_sweep.py
```

Variable being tested:

```text
PCA whitening dimension d
```

Values:

```text
d = 10, 20, 50, 100, 150, 200, 300, 500, 768
```

Fixed choices:

```text
pooling = layer12
clusterer = spherical
K = 20
models = pretrained, random-init
```

Main lesson:

- `d=100` is the current peak for NMI and purity.
- `d=50..200` is the useful range; `d=100` is the cleanest point in this pilot.
- Too few dimensions underfit the topic structure.
- Too many dimensions reintroduce noise; `d=500` and `d=768` collapse toward weak alignment.
- Random-init BERT stays near chance across all whitening dimensions.

Outputs:

- `outputs/tables/clustering/whitening_dim_sweep.csv`
- `outputs/figures/transforms/whitening_dim_sweep.png`

### 9. Cluster Interpretation

Script:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\interpret_clusters.py --layers 12 --k 20 --transform whiten100_l2 --clusterer spherical --seed 0
```

Purpose:

- Open the clustering black box.
- Plot cluster-topic heatmaps.
- Write per-cluster majority-topic summaries.
- Write c-TF-IDF keywords for each cluster.

Useful K comparison:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\interpret_clusters.py --layers 12 --k 10 --transform whiten100_l2 --clusterer spherical --seed 0
.\.venv\Scripts\python.exe experiments\clustering\interpret_clusters.py --layers 12 --k 20 --transform whiten100_l2 --clusterer spherical --seed 0
.\.venv\Scripts\python.exe experiments\clustering\interpret_clusters.py --layers 12 --k 50 --transform whiten100_l2 --clusterer spherical --seed 0
```

Main lesson:

- `K=10`: coarse semantic clusters, such as sports, vehicles, religion, computers.
- `K=20`: closest to 20NG topic granularity.
- `K=50`: many high-purity fragments, but the original topics are split too much.

Outputs:

- `outputs/figures/interpret/cluster_topic_heatmap_K10_whiten100_l2_spherical_L12.png`
- `outputs/figures/interpret/cluster_topic_heatmap_K20_whiten100_l2_spherical_L12.png`
- `outputs/figures/interpret/cluster_topic_heatmap_K50_whiten100_l2_spherical_L12.png`
- `outputs/interpret/cluster_summary_K*_whiten100_l2_spherical_L12.csv`
- `outputs/interpret/cluster_keywords_K*_whiten100_l2_spherical_L12.txt`

### 10. Resampling Stability

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\run_stability.py
.\.venv\Scripts\python.exe experiments\clustering\plot_stability.py
```

Variable being tested:

```text
80% subset-resampling robustness of the discovered partition
```

Recipes compared:

```text
baseline = layer12 + l2 + Lloyd KMeans + K=20
best     = layer12 + whiten100_l2 + spherical KMeans + K=20
```

Protocol:

- draw 50 without-replacement 80% subsets
- fit the full recipe on each subset
- predict labels for all 2000 documents
- compute pairwise adjusted Rand index across the 50 partitions

Main lesson:

- The best recipe improves topic alignment more than it improves stability.
- Pretrained baseline and pretrained best recipe have similar stability ARI, but the best recipe has much higher NMI and purity.
- Random-init baseline can be highly stable while remaining semantically meaningless.
- Therefore, resampling stability is a robustness probe, not a substitute for topic alignment.

Current pilot summary:

| model | recipe | stability ARI | resampled NMI |
|---|---:|---:|---:|
| pretrained | baseline | ~0.464 | ~0.367 |
| pretrained | best | ~0.450 | ~0.431 |
| random-init | baseline | ~0.640 | ~0.052 |
| random-init | best | ~0.225 | ~0.058 |

Outputs:

- `outputs/tables/clustering/stability_pilot.csv`
- `outputs/tables/clustering/stability_label_runs.npz`
- `outputs/figures/transforms/stability_alignment.png`

### 11. Synthetic Whitening Demo

Script:

```powershell
.\.venv\Scripts\python.exe experiments\clustering\whitening_demo.py
```

Purpose:

- Build a toy anisotropic mixture where true clusters live in low-energy signal directions.
- Add a high-variance nuisance direction unrelated to the true labels.
- Show that `L2 + Lloyd` can cluster by the nuisance geometry, while PCA whitening recovers the low-energy clusters.

Current toy result:

| space | ARI | NMI | anisotropy | PR |
|---|---:|---:|---:|---:|
| `l2` | ~0.001 | ~0.043 | ~0.893 | ~2.6 |
| `whiten_l2` | ~0.983 | ~0.969 | ~-0.002 | ~9.9 |

Main lesson:

- Whitening is not just a scoring trick in the BERT pipeline.
- In a controlled geometry, it can remove a dominant nuisance direction and reveal lower-energy cluster structure.
- This supports the geometric note, but does not prove the same mechanism is the only cause in BERT.

Outputs:

- `outputs/tables/clustering/whitening_demo.csv`
- `outputs/figures/transforms/whitening_demo.png`

## Supervised Linear Probe (Artifact 5.2 / 5.3)

A second view on the same cached embeddings: instead of asking what
structure clustering *finds* unsupervised, ask how much topic
information is *linearly readable* with supervision. 5.2 uses
multinomial logistic regression (convex optimum); 5.3 uses Fisher LDA
(Gaussian analytical optimum). Both go through one shared harness
(`src/probe.py`); the run script just switches the estimator.

Scripts:

```powershell
.\.venv\Scripts\python.exe experiments\probe\run_linear_probe.py                                  # 5.2 logreg
.\.venv\Scripts\python.exe experiments\probe\plot_linear_probe.py
.\.venv\Scripts\python.exe experiments\probe\run_linear_probe.py --probe lda --output outputs/tables/probe/lda_probe.csv   # 5.3 LDA
.\.venv\Scripts\python.exe experiments\probe\plot_linear_probe.py --csv outputs/tables/probe/lda_probe.csv --filename lda_probe_accuracy.png
```

Protocol: 5-fold stratified CV, `StandardScaler` inside the pipeline
(fit per fold, no leakage), accuracy + macro-F1, with chance (1/20)
and majority baselines.

Current pilot result (`n_docs=2000`, CV accuracy):

| | pretrained L0 | pretrained L12 | random-init L0 | random-init L12 |
|---|---:|---:|---:|---:|
| logistic regression (5.2) | ~0.563 | ~0.623 | ~0.380 | ~0.275 |
| Fisher LDA (5.3) | ~0.597 | ~0.636 | ~0.372 | ~0.283 |

Main lesson (the cross-view contrast with 5.1):

- Both supervised probes agree closely, so the curve is not an
  artifact of one estimator.
- Unlike clustering (where random-init sits near the floor), a linear
  probe reads topic information well above chance even from
  random-init BERT — and that readability *decays* with depth, while
  pretrained BERT *rises*. The topic signal is present in random-init;
  its anisotropic geometry just hides it from unsupervised clustering.

Outputs:

- `outputs/tables/probe/linear_probe.csv`, `outputs/tables/probe/lda_probe.csv`
- `outputs/figures/probe/linear_probe_accuracy.png`, `outputs/figures/probe/lda_probe_accuracy.png`

## What We Know So Far

Current working claims:

1. Pretrained BERT representations contain topic-aligned clustering structure; random-init BERT does not.
2. The signal is strongest in high layers, especially L10-L12.
3. PCA whitening makes the topic-aligned structure easier for clustering to read.
4. Spherical KMeans improves the whitened representation more than standard Lloyd KMeans.
5. The natural granularity in this pilot appears close to `K=20`, matching the number of 20NG labels.
6. Whitening dimension matters: `d=100` is currently best, while very high dimensions reintroduce noise.
7. Silhouette, Davies-Bouldin, and Calinski-Harabasz alone are not reliable indicators of semantic topic alignment.
8. Resampling stability is partially decoupled from semantic alignment: a stable partition can still be meaningless if it comes from random-init geometry.
9. A small synthetic demo supports the whitening story: when nuisance anisotropy dominates, whitening can recover low-energy cluster signal.

Short version:

> BERT L12 document-segment embeddings contain a topic-aligned organization that becomes much clearer after whitening and spherical KMeans; this organization is strongly different from random-init and is most visible near K=20.

## Next Candidate Experiments

Possible next steps:

1. Scale the narrowed confirmation from `n_docs=2000` pilot to the full 20NG dataset.
2. Add stretch domain separability with Wiki / Reddit / arXiv segments.
3. Polish the artifact note and SOP-ready paragraph.

## Repository Layout

Scripts are grouped by sub-artifact of the BERT representation-probes
umbrella. `extract_embeddings.py` stays at the top of `experiments/`
because the cache it produces is shared by every view.

```text
src/                          importable library code (shared)
  probe.py                    supervised linear-probe harness (5.2 + 5.3)
experiments/
  extract_embeddings.py       shared cache producer
  clustering/                 Artifact 5.1 (unsupervised KMeans view)
  probe/                      Artifact 5.2 logistic regression + 5.3 Fisher LDA
tests/                        unit tests
outputs/cache/                cached BERT layer embeddings, gitignored
outputs/tables/clustering/    5.1 CSV metrics and sweep results
outputs/tables/probe/         5.2 / 5.3 probe-accuracy CSVs
outputs/interpret/            cluster summaries and c-TF-IDF keywords
outputs/figures/pilot/        5.1 baseline pilot figures
outputs/figures/transforms/   5.1 sweep figures
outputs/figures/interpret/    5.1 cluster-topic heatmaps
outputs/figures/probe/        5.2 / 5.3 probe-accuracy figures
```

## Development

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current checked test status:

```text
64 passed
```
