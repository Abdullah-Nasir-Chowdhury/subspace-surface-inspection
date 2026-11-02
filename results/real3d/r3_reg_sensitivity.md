# Registration sensitivity (round 4)

## A. Per-category object AUROC against the median registration residual (items_nominal_feat_all, 597 items)

| category | median residual mm | n | dentnet_nom_rec3 max / covered | dentnet_nom_rec3_norm max / covered | pnpp_nom_rec3_norm max / covered |
|---|---|---|---|---|---|
| seahorse | 0.028 | 50 | 0.66 / 0.97 | 1.00 / 0.97 | 0.97 / 0.97 |
| starfish | 0.028 | 50 | 0.70 / 0.80 | 0.81 / 0.84 | 0.82 / 0.82 |
| fish | 0.031 | 50 | 0.86 / 1.00 | 1.00 / 0.98 | 1.00 / 0.98 |
| toffees | 0.039 | 49 | 0.90 / 1.00 | 1.00 / 0.99 | 0.96 / 1.00 |
| car | 0.040 | 50 | 0.78 / 0.81 | 0.95 / 0.85 | 0.89 / 0.78 |
| candybar | 0.046 | 50 | 0.81 / 0.99 | 1.00 / 0.99 | 1.00 / 1.00 |
| gemstone | 0.139 | 50 | 0.59 / 0.69 | 0.63 / 0.69 | 0.65 / 0.69 |
| shell | 0.142 | 50 | 0.56 / 0.61 | 0.45 / 0.66 | 0.43 / 0.54 |
| chicken | 0.155 | 50 | 0.72 / 0.57 | 0.67 / 0.73 | 0.65 / 0.64 |
| duck | 0.207 | 50 | 0.84 / 0.90 | 0.55 / 0.57 | 0.69 / 0.69 |
| diamond | 0.224 | 48 | 0.73 / 0.91 | 0.86 / 0.82 | 0.87 / 0.82 |
| airplane | 0.233 | 50 | 0.49 / 0.60 | 0.45 / 0.60 | 0.52 / 0.62 |
| Spearman rho (12 categories) | | | -0.27 / -0.52 | -0.71 / -0.71 | -0.64 / -0.66 |

### Item-level AUROC within residual bins (pooled over categories; max / covered, 95 % item bootstrap)

| bin (mm) | n (defective) | dentnet_nom_rec3 | dentnet_nom_rec3_norm | pnpp_nom_rec3_norm |
|---|---|---|---|---|
| [0.00, 0.05) | 301 (146) | 0.76 [0.70, 0.81] / 0.93 [0.90, 0.96] | 0.93 [0.90, 0.96] / 0.89 [0.84, 0.92] | 0.92 [0.88, 0.95] / 0.90 [0.86, 0.94] |
| [0.05, 0.15) | 112 (47) | 0.69 [0.58, 0.78] / 0.71 [0.61, 0.80] | 0.66 [0.56, 0.76] / 0.66 [0.56, 0.75] | 0.69 [0.59, 0.79] / 0.68 [0.58, 0.78] |
| [0.15, 0.30) | 126 (63) | 0.49 [0.39, 0.59] / 0.55 [0.44, 0.65] | 0.48 [0.39, 0.59] / 0.58 [0.48, 0.69] | 0.50 [0.40, 0.60] / 0.57 [0.47, 0.67] |
| [0.30, inf) | 58 (39) | 0.34 [0.20, 0.48] / 0.55 [0.37, 0.73] | 0.33 [0.18, 0.49] / 0.58 [0.42, 0.74] | 0.33 [0.19, 0.49] / 0.48 [0.30, 0.66] |

## B. Perturbed registration on the five pre-registered categories (249 items): 5-category mean object AUROC max / covered, point AUROC, hit rate

| perturbation | dentnet_nom_rec3 | dentnet_nom_rec3_norm | pnpp_nom_rec3_norm |
|---|---|---|---|
| none (baseline) | 0.803 / 0.953, pt 0.952, hit 0.81 | 0.990 / 0.957, pt 0.963, hit 0.97 | 0.964 / 0.945, pt 0.955, hit 0.94 |
| rot 0.25 deg | 0.805 / 0.948, pt 0.951, hit 0.85 | 0.985 / 0.958, pt 0.963, hit 0.97 | 0.960 / 0.954, pt 0.959, hit 0.95 |
| rot 0.50 deg | 0.804 / 0.945, pt 0.948, hit 0.77 | 0.987 / 0.953, pt 0.961, hit 0.97 | 0.950 / 0.960, pt 0.959, hit 0.94 |
| rot 1.00 deg | 0.771 / 0.929, pt 0.938, hit 0.69 | 0.983 / 0.946, pt 0.956, hit 0.97 | 0.946 / 0.932, pt 0.951, hit 0.93 |
| rot 2.00 deg | 0.681 / 0.814, pt 0.901, hit 0.56 | 0.909 / 0.885, pt 0.933, hit 0.90 | 0.868 / 0.860, pt 0.925, hit 0.82 |
| trans 0.10 mm | 0.801 / 0.944, pt 0.949, hit 0.80 | 0.991 / 0.952, pt 0.962, hit 0.98 | 0.957 / 0.954, pt 0.961, hit 0.96 |
| trans 0.25 mm | 0.758 / 0.940, pt 0.934, hit 0.71 | 0.982 / 0.948, pt 0.954, hit 0.98 | 0.934 / 0.931, pt 0.949, hit 0.95 |
| trans 0.50 mm | 0.771 / 0.878, pt 0.860, hit 0.60 | 0.878 / 0.884, pt 0.902, hit 0.90 | 0.849 / 0.821, pt 0.892, hit 0.87 |
| trans 1.00 mm | 0.638 / 0.747, pt 0.773, hit 0.35 | 0.651 / 0.701, pt 0.812, hit 0.51 | 0.643 / 0.680, pt 0.807, hit 0.54 |
