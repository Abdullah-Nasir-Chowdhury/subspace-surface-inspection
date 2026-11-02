# Real3D-AD, training-free statistics, defect versus good

### radius 2.5 mm, 249 items (124 defective), 673 seeds per item (median)
| statistic | object AUROC [95% CI] | object AUROC, template-z | object AUROC, edge-filtered | point AUROC [95% CI] | loc. hit rate [95% CI] | median dist mm | median rank frac |
|---|---|---|---|---|---|---|---|
| magd_tmpl | 0.597 [0.526, 0.667] | 0.597 | 0.708 | 0.833 [0.810, 0.854] | 0.22 [0.15, 0.29] | 9.77 | 0.029 |
| diff_rms | 0.520 [0.451, 0.594] | 0.520 | 0.709 | 0.779 [0.751, 0.803] | 0.08 [0.03, 0.13] | 11.30 | 0.136 |
| diff_absmax | 0.506 [0.433, 0.579] | 0.506 | 0.643 | 0.766 [0.741, 0.789] | 0.09 [0.04, 0.14] | 9.78 | 0.120 |
| nn_p95 | 0.916 [0.879, 0.949] | 0.916 | 0.909 | 0.975 [0.966, 0.982] | 0.91 [0.86, 0.96] | 0.62 | 0.002 |

### radius 3.6 mm, 249 items (124 defective), 336 seeds per item (median)
| statistic | object AUROC [95% CI] | object AUROC, template-z | object AUROC, edge-filtered | point AUROC [95% CI] | loc. hit rate [95% CI] | median dist mm | median rank frac |
|---|---|---|---|---|---|---|---|
| magd_tmpl | 0.528 [0.453, 0.599] | 0.528 | 0.664 | 0.648 [0.616, 0.674] | 0.13 [0.07, 0.19] | 10.01 | 0.070 |
| diff_rms | 0.511 [0.438, 0.588] | 0.511 | 0.618 | 0.589 [0.554, 0.619] | 0.04 [0.01, 0.08] | 11.03 | 0.115 |
| diff_absmax | 0.548 [0.475, 0.618] | 0.548 | 0.592 | 0.571 [0.544, 0.605] | 0.07 [0.03, 0.12] | 11.35 | 0.072 |
| nn_p95 | 0.896 [0.855, 0.932] | 0.896 | 0.811 | 0.954 [0.936, 0.963] | 0.88 [0.82, 0.94] | 1.56 | 0.004 |

### radius 5.0 mm, 249 items (124 defective), 179 seeds per item (median)
| statistic | object AUROC [95% CI] | object AUROC, template-z | object AUROC, edge-filtered | point AUROC [95% CI] | loc. hit rate [95% CI] | median dist mm | median rank frac |
|---|---|---|---|---|---|---|---|
| magd_tmpl | 0.501 [0.428, 0.574] | 0.501 | 0.556 | 0.497 [0.453, 0.523] | 0.15 [0.09, 0.21] | 9.52 | 0.077 |
| diff_rms | 0.516 [0.442, 0.590] | 0.516 | 0.559 | 0.460 [0.421, 0.495] | 0.15 [0.10, 0.22] | 9.91 | 0.078 |
| diff_absmax | 0.515 [0.447, 0.585] | 0.515 | 0.480 | 0.470 [0.432, 0.510] | 0.16 [0.10, 0.23] | 9.76 | 0.055 |
| nn_p95 | 0.860 [0.812, 0.903] | 0.860 | 0.750 | 0.910 [0.891, 0.930] | 0.85 [0.79, 0.91] | 2.79 | 0.007 |
