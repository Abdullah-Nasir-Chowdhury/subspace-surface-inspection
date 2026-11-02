# Real3D-AD object-level AUROC (%): published footing (every item, max over windows) beside the round-3 covered-window protocol

Published rows: PO3AD (CVPR 2025) Table 3. Ours: item-clustered, category mean with a 95 % within-category item bootstrap interval.

## Table A: every item (597), aggregation = max over all valid windows (the published convention)

| method | airplane | car | candybar | chicken | diamond | duck | fish | gemstone | seahorse | shell | starfish | toffees | mean 12 [CI] | mean 5 reg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTF (Raw) | 73.0 | 64.7 | 53.9 | 78.9 | 70.7 | 69.1 | 60.2 | 68.6 | 59.6 | 39.6 | 53.0 | 70.3 | 63.5 | 61.7 |
| PatchCore (FPFH) | 88.2 | 59.0 | 54.1 | 83.7 | 57.4 | 54.6 | 67.5 | 37.0 | 50.5 | 58.9 | 44.1 | 56.5 | 59.3 | 57.5 |
| CPMF | 70.1 | 55.1 | 55.2 | 50.4 | 52.3 | 58.2 | 55.8 | 58.9 | 72.9 | 65.3 | 70.0 | 39.0 | 58.6 | 55.6 |
| Reg3D-AD | 71.6 | 69.7 | 68.5 | 85.2 | 90.0 | 58.4 | 91.5 | 41.7 | 76.2 | 58.3 | 50.6 | 82.7 | 70.4 | 77.7 |
| IMRNet | 76.2 | 71.1 | 75.5 | 78.0 | 90.5 | 51.7 | 88.0 | 67.4 | 60.4 | 66.5 | 67.4 | 77.4 | 72.5 | 74.5 |
| R3D-AD | 77.2 | 69.3 | 71.3 | 71.4 | 68.5 | 90.9 | 69.2 | 66.5 | 72.0 | 84.0 | 70.1 | 70.3 | 73.4 | 70.4 |
| Group3AD | 74.4 | 72.8 | 84.7 | 78.6 | 93.2 | 67.9 | 97.6 | 53.9 | 84.1 | 58.5 | 56.2 | 79.6 | 75.1 | 83.8 |
| PO3AD | 80.4 | 65.4 | 78.5 | 68.6 | 80.1 | 82.0 | 85.9 | 69.3 | 75.6 | 80.0 | 75.8 | 77.1 | 76.6 | 76.5 |
| point-to-template distance, 2.5 mm | 59.7 | 85.4 | 92.2 | 61.2 | 66.8 | 69.4 | 98.6 | 63.8 | 97.9 | 43.5 | 83.7 | 94.2 | 76.4 [72.8, 80.1] | 93.6 |
| Mag(D) vs nearest nominal patch, 2.5 mm | 56.3 | 66.7 | 42.1 | 64.4 | 64.2 | 47.2 | 82.7 | 56.0 | 67.2 | 61.1 | 48.6 | 83.8 | 61.7 [56.8, 66.3] | 68.5 |
| nominal subspace, height field, normalized, 2.5 mm | 55.7 | 90.6 | 60.6 | 70.2 | 88.2 | 55.0 | 98.7 | 54.2 | 96.5 | 53.8 | 76.0 | 93.0 | 74.4 [70.5, 78.2] | 87.9 |
| nearest nominal feature, DentNet | 52.2 | 79.5 | 81.6 | 66.8 | 74.4 | 85.0 | 88.6 | 54.9 | 73.3 | 54.9 | 75.7 | 85.0 | 72.7 [68.5, 76.6] | 81.6 |
| local nominal subspace, DentNet features | 48.6 | 78.1 | 81.3 | 72.3 | 73.2 | 84.5 | 86.2 | 58.7 | 65.6 | 56.0 | 70.4 | 90.2 | 72.1 [67.9, 76.2] | 80.3 |
| local nominal subspace, DentNet features, normalized | 45.1 | 95.2 | 100.0 | 67.5 | 86.3 | 55.2 | 100.0 | 63.0 | 100.0 | 45.0 | 81.4 | 99.7 | 78.2 [74.8, 81.5] | 99.0 |
| local nominal subspace, PointNet++ features | 48.2 | 82.6 | 97.8 | 65.2 | 73.6 | 79.4 | 96.0 | 57.0 | 85.9 | 47.6 | 79.4 | 84.3 | 74.7 [71.0, 78.5] | 89.3 |
| local nominal subspace, PointNet++ features, normalized | 51.5 | 89.1 | 99.5 | 65.1 | 86.6 | 69.4 | 100.0 | 65.0 | 97.3 | 43.3 | 81.9 | 96.0 | 78.7 [75.2, 82.1] | 96.4 |

## Table B: our statistics under every item set and aggregation (12-category mean [CI]; mean over 5 pre-registered; pooled item AUROC)

| statistic | set | n | max | top5 | covered | top5cov |
|---|---|---|---|---|---|---|
| point-to-template distance, 2.5 mm | all597 | 597 | 76.4 [72.8, 80.1]; 5-reg 93.6; pooled 0.660 | 76.3 [72.7, 79.9]; 5-reg 93.4; pooled 0.662 | 72.6 [68.7, 76.3]; 5-reg 94.0; pooled 0.654 | 73.4 [69.6, 77.1]; 5-reg 93.4; pooled 0.655 |
| point-to-template distance, 2.5 mm | strict | 591 | 76.4 [72.6, 80.0]; 5-reg 93.6; pooled 0.661 | 76.3 [72.6, 79.8]; 5-reg 93.4; pooled 0.662 | 72.5 [68.5, 76.1]; 5-reg 94.0; pooled 0.655 | 73.3 [69.3, 76.9]; 5-reg 93.4; pooled 0.656 |
| Mag(D) vs nearest nominal patch, 2.5 mm | all597 | 597 | 61.7 [56.8, 66.3]; 5-reg 68.5; pooled 0.567 | 64.7 [60.1, 69.4]; 5-reg 73.5; pooled 0.566 | 75.7 [71.9, 79.6]; 5-reg 89.3; pooled 0.672 | 75.8 [71.9, 79.7]; 5-reg 89.7; pooled 0.669 |
| Mag(D) vs nearest nominal patch, 2.5 mm | strict | 591 | 61.3 [56.7, 65.8]; 5-reg 68.5; pooled 0.564 | 64.3 [59.8, 68.9]; 5-reg 73.5; pooled 0.564 | 75.7 [71.8, 79.6]; 5-reg 89.3; pooled 0.672 | 75.7 [71.8, 79.6]; 5-reg 89.7; pooled 0.669 |
| nominal subspace, height field, normalized, 2.5 mm | all597 | 597 | 74.4 [70.5, 78.2]; 5-reg 87.9; pooled 0.679 | 75.3 [71.4, 79.1]; 5-reg 88.5; pooled 0.680 | 79.1 [75.7, 82.6]; 5-reg 96.9; pooled 0.747 | 79.1 [75.6, 82.5]; 5-reg 96.7; pooled 0.745 |
| nominal subspace, height field, normalized, 2.5 mm | strict | 591 | 74.4 [70.6, 78.2]; 5-reg 87.9; pooled 0.680 | 75.3 [71.8, 79.1]; 5-reg 88.5; pooled 0.681 | 78.8 [75.3, 82.5]; 5-reg 96.9; pooled 0.747 | 78.9 [75.4, 82.3]; 5-reg 96.7; pooled 0.745 |
| nearest nominal feature, DentNet | all597 | 597 | 72.7 [68.5, 76.6]; 5-reg 81.6; pooled 0.649 | 73.5 [69.5, 77.4]; 5-reg 83.3; pooled 0.635 | 81.5 [78.1, 84.8]; 5-reg 94.4; pooled 0.738 | 80.6 [77.2, 83.9]; 5-reg 94.1; pooled 0.698 |
| nearest nominal feature, DentNet | strict | 591 | 72.7 [68.6, 76.7]; 5-reg 81.6; pooled 0.649 | 73.3 [69.2, 77.1]; 5-reg 83.3; pooled 0.635 | 81.4 [78.0, 84.8]; 5-reg 94.4; pooled 0.739 | 80.4 [77.0, 83.8]; 5-reg 94.1; pooled 0.699 |
| local nominal subspace, DentNet features | all597 | 597 | 72.1 [67.9, 76.2]; 5-reg 80.3; pooled 0.643 | 74.5 [70.8, 78.2]; 5-reg 84.3; pooled 0.635 | 82.1 [78.7, 85.2]; 5-reg 95.3; pooled 0.741 | 81.1 [77.5, 84.2]; 5-reg 95.2; pooled 0.706 |
| local nominal subspace, DentNet features | strict | 591 | 71.8 [67.8, 75.7]; 5-reg 80.3; pooled 0.643 | 74.2 [70.4, 77.8]; 5-reg 84.3; pooled 0.635 | 81.9 [78.5, 85.3]; 5-reg 95.3; pooled 0.741 | 80.8 [77.4, 84.2]; 5-reg 95.2; pooled 0.707 |
| local nominal subspace, DentNet features, normalized | all597 | 597 | 78.2 [74.8, 81.5]; 5-reg 99.0; pooled 0.710 | 78.6 [75.3, 81.9]; 5-reg 98.9; pooled 0.702 | 80.7 [77.1, 84.1]; 5-reg 95.7; pooled 0.734 | 80.5 [76.8, 84.0]; 5-reg 95.2; pooled 0.708 |
| local nominal subspace, DentNet features, normalized | strict | 591 | 78.1 [74.6, 81.6]; 5-reg 99.0; pooled 0.711 | 78.5 [75.1, 81.9]; 5-reg 98.9; pooled 0.702 | 80.5 [77.0, 83.9]; 5-reg 95.7; pooled 0.734 | 80.3 [76.8, 83.8]; 5-reg 95.2; pooled 0.708 |
| local nominal subspace, PointNet++ features | all597 | 597 | 74.7 [71.0, 78.5]; 5-reg 89.3; pooled 0.667 | 76.9 [73.3, 80.5]; 5-reg 90.2; pooled 0.649 | 80.9 [77.5, 84.2]; 5-reg 94.8; pooled 0.723 | 80.6 [77.0, 83.9]; 5-reg 94.8; pooled 0.697 |
| local nominal subspace, PointNet++ features | strict | 591 | 74.5 [70.7, 78.3]; 5-reg 89.3; pooled 0.667 | 76.8 [73.1, 80.3]; 5-reg 90.2; pooled 0.649 | 80.8 [77.4, 84.0]; 5-reg 94.8; pooled 0.724 | 80.4 [76.9, 83.8]; 5-reg 94.8; pooled 0.698 |
| local nominal subspace, PointNet++ features, normalized | all597 | 597 | 78.7 [75.2, 82.1]; 5-reg 96.4; pooled 0.733 | 79.5 [76.2, 82.8]; 5-reg 97.1; pooled 0.713 | 79.5 [75.6, 82.9]; 5-reg 94.5; pooled 0.744 | 79.6 [75.8, 83.0]; 5-reg 94.4; pooled 0.712 |
| local nominal subspace, PointNet++ features, normalized | strict | 591 | 78.5 [75.0, 81.9]; 5-reg 96.4; pooled 0.733 | 79.3 [76.0, 82.6]; 5-reg 97.1; pooled 0.714 | 79.6 [76.2, 83.2]; 5-reg 94.5; pooled 0.746 | 79.6 [76.1, 83.2]; 5-reg 94.4; pooled 0.714 |

## Table C: per-category, every item, covered-window aggregation (the round-3 protocol, now on all items)

| method | airplane | car | candybar | chicken | diamond | duck | fish | gemstone | seahorse | shell | starfish | toffees | mean 12 [CI] | mean 5 reg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| point-to-template distance, 2.5 mm | 51.4 | 85.8 | 94.7 | 48.7 | 73.2 | 48.8 | 99.5 | 59.0 | 95.5 | 43.0 | 76.5 | 94.7 | 72.6 [68.7, 76.3] | 94.0 |
| Mag(D) vs nearest nominal patch, 2.5 mm | 63.7 | 75.5 | 99.7 | 64.9 | 75.0 | 75.2 | 98.7 | 55.7 | 92.0 | 59.2 | 68.8 | 80.5 | 75.7 [71.9, 79.6] | 89.3 |
| nominal subspace, height field, normalized, 2.5 mm | 68.2 | 93.8 | 100.0 | 76.1 | 79.3 | 45.1 | 100.0 | 55.8 | 93.1 | 68.2 | 72.5 | 97.7 | 79.1 [75.7, 82.6] | 96.9 |
| nearest nominal feature, DentNet | 62.7 | 78.9 | 99.5 | 51.8 | 91.3 | 89.9 | 97.8 | 64.8 | 97.6 | 64.0 | 82.1 | 98.2 | 81.5 [78.1, 84.8] | 94.4 |
| local nominal subspace, DentNet features | 59.5 | 81.0 | 99.4 | 57.4 | 90.8 | 89.9 | 99.5 | 69.3 | 96.8 | 61.0 | 80.2 | 100.0 | 82.1 [78.7, 85.2] | 95.3 |
| local nominal subspace, DentNet features, normalized | 59.7 | 85.0 | 99.4 | 72.6 | 82.3 | 57.1 | 97.8 | 69.4 | 97.1 | 65.9 | 83.5 | 99.2 | 80.7 [77.1, 84.1] | 95.7 |
| local nominal subspace, PointNet++ features | 64.8 | 79.0 | 99.7 | 61.2 | 85.7 | 85.1 | 99.2 | 64.0 | 97.4 | 52.9 | 83.2 | 98.8 | 80.9 [77.5, 84.2] | 94.8 |
| local nominal subspace, PointNet++ features, normalized | 61.8 | 77.8 | 99.8 | 64.1 | 81.6 | 68.6 | 97.9 | 68.8 | 97.3 | 54.5 | 81.8 | 99.8 | 79.5 [75.6, 82.9] | 94.5 |
