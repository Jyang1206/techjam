# Robustness table (AUC per view)

_eval_face_cdf is reals-only: FPR under domain shift, no AUC._

## face

| view | AUC |
|---|---|
| orig | 0.7258 |
| canon_jpeg95 | 0.7263 |
| jpeg90 | 0.7239 |
| jpeg70 | 0.7123 |
| jpeg50 | 0.7120 |
| jpeg30 | 0.7033 |
| blur0.5 | 0.7346 |
| blur1.0 | 0.7217 |
| blur2.0 | 0.6861 |
| resize0.5 | 0.7185 |
| resize0.25 | 0.6962 |
| noise0.02 | 0.7246 |
| noise0.05 | 0.6847 |
| noise0.10 | 0.6410 |
| color+20 | 0.7194 |
| color-20 | 0.7290 |
| crop80 | 0.6852 |

## scene

| view | AUC |
|---|---|
| orig | 0.8789 |
| canon_jpeg95 | 0.8766 |
| jpeg90 | 0.8754 |
| jpeg70 | 0.8624 |
| jpeg50 | 0.8435 |
| jpeg30 | 0.8432 |
| blur0.5 | 0.8894 |
| blur1.0 | 0.9203 |
| blur2.0 | 0.9526 |
| resize0.5 | 0.9360 |
| resize0.25 | 0.9367 |
| noise0.02 | 0.8679 |
| noise0.05 | 0.8358 |
| noise0.10 | 0.7811 |
| color+20 | 0.8789 |
| color-20 | 0.8799 |
| crop80 | 0.8808 |
