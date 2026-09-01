# Evaluation — face

Winner: logreg / dino+fft / all_views (val AUC 0.8320, threshold 0.647)

_Negative (real) reference for AUC: eval_face_cdf_

_eval_face_cdf is reals-only: FPR under domain shift, no AUC._

## Per-slice metrics (at fitted threshold)

| slice | view | n | AUC | bal_acc | FPR | FNR |
|---|---|---|---|---|---|---|
| eval_face_cdf | blur0.5 | 2664 | nan | nan | 0.2541 | nan |
| eval_face_cdf | blur1.0 | 2664 | nan | nan | 0.3108 | nan |
| eval_face_cdf | blur2.0 | 2664 | nan | nan | 0.3863 | nan |
| eval_face_cdf | bright+20 | 2664 | nan | nan | 0.3078 | nan |
| eval_face_cdf | bright-20 | 2664 | nan | nan | 0.1787 | nan |
| eval_face_cdf | canon_jpeg95 | 2664 | nan | nan | 0.2770 | nan |
| eval_face_cdf | color+20 | 2664 | nan | nan | 0.2770 | nan |
| eval_face_cdf | color-20 | 2664 | nan | nan | 0.2271 | nan |
| eval_face_cdf | contrast+20 | 2664 | nan | nan | 0.3619 | nan |
| eval_face_cdf | contrast-20 | 2664 | nan | nan | 0.1509 | nan |
| eval_face_cdf | crop80 | 2664 | nan | nan | 0.3232 | nan |
| eval_face_cdf | jpeg30 | 2664 | nan | nan | 0.2673 | nan |
| eval_face_cdf | jpeg50 | 2664 | nan | nan | 0.2917 | nan |
| eval_face_cdf | jpeg70 | 2664 | nan | nan | 0.3108 | nan |
| eval_face_cdf | jpeg90 | 2664 | nan | nan | 0.3022 | nan |
| eval_face_cdf | noise0.02 | 2664 | nan | nan | 0.2774 | nan |
| eval_face_cdf | noise0.05 | 2664 | nan | nan | 0.3758 | nan |
| eval_face_cdf | noise0.10 | 2664 | nan | nan | 0.4155 | nan |
| eval_face_cdf | orig | 2664 | nan | nan | 0.2489 | nan |
| eval_face_cdf | resize0.25 | 2664 | nan | nan | 0.2864 | nan |
| eval_face_cdf | resize0.5 | 2664 | nan | nan | 0.2695 | nan |
| eval_face_unseen | blur0.5 | 3006 | 0.7346 | 0.6652 | 0.2541 | 0.4155 |
| eval_face_unseen | blur1.0 | 3006 | 0.7217 | 0.6573 | 0.3108 | 0.3746 |
| eval_face_unseen | blur2.0 | 3006 | 0.6861 | 0.6294 | 0.3863 | 0.3550 |
| eval_face_unseen | bright+20 | 3006 | 0.7197 | 0.6626 | 0.3078 | 0.3669 |
| eval_face_unseen | bright-20 | 3006 | 0.7287 | 0.6342 | 0.1787 | 0.5529 |
| eval_face_unseen | canon_jpeg95 | 3006 | 0.7263 | 0.6614 | 0.2770 | 0.4002 |
| eval_face_unseen | color+20 | 3006 | 0.7194 | 0.6539 | 0.2770 | 0.4152 |
| eval_face_unseen | color-20 | 3006 | 0.7290 | 0.6513 | 0.2271 | 0.4704 |
| eval_face_unseen | contrast+20 | 3006 | 0.7186 | 0.6639 | 0.3619 | 0.3104 |
| eval_face_unseen | contrast-20 | 3006 | 0.7272 | 0.6263 | 0.1509 | 0.5965 |
| eval_face_unseen | crop80 | 3006 | 0.6852 | 0.6263 | 0.3232 | 0.4242 |
| eval_face_unseen | jpeg30 | 3006 | 0.7033 | 0.6338 | 0.2673 | 0.4651 |
| eval_face_unseen | jpeg50 | 3006 | 0.7120 | 0.6462 | 0.2917 | 0.4158 |
| eval_face_unseen | jpeg70 | 3006 | 0.7123 | 0.6493 | 0.3108 | 0.3906 |
| eval_face_unseen | jpeg90 | 3006 | 0.7239 | 0.6588 | 0.3022 | 0.3802 |
| eval_face_unseen | noise0.02 | 3006 | 0.7246 | 0.6550 | 0.2774 | 0.4125 |
| eval_face_unseen | noise0.05 | 3006 | 0.6847 | 0.6283 | 0.3758 | 0.3676 |
| eval_face_unseen | noise0.10 | 3006 | 0.6410 | 0.5943 | 0.4155 | 0.3959 |
| eval_face_unseen | orig | 3006 | 0.7258 | 0.6563 | 0.2489 | 0.4385 |
| eval_face_unseen | resize0.25 | 3006 | 0.6962 | 0.6294 | 0.2864 | 0.4548 |
| eval_face_unseen | resize0.5 | 3006 | 0.7185 | 0.6490 | 0.2695 | 0.4325 |

## Robustness by view (eval fakes vs eval_face_cdf reals)

val (canonical) reference AUC: **0.8320**

| view | n fakes | AUC | TPR@thr |
|---|---|---|---|
| orig | 3006 | 0.7258 | 0.5615 |
| canon_jpeg95 | 3006 | 0.7263 | 0.5998 |
| jpeg90 | 3006 | 0.7239 | 0.6198 |
| jpeg70 | 3006 | 0.7123 | 0.6094 |
| jpeg50 | 3006 | 0.7120 | 0.5842 |
| jpeg30 | 3006 | 0.7033 | 0.5349 |
| blur0.5 | 3006 | 0.7346 | 0.5845 |
| blur1.0 | 3006 | 0.7217 | 0.6254 |
| blur2.0 | 3006 | 0.6861 | 0.6450 |
| resize0.5 | 3006 | 0.7185 | 0.5675 |
| resize0.25 | 3006 | 0.6962 | 0.5452 |
| noise0.02 | 3006 | 0.7246 | 0.5875 |
| noise0.05 | 3006 | 0.6847 | 0.6324 |
| noise0.10 | 3006 | 0.6410 | 0.6041 |
| color+20 | 3006 | 0.7194 | 0.5848 |
| color-20 | 3006 | 0.7290 | 0.5296 |
| bright+20 | 3006 | 0.7197 | 0.6331 |
| bright-20 | 3006 | 0.7287 | 0.4471 |
| contrast+20 | 3006 | 0.7186 | 0.6896 |
| contrast-20 | 3006 | 0.7272 | 0.4035 |
| crop80 | 3006 | 0.6852 | 0.5758 |

## Error gallery

error_gallery/face/fp (reals from eval_face_cdf scored fake) and /fn (eval fakes scored real), top-15 by confidence, canonical view; CSVs alongside.
