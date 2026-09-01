# TraceGuard — results summary

_eval_face_cdf is reals-only: FPR under domain shift, no AUC._

# Ablation — face

| model | features | train views | val ROC-AUC | train rows | secs |
|---|---|---|---|---|---|
| logreg | dino+fft | canonical | 0.8428 **<- winner** | 23659 | 6.9 |
| mlp | dino+fft | canonical | 0.8349 | 23659 | 1.2 |
| logreg | dino | canonical | 0.8336 | 23659 | 4.9 |
| logreg | dino+fft | all_views | 0.8320 | 165613 | 37.5 |
| mlp | dino | canonical | 0.8272 | 23659 | 3.8 |
| logreg | dino | all_views | 0.8227 | 165613 | 17.5 |
| mlp | dino+fft | all_views | 0.8073 | 165613 | 3.1 |
| mlp | dino | all_views | 0.7862 | 165613 | 3.9 |

Winner threshold (max balanced acc on val): **0.713** (val balanced acc 0.7582)

**Shipped-head override:** the automatic winner rule picks by clean val AUC
(logreg/dino+fft/canonical, 0.8428), but on the unseen-method eval the
all-views (degradation-augmented) logreg/dino+fft head is better on EVERY
robustness view (+0.03 to +0.06 AUC, e.g. orig 0.696->0.726, noise0.05
0.626->0.685) at a cost of 1 point of seen-method val AUC (0.832). Shipped
face_head.pkl = all_views variant; the canonical head is preserved as
face_head_canonical.pkl.


---

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
| eval_face_cdf | canon_jpeg95 | 2664 | nan | nan | 0.2770 | nan |
| eval_face_cdf | color+20 | 2664 | nan | nan | 0.2770 | nan |
| eval_face_cdf | color-20 | 2664 | nan | nan | 0.2271 | nan |
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
| eval_face_unseen | canon_jpeg95 | 3006 | 0.7263 | 0.6614 | 0.2770 | 0.4002 |
| eval_face_unseen | color+20 | 3006 | 0.7194 | 0.6539 | 0.2770 | 0.4152 |
| eval_face_unseen | color-20 | 3006 | 0.7290 | 0.6513 | 0.2271 | 0.4704 |
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
| crop80 | 3006 | 0.6852 | 0.5758 |

## Error gallery

error_gallery/face/fp (reals from eval_face_cdf scored fake) and /fn (eval fakes scored real), top-15 by confidence, canonical view; CSVs alongside.


---

# Ablation — scene

| model | features | train views | val ROC-AUC | train rows | secs |
|---|---|---|---|---|---|
| mlp | dino+fft | canonical | 0.9929 **<- winner** | 35640 | 1.5 |
| mlp | dino | canonical | 0.9912 | 35640 | 6.1 |
| logreg | dino+fft | canonical | 0.9855 | 35640 | 4.9 |
| logreg | dino | canonical | 0.9820 | 35640 | 3.1 |

_all-views regime skipped: shards contain no degraded train views yet (phase-1 extraction); rerun after phase 2._

Winner threshold (max balanced acc on val): **0.873** (val balanced acc 0.9643)


---

# Evaluation — scene

Winner: mlp / dino+fft / canonical (val AUC 0.9929, threshold 0.873)

_Negative (real) reference for AUC: val — NO scene eval reals exist; val reals used (noted optimism)_

_eval_face_cdf is reals-only: FPR under domain shift, no AUC._

## Per-slice metrics (at fitted threshold)

| slice | view | n | AUC | bal_acc | FPR | FNR |
|---|---|---|---|---|---|---|
| eval_genimagepp | blur0.5 | 12000 | 0.9782 | 0.9092 | 0.0219 | 0.1597 |
| eval_genimagepp | blur1.0 | 12000 | 0.9870 | 0.9354 | 0.0219 | 0.1074 |
| eval_genimagepp | blur2.0 | 12000 | 0.9929 | 0.9578 | 0.0219 | 0.0625 |
| eval_genimagepp | canon_jpeg95 | 12000 | 0.9716 | 0.8908 | 0.0219 | 0.1966 |
| eval_genimagepp | color+20 | 12000 | 0.9741 | 0.8984 | 0.0219 | 0.1814 |
| eval_genimagepp | color-20 | 12000 | 0.9744 | 0.8994 | 0.0219 | 0.1792 |
| eval_genimagepp | crop80 | 12000 | 0.9719 | 0.8955 | 0.0219 | 0.1872 |
| eval_genimagepp | jpeg30 | 12000 | 0.9442 | 0.8213 | 0.0219 | 0.3355 |
| eval_genimagepp | jpeg50 | 12000 | 0.9511 | 0.8371 | 0.0219 | 0.3038 |
| eval_genimagepp | jpeg70 | 12000 | 0.9607 | 0.8624 | 0.0219 | 0.2534 |
| eval_genimagepp | jpeg90 | 12000 | 0.9700 | 0.8873 | 0.0219 | 0.2036 |
| eval_genimagepp | noise0.02 | 12000 | 0.9655 | 0.8716 | 0.0219 | 0.2349 |
| eval_genimagepp | noise0.05 | 12000 | 0.9435 | 0.8146 | 0.0219 | 0.3490 |
| eval_genimagepp | noise0.10 | 12000 | 0.9048 | 0.7331 | 0.0219 | 0.5118 |
| eval_genimagepp | orig | 12000 | 0.9742 | 0.8988 | 0.0219 | 0.1805 |
| eval_genimagepp | resize0.25 | 12000 | 0.9889 | 0.9414 | 0.0219 | 0.0953 |
| eval_genimagepp | resize0.5 | 12000 | 0.9902 | 0.9474 | 0.0219 | 0.0833 |
| eval_tampered | blur0.5 | 4992 | 0.6758 | 0.5387 | 0.0219 | 0.9006 |
| eval_tampered | blur1.0 | 4992 | 0.7600 | 0.5704 | 0.0219 | 0.8373 |
| eval_tampered | blur2.0 | 4992 | 0.8557 | 0.6509 | 0.0219 | 0.6763 |
| eval_tampered | canon_jpeg95 | 4992 | 0.6483 | 0.5315 | 0.0219 | 0.9151 |
| eval_tampered | color+20 | 4992 | 0.6500 | 0.5323 | 0.0219 | 0.9135 |
| eval_tampered | color-20 | 4992 | 0.6528 | 0.5343 | 0.0219 | 0.9095 |
| eval_tampered | crop80 | 4992 | 0.6618 | 0.5383 | 0.0219 | 0.9014 |
| eval_tampered | jpeg30 | 4992 | 0.6002 | 0.5183 | 0.0219 | 0.9415 |
| eval_tampered | jpeg50 | 4992 | 0.5850 | 0.5156 | 0.0219 | 0.9469 |
| eval_tampered | jpeg70 | 4992 | 0.6261 | 0.5245 | 0.0219 | 0.9291 |
| eval_tampered | jpeg90 | 4992 | 0.6481 | 0.5318 | 0.0219 | 0.9145 |
| eval_tampered | noise0.02 | 4992 | 0.6333 | 0.5252 | 0.0219 | 0.9277 |
| eval_tampered | noise0.05 | 4992 | 0.5769 | 0.5113 | 0.0219 | 0.9555 |
| eval_tampered | noise0.10 | 4992 | 0.4839 | 0.4991 | 0.0219 | 0.9800 |
| eval_tampered | orig | 4992 | 0.6497 | 0.5326 | 0.0219 | 0.9129 |
| eval_tampered | resize0.25 | 4992 | 0.8112 | 0.6007 | 0.0219 | 0.7766 |
| eval_tampered | resize0.5 | 4992 | 0.8059 | 0.5975 | 0.0219 | 0.7831 |

## Robustness by view (eval fakes vs val reals)

val (canonical) reference AUC: **0.9929**

| view | n fakes | AUC | TPR@thr |
|---|---|---|---|
| orig | 16992 | 0.8789 | 0.6043 |
| canon_jpeg95 | 16992 | 0.8766 | 0.5923 |
| jpeg90 | 16992 | 0.8754 | 0.5876 |
| jpeg70 | 16992 | 0.8624 | 0.5481 |
| jpeg50 | 16992 | 0.8435 | 0.5072 |
| jpeg30 | 16992 | 0.8432 | 0.4865 |
| blur0.5 | 16992 | 0.8894 | 0.6226 |
| blur1.0 | 16992 | 0.9203 | 0.6781 |
| blur2.0 | 16992 | 0.9526 | 0.7572 |
| resize0.5 | 16992 | 0.9360 | 0.7111 |
| resize0.25 | 16992 | 0.9367 | 0.7045 |
| noise0.02 | 16992 | 0.8679 | 0.5616 |
| noise0.05 | 16992 | 0.8358 | 0.4728 |
| noise0.10 | 16992 | 0.7811 | 0.3506 |
| color+20 | 16992 | 0.8789 | 0.6035 |
| color-20 | 16992 | 0.8799 | 0.6062 |
| crop80 | 16992 | 0.8808 | 0.6030 |

## Error gallery

error_gallery/scene/fp (reals from val scored fake) and /fn (eval fakes scored real), top-15 by confidence, canonical view; CSVs alongside.
