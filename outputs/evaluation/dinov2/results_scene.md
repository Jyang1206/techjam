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
