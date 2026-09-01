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
