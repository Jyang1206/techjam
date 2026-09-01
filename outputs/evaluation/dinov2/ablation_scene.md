# Ablation — scene

| model | features | train views | val ROC-AUC | train rows | secs |
|---|---|---|---|---|---|
| mlp | dino+fft | canonical | 0.9929 **<- winner** | 35640 | 1.5 |
| mlp | dino | canonical | 0.9912 | 35640 | 6.1 |
| logreg | dino+fft | canonical | 0.9855 | 35640 | 4.9 |
| logreg | dino | canonical | 0.9820 | 35640 | 3.1 |

_all-views regime skipped: shards contain no degraded train views yet (phase-1 extraction); rerun after phase 2._

Winner threshold (max balanced acc on val): **0.873** (val balanced acc 0.9643)
