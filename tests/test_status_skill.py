import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / ".claude"
    / "skills"
    / "traceguard-status"
    / "scripts"
    / "inspect_checkpoint.py"
)


def load_status_module():
    spec = importlib.util.spec_from_file_location("inspect_checkpoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_overfitting_diagnosis_distinguishes_loss_minimum_from_best_auc():
    status = load_status_module()
    history = [
        {"epoch": 1, "train_loss": 0.5, "validation_loss": 0.8, "roc_auc": 0.65},
        {"epoch": 2, "train_loss": 0.3, "validation_loss": 0.6, "roc_auc": 0.67},
        {"epoch": 3, "train_loss": 0.2, "validation_loss": 0.9, "roc_auc": 0.72},
        {"epoch": 4, "train_loss": 0.1, "validation_loss": 1.1, "roc_auc": 0.60},
    ]

    diagnosis = status.diagnose(history)

    assert "validation loss bottomed out at epoch 2" in diagnosis
    assert "Best ROC-AUC occurred at epoch 3" in diagnosis
    assert "which best.pt preserves" in diagnosis
