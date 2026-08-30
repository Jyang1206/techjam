from pathlib import Path

import torch

from traceguard.train import pick_resume_checkpoint, save_training_state


def test_pick_resume_checkpoint_prefers_latest(tmp_path: Path):
    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"latest")
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")

    assert pick_resume_checkpoint(tmp_path, None) == latest
    assert pick_resume_checkpoint(tmp_path, str(best)) == best


def test_pick_resume_checkpoint_returns_none_when_nothing_to_resume(tmp_path: Path):
    assert pick_resume_checkpoint(tmp_path, None) is None


def test_save_training_state_round_trips_optimizer_state(tmp_path: Path):
    model = torch.nn.Linear(3, 1)
    # AdamW (what train() actually uses) always populates per-parameter state after a step, unlike
    # plain SGD, which stores nothing unless momentum is enabled. This step makes the round trip
    # below representative of the checkpoints train() really writes.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(2, 3)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    history = [{"epoch": 1, "val_auc": 0.9}]

    checkpoint = save_training_state(
        tmp_path / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        threshold=0.7,
        training_source="unit-test",
        metrics={"roc_auc": 0.91, "balanced_accuracy": 0.85},
        best_auc=0.91,
        history=history,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["history"] == history
    assert checkpoint["best_auc"] == 0.91
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["scheduler_state"]["last_epoch"] == 0
    assert (tmp_path / "latest.pt").is_file()

    reloaded = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
    assert reloaded["epoch"] == 3
    assert reloaded["history"] == history
