from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from traceguard.data import ImageRecord
from traceguard.model import ModelConfig, TraceGuard
from traceguard.train import balanced_group_weights


def test_spatial_only_model_round_trips_checkpoint_config():
    config = ModelConfig(pretrained=False, normalization="clip", use_frequency=False, dropout=0.5)
    model = TraceGuard(config)
    output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2,)
    assert model.frequency is None
    assert model.checkpoint_config()["normalization"] == "clip"
    assert model.checkpoint_config()["use_frequency"] is False


def test_projected_backbone_uses_native_embedding_dimension():
    config = ModelConfig(
        backbone="vit_base_patch16_clip_224.openai",
        pretrained=False,
        use_frequency=False,
        backbone_projection=True,
        classifier_layernorm=False,
    )
    model = TraceGuard(config)

    assert model.backbone.num_classes == 512
    assert model.classifier[-1].in_features == 512


def test_classifying_extracted_features_matches_forward_pass():
    model = TraceGuard(ModelConfig(pretrained=False, use_frequency=False)).eval()
    images = torch.zeros(2, 3, 224, 224)

    with torch.inference_mode():
        direct = model(images)
        cached = model.classify_features(model.extract_features(images))

    assert torch.equal(direct, cached)


def test_balanced_group_weights_equalize_group_and_label_mass():
    records = [
        ImageRecord(Path(f"real_a_{index}.jpg"), 0, "real_a") for index in range(4)
    ]
    records += [ImageRecord(Path("real_b.jpg"), 0, "real_b")]
    records += [
        ImageRecord(Path(f"fake_a_{index}.jpg"), 1, "fake_a") for index in range(2)
    ]
    weights = balanced_group_weights(records)

    masses = {}
    for record, weight in zip(records, weights):
        masses[(record.label, record.group)] = masses.get((record.label, record.group), 0) + weight
    assert masses == {(0, "real_a"): 0.25, (0, "real_b"): 0.25, (1, "fake_a"): 0.5}
