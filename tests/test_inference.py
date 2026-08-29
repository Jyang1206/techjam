import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from PIL import Image

from traceguard.model import ModelConfig, TraceGuard
from traceguard.predict import predict_directory


def test_directory_prediction_writes_required_json_contract(tmp_path):
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    Image.new("RGB", (256, 256), (80, 120, 160)).save(image_directory / "sample.png")

    model = TraceGuard(ModelConfig(pretrained=False))
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.checkpoint_config(),
            "threshold": 0.5,
            "temperature": 1.0,
        },
        checkpoint,
    )
    output = tmp_path / "predictions.json"

    results = predict_directory(image_directory, checkpoint, output, device="cpu", tta="none")
    serialized = json.loads(output.read_text(encoding="utf-8"))

    assert serialized == results
    assert set(serialized[0]) == {"image_path", "pred"}
    assert serialized[0]["image_path"].endswith("sample.png")
    assert 0 <= serialized[0]["pred"] <= 1
