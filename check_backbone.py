import sys

import torch

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cifake/run_002/best.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
config = checkpoint.get("model_config", {})

print(path)
print("  backbone       :", config.get("backbone"))
print("  freeze_backbone:", config.get("freeze_backbone"))
print("  use_frequency  :", config.get("use_frequency"))
print("  trained on     :", checkpoint.get("training_source"))
