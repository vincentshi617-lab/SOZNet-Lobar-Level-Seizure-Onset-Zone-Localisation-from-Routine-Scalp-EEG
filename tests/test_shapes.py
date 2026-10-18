import torch
from soznet.models import create_model


def test_three_models_forward():
    x = torch.randn(2, 18, 2048)
    for name in ["m1", "m2", "m3"]:
        model = create_model(
            name, chunk_size=2048, num_electrodes=18,
            F1=8, F2=16, D=2, num_classes=6,
            kernel_1=64, kernel_2=16, dropout=0.3,
        ).eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6), f"{name} produced {out.shape}"
