import torch
from soznet.losses import MultilabelFocalLoss, hemispheric_penalty


def test_focal_loss_backprop():
    logits = torch.randn(4, 6, requires_grad=True)
    targets = (torch.rand(4, 6) > 0.5).float()
    loss = MultilabelFocalLoss(gamma=2.0)(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss).item()


def test_hemispheric_penalty_is_low_for_unilateral():
    logits = torch.zeros(1, 6)
    logits[0, 0] = 10.0  # strongly left, right stays at sigmoid(0)=0.5
    left_high = hemispheric_penalty(logits)

    logits_both = torch.zeros(1, 6)
    logits_both[0, 0] = 10.0
    logits_both[0, 1] = 10.0
    bilateral = hemispheric_penalty(logits_both)

    assert bilateral > left_high
