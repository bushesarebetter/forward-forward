import torch

import ff


def test_goodness_is_sum_of_squares():
    h = torch.tensor([[1.0, 2.0, 0.0], [0.5, -0.5, 3.0]])
    assert torch.allclose(ff.goodness(h), torch.tensor([5.0, 9.5]))


def test_loss_is_logistic_around_threshold():
    # at goodness == theta both terms are log(2)
    loss = ff.ff_loss(torch.tensor([2.0]), torch.tensor([2.0]), theta=2.0)
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)))
    good = ff.ff_loss(torch.tensor([10.0]), torch.tensor([0.0]), theta=2.0)
    bad = ff.ff_loss(torch.tensor([0.0]), torch.tensor([10.0]), theta=2.0)
    assert good < 0.2 < 5.0 < bad


def test_label_embedding():
    x = torch.rand(3, 64)
    e = ff.embed(x, torch.tensor([0, 4, 9]))
    assert torch.equal(e[:, :10].argmax(1), torch.tensor([0, 4, 9]))
    assert torch.equal(e[:, :10].sum(1), torch.ones(3))
    assert torch.equal(e[:, 10:], x[:, 10:])


def test_no_gradient_between_layers():
    torch.manual_seed(0)
    net = ff.FFNet([64, 32, 32, 32], theta=2.0, lr=0.01)
    x = torch.rand(8, 64)
    losses = net.layer_losses(x, x.flip(0))
    losses[2].backward()
    assert net.layers[0].weight.grad is None
    assert net.layers[1].weight.grad is None
    assert net.layers[2].weight.grad.abs().sum() > 0

    # backward on the summed loss gives layer 0 exactly the gradient of its own loss
    net.zero_grad()
    net.layer_losses(x, x.flip(0))[0].backward()
    g0 = net.layers[0].weight.grad.clone()
    net.zero_grad()
    sum(net.layer_losses(x, x.flip(0))).backward()
    assert torch.allclose(net.layers[0].weight.grad, g0)


def test_quick_run_beats_chance():
    hist = ff.run(hidden=100, epochs=5)
    assert hist["ff"][-1] > 0.3
    assert hist["bp"][-1] > 0.5
