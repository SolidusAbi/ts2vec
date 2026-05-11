import torch
from torch import nn
from torch.nn.functional import normalize


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer"""

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.n_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, D)
        """
        _, _, D = x.shape
        u = torch.randn(D, self.n_proj, device=x.device)
        u = normalize(u, p=2, dim=0)

        x_t = (x @ u).unsqueeze(-1) * self.t

        statistic = self._epps_pulley_statistic(x_t)
        return torch.mean(statistic)

        # A = torch.randn(x.size(-1), self.n_proj, device=x.device)
        # A = A.div_(A.norm(p=2, dim=0))
        # # compute the epps-pulley statistic
        # x_t = (x @ A).unsqueeze(-1) * self.t
        # err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # statistic = (err @ self.weights) * x.size(-2)
        # return statistic.mean()  # average over projections and time

    def _epps_pulley_statistic(self, x_t):
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights) * x_t.size(-3)
