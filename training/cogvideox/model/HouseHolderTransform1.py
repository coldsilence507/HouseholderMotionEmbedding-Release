import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def canonicalize_plucker(
    d: torch.Tensor,   # [..., 3]
    m: torch.Tensor,   # [..., 3]
    eps: float = 1e-9
):
    """
    Returns:
      d_hat: [...,3]  unit direction, optionally sign-canonicalized
      m_can: [...,3]  moment, re-orthogonalized to d_hat; scaled consistently with d
      flipped_mask: [...,] bool (True where a global sign flip was applied; all False if oriented=True)

    What it does:
      1) Normalize direction: d_hat = d / ||d||.
      2) Scale moment consistently: m <- m / ||d||  (preserve Plücker homogeneity).
      3) Enforce orthogonality: m_can = m - (m·d_hat) d_hat.
      4) If oriented=False (unoriented line), fix global sign of d_hat by the max-abs rule:
         find k = argmax(|d_hat_k|); if d_hat_k < 0, flip both d_hat and m_can.
    """
    assert d.shape[-1] == 3 and m.shape[-1] == 3 and d.shape[:-1] == m.shape[:-1], "shape mismatch"

    # 1) unit direction
    dn = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(eps)
    d_hat = d / dn

    # 2) consistent scaling of m (keep the same representative in the projective class)
    m_scaled = m / dn

    # 3) re-orthogonalize m to d_hat (numeric cleanup)
    md = (m_scaled * d_hat).sum(dim=-1, keepdim=True)
    m_can = m_scaled - md * d_hat

    return d_hat, m_can

def _norm_or_zero(v, dim=-1, eps=1e-6):
    n = torch.linalg.norm(v, dim=dim, keepdim=True)
    ok = (n > eps).to(v.dtype)
    vhat = torch.where(ok > 0, v / n.clamp_min(eps), torch.zeros_like(v))
    return vhat, ok

def _apply_householder(v, x, eps=1e-6):
    vhat, ok = _norm_or_zero(v, dim=-1, eps=eps)   # [...,C]
    proj = (vhat * x).sum(dim=-1, keepdim=True)
    y = x - 2.0 * vhat * proj
    return y


class HouseholderTransform(nn.Module):
    """
    Guarantees Q(d,-m) = Q(d,m)^-1 for any number of pairs P.
    noise: [B,F,C,H,W], d_hat/m_can: [B,F,H,W,3]
    """
    def __init__(self, C, P=3, seed=0, tau=1e-6, device=None, dtype=None):
        super().__init__()
        self.C, self.P, self.tau = C, P, tau
        fw = {"device": device, "dtype": dtype or torch.float32}
        g = torch.Generator(device=device if device is not None else "cpu"); g.manual_seed(seed)

        # Feature dims: even=[|d|(3), |m|(3), ||m||(1), 1] (8), odd=[m(3), d×m(3)] (6)
        self.Fe, self.Fo = 8, 6
        We = torch.randn(P, C, self.Fe, generator=g, **fw) / (self.Fe**0.5)
        Wo = torch.randn(P, C, self.Fo, generator=g, **fw) / (self.Fo**0.5)
        b  = 0.1 * torch.ones(P, C, **fw)  # nonzero even bias avoids collapse
        self.W_even = nn.Parameter(We)
        self.W_odd  = nn.Parameter(Wo)
        self.b      = nn.Parameter(b)
        
        # Sensitivity/robustness knobs (register as buffers so they move with .to()/.cuda())
        # self.register_buffer("alpha_odd", torch.tensor(0.9, **fw))   # ≠ 1 to avoid pi-rotation degeneracy
        # self.register_buffer("gamma_flow", torch.tensor(1.5, **fw))  # odd gain
        # self.register_buffer("beta_odd", torch.tensor(1.0, **fw))    # tanh slope (odd MLP); 0 -> linear

    @staticmethod
    def _build_feats(d_hat, m_can):
        abs_d, abs_m = d_hat.abs(), m_can.abs()
        dist  = m_can.norm(dim=-1, keepdim=True)
        ones  = torch.ones_like(dist)
        even = torch.cat([abs_d, abs_m, dist, ones], dim=-1)          # [B,F,H,W,8]
        odd  = torch.cat([m_can, torch.linalg.cross(d_hat, m_can)], dim=-1)  # [B,F,H,W,6] (odd in m)
        return even, odd

    def forward(self, noise, d_hat, m_can, alpha=0.9):
        B,F,C,H,W = noise.shape
        x = noise.permute(0,1,3,4,2).contiguous()   # [B,F,H,W,C]
        even, odd = self._build_feats(d_hat, m_can)
        
        # magnify the sensitivity of Q
        # Odd temperature & odd nonlinearity (preserves parity) for stronger odd path

        gamma = 1.5
        beta  = 1.0
        odd_nl = torch.tanh(beta * odd) if beta != 0.0 else odd    # tanh is odd: tanh(-x) = -tanh(x)
        odd_nl = gamma * odd_nl        
        
        # Build v+ and v- for all pairs: [B,F,H,W,P,C]
        a = torch.einsum('pcf,bnhwf->bnhwpc', self.W_even, even) + self.b.view(1,1,1,1,self.P,self.C)
        o = torch.einsum('pcf,bnhwf->bnhwpc', self.W_odd , odd_nl)
        
        # alpha ≠ 1.0 increases sensitivity; still preserves flip→inverse
        o = alpha * o
        v_plus, v_minus = a + o, a - o

        # A(d,m): forward order of H(v+)
        y = x
        for p in range(self.P):
            y = _apply_householder(v_plus[..., p, :], y, eps=self.tau)

        # B(d,m): reverse order of H(v-)
        for p in reversed(range(self.P)):
            y = _apply_householder(v_minus[..., p, :], y, eps=self.tau)

        return y.permute(0,1,4,2,3).contiguous()  # [B,F,C,H,W]
