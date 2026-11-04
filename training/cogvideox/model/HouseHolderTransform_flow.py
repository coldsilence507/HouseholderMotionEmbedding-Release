import torch
import torch.nn as nn

def _norm_or_zero(v, eps=1e-6):
    n = v.norm(dim=-1, keepdim=True)
    ok = (n > eps).to(v.dtype)
    vhat = torch.where(ok > 0, v / n.clamp_min(eps), torch.zeros_like(v))
    return vhat, ok

def _apply_householder(v, x, eps=1e-6):
    """
    v:  [..., C] reflection normal per pixel
    x:  [..., C] channel vector per pixel
    """
    vhat, ok = _norm_or_zero(v, eps)
    proj = (vhat * x).sum(dim=-1, keepdim=True)
    y = x - 2.0 * vhat * proj
    return torch.where(ok > 0, y, x)

class FlowHouseholder(nn.Module):
    """
    Palindromic Householder transformer using pixel-wise optical flow.
    Guarantees: Q(flip flow) = inverse (Q(-f) = Q(f)^-1)
    Inputs:
      - noise: [B, F, C, H, W]
      - flow : [B, F, H, W, 2]  (u,v) in pixels (RAFT-style)
    """
    def __init__(self, C, P=2, tau=1e-6, seed=0, device=None, dtype=None):
        super().__init__()
        self.C, self.P, self.tau = C, P, tau
        fw = {"device": device, "dtype": dtype or torch.float32}
        g = torch.Generator(device=device if device is not None else "cpu")
        g.manual_seed(seed)

        # Feature dims
        # Even (Fe=3+1=4): [ |u|, |v|, ||f||,      (optional scale-invariant bias via explicit b) ]
        # We keep bias as a parameter, so no homogeneous "1" channel needed.
        self.Fe = 3
        # Odd  (Fo=4): [ u, v, f_perp_x, f_perp_y ]  where f_perp = (-v, u)
        self.Fo = 4

        We = torch.randn(P, C, self.Fe, generator=g, **fw) / (self.Fe ** 0.5)
        Wo = torch.randn(P, C, self.Fo, generator=g, **fw) / (self.Fo ** 0.5)
        b  = 0.1 * torch.ones(P, C, **fw)  # nonzero even bias to avoid a ≈ 0 collapse

        self.W_even = nn.Parameter(We)
        self.W_odd  = nn.Parameter(Wo)
        self.b      = nn.Parameter(b)

        # Sensitivity/robustness knobs (register as buffers so they move with .to()/.cuda())
        self.register_buffer("alpha_odd", torch.tensor(0.9, **fw))   # ≠ 1 to avoid pi-rotation degeneracy
        self.register_buffer("gamma_flow", torch.tensor(1.5, **fw))  # odd gain
        self.register_buffer("beta_odd", torch.tensor(1.0, **fw))    # tanh slope (odd MLP); 0 -> linear

    @staticmethod
    def _normalize_flow(flow, H, W, mode="pixels"):
        """
        Convert flow to a scale comparable across resolutions.
        - mode="pixels": scale to fraction of image (divide by [W, H]).
        - You can adapt to your coordinate convention as needed.
        """
        if mode == "pixels":
            # Normalize u by W, v by H
            scale = flow.new_tensor([W, H]).view(*(1,)* (flow.dim()-1), 2)
            return flow / scale
        else:
            return flow

    @staticmethod
    def _build_feats(flow_norm):
        """
        flow_norm: [B, F, H, W, 2] (normalized flow)
        Returns:
          even: [B,F,H,W, 3]  = [ |u|, |v|, ||f|| ]
          odd : [B,F,H,W, 4]  = [ u, v, -v, u ]  (f and f_perp)
        """
        u = flow_norm[..., 0:1]
        v = flow_norm[..., 1:2]
        mag = torch.sqrt((u*u + v*v).clamp_min(1e-12))
        even = torch.cat([u.abs(), v.abs(), mag], dim=-1)               # 3
        f_perp = torch.cat([-v, u], dim=-1)                             # 2
        odd = torch.cat([u, v, f_perp], dim=-1)                         # 4 (odd in flow)
        return even, odd, mag

    def _stabilize_pair(self, a, o, ref_dir, tol=1e-3, jitter=1e-2):
        """
        Avoid pi-rotation degeneracy when ||a|| ~ ||o|| by slightly scaling 'o'
        with a deterministic sign derived from ref_dir (independent of flow sign).
        a,o: [B,F,H,W,P,C]; ref_dir: [B,F,H,W,2] or [B,F,H,W,3]; only sign used.
        """
        aN = a.norm(dim=-1, keepdim=True)
        oN = o.norm(dim=-1, keepdim=True)
        near = ((aN - oN).abs() <= tol * (aN.maximum(oN) + 1e-8))

        # build a stable sign from the dominant component of ref_dir
        if ref_dir.size(-1) == 2:
            # use flow direction (but we only take sign, and then make it even by abs in next step)
            dom = torch.where(ref_dir[..., 0:1].abs() >= ref_dir[..., 1:2].abs(),
                              ref_dir[..., 0:1], ref_dir[..., 1:2])
        else:
            # if passing a 3D ref (e.g., some camera dir), use its dominant component
            absd = ref_dir.abs()
            idx = absd.argmax(dim=-1, keepdim=True)
            dom = torch.gather(ref_dir, dim=-1, index=idx)

        sgn = torch.sign(dom)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)  # treat exact zero as +1
        while sgn.dim() < a.dim(): sgn = sgn.unsqueeze(-1)
        scale = 1.0 + jitter * sgn.type_as(o)
        o = torch.where(near, scale * o, o)
        return a, o

    def forward(self, noise, flow, flow_unit="pixels"):
        """
        noise: [B, F, C, H, W]
        flow : [B, F, H, W, 2]  (u,v), typically RAFT output in pixel units
        flow_unit: "pixels" (default) scales by image size to be resolution-invariant
        """
        assert noise.dim() == 5 and flow.dim() == 5
        B, F, C, H, W = noise.shape
        assert C == self.C, f"C mismatch: got C={C}, expected {self.C}"
        assert flow.shape[:2] == (B, F) and flow.shape[2:4] == (H, W) and flow.shape[-1] == 2

        # Per-pixel channel vectors
        x = noise.permute(0,1,3,4,2).contiguous()                  # [B,F,H,W,C]

        # Normalize flow by image size to get stable scale across datasets
        flow_norm = self._normalize_flow(flow, H, W, mode=flow_unit)

        # Build parity-clean features
        even, odd, mag = self._build_feats(flow_norm)              # even: [..,3], odd: [..,4]

        # Odd gain and odd nonlinearity (preserve odd parity)
        gamma = float(self.gamma_flow.item())
        beta  = float(self.beta_odd.item())
        odd_nl = torch.tanh(beta * odd) if beta != 0.0 else odd    # tanh is odd: tanh(-x) = -tanh(x)
        odd_nl = gamma * odd_nl

        # Project to per-pixel reflection vectors for all pairs
        # a,o: [B,F,H,W,P,C]
        a = (torch.einsum('pcf,bnhwf->bnhwpc', self.W_even, even)
             + self.b.view(1,1,1,1,self.P,self.C))
        o = torch.einsum('pcf,bnhwf->bnhwpc', self.W_odd, odd_nl)

        # Anti-pi stabilization (uses flow_norm as a stable reference; independent of sign flip because of tolerances)
        # a, o = self._stabilize_pair(a, o, ref_dir=flow_norm, tol=1e-3, jitter=1e-2)

        # Off-pi scaling for odd term
        alpha = float(self.alpha_odd.item())
        v_plus  = a + alpha * o
        v_minus = a - alpha * o

        # Palindromic product: A(f) then B(f)
        y = x
        for p in range(self.P):                 # A = Π H(v_plus)_p
            y = _apply_householder(v_plus[..., p, :], y, eps=self.tau)
        for p in reversed(range(self.P)):       # B = Π_rev H(v_minus)_p
            y = _apply_householder(v_minus[..., p, :], y, eps=self.tau)

        return y.permute(0,1,4,2,3).contiguous()
