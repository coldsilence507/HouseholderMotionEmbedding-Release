import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Pixel-wise orthogonal Transformation (per frame) ----------
class HouseholderTransform(nn.Module):
    """
    Build an orthogonal transform Q(u,v) from a per-pixel 6D Plücker embedding.
    Applies Q to the C-dim channel vector at each pixel. Keeps spatial i.i.d.
    """
    def __init__(self, C, k_reflections=2, seed=0, learnable=False, eps=1e-9):
        super().__init__()
        assert k_reflections >= 1
        self.C = C
        self.k = k_reflections
        self.eps = eps

        g = torch.Generator()
        g.manual_seed(seed)
        W = torch.randn(self.k, C, 6, generator=g) / math.sqrt(6.0)  # stable scaling

        if learnable:
            self.W = nn.Parameter(W)     # trainable mapping Plücker->v
        else:
            self.register_buffer("W", W) # frozen zero-shot mapping


    @staticmethod
    def _apply_householder(v, x, eps):
        """
        v, x: [B, N, C, H, W]
        Per-pixel reflection across v (normalized over channel dim C).
        """
        v = F.normalize(v, dim=2, eps=eps)                # norm over channels
        ## Hx = x - 2 v (v^T x)
        proj = (v * x).sum(dim=2, keepdim=True)           # [B,N,1,H,W]
        
        return x - 2.0 * v * proj                         # [B,N,C,H,W]
    
    def forward(self, noise, plucker):
        """
        noise:   [B,N,C,H,W]
        plucker: [B,N,H,W,6]
        returns: [B,N,C,H,W]  (same dtype/device as noise)
        """
        B, N, C, H, W = noise.shape
        assert C == self.C, f"expected C={self.C}, got {C}"
        assert plucker.shape == (B, N, H, W, 6)

        # Compute k Householder vectors v_i = W_i @ p  → [B,N,k,C,H,W]
        # einsum: ('k c s, b n h w s -> b n k c h w')
        v = torch.einsum('kcs,bnhws->bnkchw', self.W, plucker)

        y = noise
        for i in range(self.k):
            vi = v[:, :, i, :, :, :]                      # [B,N,C,H,W]
            # print(vi.shape)
            y  = self._apply_householder(vi, y, self.eps)
        return y
