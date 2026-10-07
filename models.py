"""
models.py — ACF-XAI-FL backbone.

3-D ResNet-34 encoder  +  MRI Modality-Aware Attention Module (MMAM, Eqs. 10-18)
+  U-Net decoder (segmentation)  +  grade-classification head
+  auxiliary modality classifier psi (MDGP, Eq. 19)
+  projection heads for CEA explanation embedding (Eq. 37) and FTPMB latent (Sec. 3.7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- ResNet-34 (3-D)
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                                      nn.InstanceNorm3d(out_ch, affine=True))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt, inplace=True)


class ResNet34Encoder3D(nn.Module):
    """ResNet-34 layout [3, 4, 6, 3]; returns skip features at strides 2, 4, 8, 16, 32."""

    def __init__(self, in_channels: int = 4, widths=(64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], 7, 2, 3, bias=False),
            nn.InstanceNorm3d(widths[0], affine=True), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool3d(3, 2, 1)
        self.layer1 = self._make(widths[0], widths[0], 3, 1)
        self.layer2 = self._make(widths[0], widths[1], 4, 2)
        self.layer3 = self._make(widths[1], widths[2], 6, 2)
        self.layer4 = self._make(widths[2], widths[3], 3, 2)
        self.out_channels = widths

    @staticmethod
    def _make(in_ch, out_ch, n, stride):
        layers = [BasicBlock3D(in_ch, out_ch, stride)]
        layers += [BasicBlock3D(out_ch, out_ch, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        s0 = self.stem(x)                 # /2
        s1 = self.layer1(self.pool(s0))   # /4
        s2 = self.layer2(s1)              # /8
        s3 = self.layer3(s2)              # /16
        s4 = self.layer4(s3)              # /32  -> F (Eq. 10)
        return [s0, s1, s2, s3, s4]


# --------------------------------------------------------------------------- MMAM
class MMAM(nn.Module):
    """Channel attention from GAP (Eq. 12) + GMP (Eq. 13) descriptors -> 2 FC gating (Eq. 15) -> recalibration (Eq. 18)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(nn.Linear(2 * channels, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, f):
        gap = f.mean(dim=(2, 3, 4))
        gmp = f.amax(dim=(2, 3, 4))
        a = self.fc(torch.cat([gap, gmp], dim=1))           # attention vector a in (0,1)^C
        return f * a[:, :, None, None, None], a


# --------------------------------------------------------------------------- decoder
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False), nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False), nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNetDecoder3D(nn.Module):
    def __init__(self, enc_ch=(64, 64, 128, 256, 512), out_classes=3):
        super().__init__()
        c0, c1, c2, c3, c4 = enc_ch
        self.up3 = UpBlock(c4, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.up0 = UpBlock(64, c0, 32)
        self.head = nn.Conv3d(32, out_classes, 1)

    def forward(self, feats, out_size):
        s0, s1, s2, s3, f = feats
        x = self.up3(f, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        x = F.interpolate(x, size=out_size, mode="trilinear", align_corners=False)
        return self.head(x)


# --------------------------------------------------------------------------- full model
class ACFXAIFLNet(nn.Module):
    """
    forward(x) returns a dict with
        seg_logits  (B, 3, D, H, W)   TC/WT/ET logits
        cls_logits  (B, num_grades)   tumour grade logits
        feat        (B, 512, d, h, w) MMAM-recalibrated bottleneck F'  (Grad-CAM target layer)
        z           (B, D_z)          FTPMB latent embedding
        mod_logits  (B, K)            auxiliary modality / hospital logits (psi, Eq. 19)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.use_mmam = cfg.get("use_mmam", True)
        self.encoder = ResNet34Encoder3D(cfg.get("in_channels", 4))
        c = self.encoder.out_channels
        self.mmam = MMAM(c[3], cfg.get("mmam_reduction", 16))
        self.decoder = UNetDecoder3D((c[0], c[0], c[1], c[2], c[3]), cfg.get("num_seg_classes", 3))
        self.cls_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(c[3], cfg.get("num_grades", 3)))
        h = cfg.get("mdgp_hidden", 256)
        self.mod_head = nn.Sequential(nn.Linear(c[3], h), nn.ReLU(inplace=True),
                                      nn.Linear(h, cfg.get("num_clients", 4)))       # psi
        self.latent_proj = nn.Sequential(nn.Linear(c[3], cfg.get("ftpmb_embed_dim", 256)))
        # phi: explanation embedding of the Grad-CAM map pooled to 4x4x4 (Eq. 37)
        self.cam_pool = (4, 4, 4)
        self.expl_proj = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True),
                                       nn.Linear(128, cfg.get("cea_embed_dim", 256)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)          # Algorithm 1: w0 <- Xavier-uniform
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # parameter groups -------------------------------------------------------
    def shared_parameters(self):
        """Encoder (+MMAM): parameters where MDGP projection is applied."""
        return list(self.encoder.parameters()) + list(self.mmam.parameters())

    def forward(self, x):
        feats = self.encoder(x)
        f = feats[-1]
        if self.use_mmam:
            f, _ = self.mmam(f)
        feats = feats[:-1] + [f]
        pooled = f.mean(dim=(2, 3, 4))
        return {
            "seg_logits": self.decoder(feats, x.shape[2:]),
            "cls_logits": self.cls_head(pooled),
            "feat": f,
            "pooled": pooled,
            "z": self.latent_proj(pooled),
            "mod_logits": self.mod_head(pooled),
        }

    # Grad-CAM on the classification branch (Eq. 36) ------------------------
    def gradcam(self, feat, cls_logits, target=None, create_graph=False):
        if target is None:
            target = cls_logits.argmax(1)
        score = cls_logits.gather(1, target.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, feat, create_graph=create_graph, retain_graph=True)[0]
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)              # alpha_k
        cam = F.relu((weights * feat).sum(1, keepdim=True))           # H = ReLU(sum_k alpha_k A^k)
        cmin = cam.flatten(1).min(1)[0].view(-1, 1, 1, 1, 1)
        cmax = cam.flatten(1).max(1)[0].view(-1, 1, 1, 1, 1)
        return (cam - cmin) / (cmax - cmin + 1e-8)                     # scaled to [0, 1]

    def explanation_embedding(self, cam):
        pooled = F.adaptive_avg_pool3d(cam, self.cam_pool).flatten(1)
        return self.expl_proj(pooled)                                  # h_k = phi(H_k)
