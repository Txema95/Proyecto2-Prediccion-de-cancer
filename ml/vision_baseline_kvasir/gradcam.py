"""
Grad-CAM (Selvaraju et al., 2017) para ResNet-18: mapa de calor sobre `layer4`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


def grad_cam_resnet18(
    model: nn.Module,
    input_bchw: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    """
    `input_bchw`: lote 1x3xHxW en el dispositivo del modelo.
    Devuelve mapa 2D (H, W) en [0, 1] interpolado al tamaño espacial de `input_bchw` (p. ej. 224x224).
    """
    if input_bchw.dim() != 4 or input_bchw.size(0) != 1:
        raise ValueError("Se espera un tensor 1x3xHxW.")
    device = input_bchw.device
    _c, h, w = int(input_bchw.shape[1]), int(input_bchw.shape[2]), int(input_bchw.shape[3])
    t = input_bchw.clone().detach().to(device)
    t.requires_grad_(True)
    act_list: list[torch.Tensor] = []
    grad_list: list[torch.Tensor] = []

    def f_hook(_m, _i, o: torch.Tensor) -> None:
        act_list.append(o)

    def b_hook(_m, _go, g_out: tuple[torch.Tensor, ...]) -> None:
        grad_list.append(g_out[0])

    h1 = model.layer4.register_forward_hook(f_hook)
    h2 = model.layer4.register_full_backward_hook(b_hook)
    model.eval()
    try:
        with torch.enable_grad():
            out = model(t)
            s = out[0, int(class_idx)]
            model.zero_grad()
            s.backward()
        if not act_list or not grad_list:
            raise RuntimeError("No se capturaron activaciones o gradientes en layer4.")
        a = act_list[0]
        g = grad_list[0]
        pesos = g.mean(dim=(2, 3), keepdim=True)
        cam = (pesos * a).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).detach()
        cmin = float(cam.min().item())
        cmax = float(cam.max().item())
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = torch.zeros_like(cam)
        cam2 = cam.view(1, 1, cam.size(0), cam.size(1))
        up = F.interpolate(cam2, size=(h, w), mode="bilinear", align_corners=False)
        return up.squeeze().cpu().numpy().astype(np.float64)
    finally:
        h1.remove()
        h2.remove()


def superponer_heatmap_sobre_imagen(
    imagen_rgb: Image.Image,
    heatmap: np.ndarray,
    tam: tuple[int, int] = (224, 224),
    alfa: float = 0.45,
) -> np.ndarray:
    """
    Redimensiona la imagen, aplica colormap al mapa (mismo `tam`) y mezcla.
    Devuelve uint8 (H, W, 3).
    """
    im = np.array(imagen_rgb.convert("RGB").resize(tam), dtype=np.float32) / 255.0
    if heatmap.shape[0] != tam[0] or heatmap.shape[1] != tam[1]:
        raise ValueError("heatmap y tam deben coincidir en espacial.")
    cmap = plt.get_cmap("jet")
    color = cmap(heatmap)[..., :3]
    out = (1.0 - alfa) * im + alfa * color
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


