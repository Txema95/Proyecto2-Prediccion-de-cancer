from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


def crear_resnet18(n_clases: int) -> nn.Module:
    pesos = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=pesos)
    m.fc = nn.Linear(m.fc.in_features, n_clases)
    return m


@torch.inference_mode()
def evaluar_cargador(
    modelo: nn.Module, cargador: DataLoader, dispositivo: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    modelo.eval()
    y_v, p_v = [], []
    for lote in cargador:
        x = lote["x"].to(dispositivo, non_blocking=True)
        y = lote["y"].to(dispositivo, non_blocking=True)
        logits = modelo(x)
        pr = torch.argmax(logits, dim=1)
        y_v.append(y.cpu().numpy())
        p_v.append(pr.cpu().numpy())
    return np.concatenate(y_v), np.concatenate(p_v)


@torch.inference_mode()
def inferencia_con_probabilidades(
    modelo: nn.Module, cargador: DataLoader, dispositivo: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Devuelve etiquetas reales, prediccion (argmax), matriz de probabilidades (N, C) e
    `image_id` por muestra (lista alineada con filas de probs).
    """
    modelo.eval()
    y_trozos, pr_trozos, prob_trozos = [], [], []
    ids: list[str] = []
    for lote in cargador:
        x = lote["x"].to(dispositivo, non_blocking=True)
        y = lote["y"].to(dispositivo, non_blocking=True)
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)
        pr = torch.argmax(logits, dim=1)
        y_trozos.append(y.cpu().numpy())
        pr_trozos.append(pr.cpu().numpy())
        prob_trozos.append(probs.cpu().numpy())
        lids = lote.get("image_id", [])
        if lids is None:
            lids = []
        elif isinstance(lids, str):
            lids = [lids]
        else:
            lids = list(lids)
        ids.extend(lids)
    y_all = np.concatenate(y_trozos, axis=0)
    p_all = np.concatenate(pr_trozos, axis=0)
    prob_all = np.concatenate(prob_trozos, axis=0)
    return y_all, p_all, prob_all, ids
