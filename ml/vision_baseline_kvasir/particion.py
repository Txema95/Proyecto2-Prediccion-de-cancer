from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def leer_manifest(ruta: Path) -> list[dict[str, str]]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro manifest: {ruta}")
    with ruta.open("r", newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    requeridas = {"filepath", "label", "source", "group_id", "image_id"}
    if not filas:
        raise ValueError("Manifest vacio.")
    faltan = requeridas - set(filas[0].keys())
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    return filas


def validar_proporciones(tren: float, val: float, prueba: float) -> None:
    s = tren + val + prueba
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Las proporciones deben sumar 1.0; recibido {s!r}.")


def asignar_grupos_a_splits(
    filas: list[dict[str, str]],
    ratio_train: float,
    ratio_val: float,
    ratio_test: float,
    semilla: int,
) -> dict[str, str]:
    grupos_a_filas: dict[str, list[dict[str, str]]] = defaultdict(list)
    for f in filas:
        grupos_a_filas[f["group_id"]].append(f)

    splits_nombres = ("train", "val", "test")
    obj = {"train": ratio_train, "val": ratio_val, "test": ratio_test}
    total = len(filas)
    objetivo = {n: total * r for n, r in obj.items()}
    conteo = {n: 0 for n in splits_nombres}
    asig: dict[str, str] = {}
    pares = list(grupos_a_filas.items())
    rnd = random.Random(semilla)
    rnd.shuffle(pares)
    pares.sort(key=lambda t: len(t[1]), reverse=True)

    for _gid, filas_g in pares:
        t = len(filas_g)
        deficit = {n: objetivo[n] - conteo[n] for n in splits_nombres}
        ok = [n for n in splits_nombres if deficit[n] >= t]
        if ok:
            eleg = max(ok, key=lambda n: deficit[n])
        else:
            eleg = max(splits_nombres, key=lambda n: deficit[n])
        asig[_gid] = eleg
        conteo[eleg] += t
    return asig


def dividir_por_etiqueta(
    filas: list[dict[str, str]],
    ratio_train: float,
    ratio_val: float,
    ratio_test: float,
    semilla: int,
) -> dict[str, str]:
    por_bloque: dict[str, list[dict[str, str]]] = defaultdict(list)
    for f in filas:
        por_bloque[f["label"]].append(f)
    asig_total: dict[str, str] = {}
    for i, (_et, bloque) in enumerate(sorted(por_bloque.items(), key=lambda p: p[0])):
        a = asignar_grupos_a_splits(
            filas=bloque,
            ratio_train=ratio_train,
            ratio_val=ratio_val,
            ratio_test=ratio_test,
            semilla=semilla + i,
        )
        cruce = set(asig_total) & set(a)
        if cruce:
            raise ValueError("group_id repetido entre bloques (no deberia ocurrir con manifest sano).")
        asig_total.update(a)
    return asig_total


def validar_fuga(filas: list[dict[str, str]], asig: dict[str, str]) -> None:
    s_a_g: dict[str, set[str]] = defaultdict(set)
    for f in filas:
        s_a_g[asig[f["group_id"]]].add(f["group_id"])
    t, v, te = s_a_g["train"] & s_a_g["val"], s_a_g["train"] & s_a_g["test"], s_a_g["val"] & s_a_g["test"]
    if t or v or te:
        raise ValueError("Fuga de group_id entre particiones de train/val/test.")


def escribir_splits(
    filas: list[dict[str, str]], asig: dict[str, str], ruta: Path, semilla: int
) -> dict[str, Any]:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    col = ["filepath", "label", "source", "group_id", "image_id", "split"]
    with ruta.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=col)
        w.writeheader()
        for fila in filas:
            s = {**fila, "split": asig[fila["group_id"]]}
            w.writerow(s)
    return resumen_por_split(filas, asig, ruta, semilla)


def resumen_por_split(
    filas: list[dict[str, str]], asig: dict[str, str], ruta: Path, semilla: int
) -> dict[str, Any]:
    c_global = Counter()
    c_det: dict[str, Counter[str]] = defaultdict(Counter)
    for f in filas:
        s = asig[f["group_id"]]
        c_global[s] += 1
        c_det[s][f["label"]] += 1
    out: dict[str, Any] = {
        "ruta_salida": str(ruta),
        "semilla": semilla,
        "conteo_por_split": dict(c_global),
        "etiquetas_por_split": {k: dict(v) for k, v in c_det.items()},
    }
    return out


def imprimir_resumen(res: dict[str, Any]) -> None:
    print(f"Escrito: {res['ruta_salida']}", flush=True)
    print(f"Semilla: {res['semilla']}", flush=True)
    for s in ("train", "val", "test"):
        t = res["conteo_por_split"].get(s, 0)
        d = res["etiquetas_por_split"].get(s, {})
        print(f"  {s}: n={t}  por_clase( indice->n )= {d}", flush=True)
