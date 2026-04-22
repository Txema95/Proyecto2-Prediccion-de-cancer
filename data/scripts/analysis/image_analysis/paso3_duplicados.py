"""Paso 3: duplicados exactos (MD5) y grupos por similitud de dHash (Hamming)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from configuracion import directorio_salida, manifest_filtrar_seleccionado, raiz_proyecto, ruta_dataset_kvasir


def md5_archivo(ruta: Path, tamano_bloque: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(ruta, "rb") as f:
        while True:
            bloque = f.read(tamano_bloque)
            if not bloque:
                break
            digest.update(bloque)
    return digest.hexdigest()


def dhash_bits(ruta: Path, tamano_hash: int = 8) -> int:
    """dHash en 64 bits (grises 9x8, diferencias horizontales)."""
    with Image.open(ruta) as img:
        img = img.convert("L").resize((tamano_hash + 1, tamano_hash), Image.Resampling.LANCZOS)
        pix = np.asarray(img, dtype=np.uint8)
    diff = pix[:, 1:] > pix[:, :-1]
    bits = diff.flatten()
    valor = 0
    for b in bits:
        valor = (valor << 1) | int(b)
    return valor


def hamming_64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


class UnionFind:
    def __init__(self, n: int) -> None:
        self.padre = list(range(n))
        self.rango = [0] * n

    def buscar(self, i: int) -> int:
        if self.padre[i] != i:
            self.padre[i] = self.buscar(self.padre[i])
        return self.padre[i]

    def unir(self, i: int, j: int) -> None:
        ri, rj = self.buscar(i), self.buscar(j)
        if ri == rj:
            return
        if self.rango[ri] < self.rango[rj]:
            self.padre[ri] = rj
        elif self.rango[ri] > self.rango[rj]:
            self.padre[rj] = ri
        else:
            self.padre[rj] = ri
            self.rango[ri] += 1


def ejecutar_duplicados(
    manifest_csv: Path,
    raiz_dataset: Path,
    umbral_hamming: int,
    max_pares_reporte: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Lee filas con seleccionado=True del manifest.
    Devuelve: tabla md5, tabla pares cercanos (limitada), resumen JSON-serializable.
    """
    man = pd.read_csv(manifest_csv)
    sel = manifest_filtrar_seleccionado(man)
    rutas: list[Path] = []
    for _, row in sel.iterrows():
        if "ruta_absoluta" in row and pd.notna(row["ruta_absoluta"]):
            rutas.append(Path(row["ruta_absoluta"]))
        else:
            rutas.append((raiz_dataset / row["ruta_relativa"]).resolve())

    filas_md5: list[dict] = []
    errores: list[str] = []
    for ruta in rutas:
        if not ruta.is_file():
            errores.append(f"no_existe: {ruta}")
            continue
        try:
            digest = md5_archivo(ruta)
            dh = dhash_bits(ruta)
        except OSError as exc:
            errores.append(f"{ruta}: {exc}")
            continue
        try:
            rel = str(ruta.relative_to(raiz_dataset.resolve())).replace("\\", "/")
        except ValueError:
            rel = str(ruta).replace("\\", "/")
        filas_md5.append(
            {
                "ruta_relativa": rel,
                "ruta_absoluta": str(ruta),
                "md5": digest,
                "dhash": format(dh, "016x"),
            }
        )

    df_h = pd.DataFrame(filas_md5)
    resumen: dict = {
        "archivos_hasheados": int(len(df_h)),
        "umbral_hamming": umbral_hamming,
        "errores_lectura": errores[:50],
        "total_errores_lectura": len(errores),
    }

    if df_h.empty:
        return df_h, pd.DataFrame(columns=["ruta_a", "ruta_b", "hamming"]), resumen

    # Duplicados exactos por MD5
    grupos_md5 = df_h.groupby("md5").size()
    dup_md5 = grupos_md5[grupos_md5 > 1]
    resumen["grupos_md5_duplicados"] = int(len(dup_md5))
    resumen["archivos_en_grupos_md5_duplicados"] = int(dup_md5.sum())

    # dHash como entero para Hamming
    hashes = [int(row["dhash"], 16) for _, row in df_h.iterrows()]
    n = len(hashes)
    uf = UnionFind(n)

    pares: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_64(hashes[i], hashes[j])
            if dist <= umbral_hamming:
                uf.unir(i, j)
                if len(pares) < max_pares_reporte:
                    pares.append((i, j, dist))

    comp: dict[int, list[int]] = {}
    for i in range(n):
        raiz = uf.buscar(i)
        comp.setdefault(raiz, []).append(i)

    grupos_sim = [idxs for idxs in comp.values() if len(idxs) > 1]
    resumen["componentes_similitud_mayor_1"] = int(len(grupos_sim))
    resumen["pares_listados_en_csv"] = int(len(pares))

    rutas_lista = df_h["ruta_absoluta"].tolist()
    filas_pares = []
    for i, j, dist in pares:
        filas_pares.append({"ruta_a": rutas_lista[i], "ruta_b": rutas_lista[j], "hamming": dist})

    df_pares = pd.DataFrame(filas_pares)

    # Resumen por componente (tamaño y ejemplo rutas)
    resumen["componentes_top"] = []
    grupos_sim.sort(key=len, reverse=True)
    for idxs in grupos_sim[:30]:
        muestra = [rutas_lista[k] for k in idxs[:5]]
        resumen["componentes_top"].append({"tamano": len(idxs), "muestra_rutas": muestra})

    return df_h, df_pares, resumen


def guardar_duplicados(df_hashes: pd.DataFrame, df_pares: pd.DataFrame, resumen: dict, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    df_hashes.to_csv(salida / "paso3_hashes_por_archivo.csv", index=False, encoding="utf-8-sig")
    df_pares.to_csv(salida / "paso3_pares_dhash_cercanos.csv", index=False, encoding="utf-8-sig")
    with open(salida / "paso3_duplicados_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def main() -> None:
    raiz = raiz_proyecto()
    out = directorio_salida(raiz)
    manifest = out / "paso2_manifest_muestreo.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Ejecute antes el paso 2 o cree: {manifest}")

    dataset = ruta_dataset_kvasir(raiz)
    df_h, df_p, res = ejecutar_duplicados(manifest, dataset, umbral_hamming=8, max_pares_reporte=2000)
    guardar_duplicados(df_h, df_p, res, out)
    print(f"Paso 3 listo. Hashes: {out / 'paso3_hashes_por_archivo.csv'}")
    print(f"Pares cercanos (muestra): {out / 'paso3_pares_dhash_cercanos.csv'}")
    print(f"Grupos MD5 con duplicado: {res.get('grupos_md5_duplicados', 0)}")


if __name__ == "__main__":
    main()
