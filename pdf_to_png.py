#!/usr/bin/env python3.13
"""
PDF → imagem  |  32 vCPUs  |  saída em pasta espelhada

Modos:
  python3.13 pdf_to_png.py arquivo.pdf
  python3.13 pdf_to_png.py pasta/ --format jpg -q 85

Saída (modo pasta):
  Doutrinas/AREA/livro.pdf  →  Doutrinas (jpg)/AREA/livro/page_0001.jpg

Pipeline por PDF:
  mp.Pool(32 workers, initializer)  →  cada worker abre o PDF UMA VEZ
  ThreadPoolExecutor(save_threads)  →  salva imagens em paralelo (I/O)
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _pip(*pkgs: str) -> None:
    os.system(f"{sys.executable} -m pip install -q {' '.join(pkgs)}")

try:
    import numpy as np
except ImportError:
    _pip("numpy"); import numpy as np

try:
    from PIL import Image
except ImportError:
    _pip("pillow"); from PIL import Image

try:
    import fitz
except ImportError:
    _pip("pymupdf"); import fitz


def _silence_mupdf() -> None:
    """Suprime warnings MuPDF em todos os níveis (Python + C)."""
    sys.stderr = open(os.devnull, "w")
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 2)
        os.close(null)
    except OSError:
        pass
    try:
        fitz.TOOLS.mupdf_warnings(False)
    except Exception:
        pass

_silence_mupdf()  # processo principal


DPI_DEFAULT     = 150
FORMAT_DEFAULT  = "jpg"
QUALITY_DEFAULT = 85

SAVE_OPTS = {
    "jpg":  dict(format="JPEG", quality=QUALITY_DEFAULT, optimize=True, progressive=True),
    "png":  dict(format="PNG",  compress_level=6),
    "webp": dict(format="WEBP", quality=QUALITY_DEFAULT, method=4),
}

def _ext(fmt: str) -> str:
    return "jpg" if fmt == "jpg" else fmt


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER (um por processo filho, inicializado uma vez)
# ══════════════════════════════════════════════════════════════════════════════

_doc  = None
_zoom = None

def _init_worker(pdf_path: str, zoom: float) -> None:
    global _doc, _zoom
    _silence_mupdf()
    _doc, _zoom = fitz.open(pdf_path), zoom

def _render_page(page_idx: int) -> tuple:
    page = _doc.load_page(page_idx)
    pix  = page.get_pixmap(matrix=fitz.Matrix(_zoom, _zoom),
                            alpha=False, colorspace=fitz.csRGB)
    return page_idx, pix.width, pix.height, bytes(pix.samples)

def _save_page(args: tuple) -> None:
    _, w, h, raw, out_path, save_opts = args
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    Image.fromarray(arr).save(str(out_path), **save_opts)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERTE UM PDF
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_img(
    pdf_path: Path,
    output_dir: Path,
    dpi: int             = DPI_DEFAULT,
    fmt: str             = FORMAT_DEFAULT,
    quality: int         = QUALITY_DEFAULT,
    workers: int | None  = None,
    save_threads: int | None = None,
    skip_existing: bool  = True,
    prefix: str          = "",
) -> bool:
    """Converte um PDF para imagens. Retorna True se converteu, False se pulou/erro."""
    workers      = workers      or mp.cpu_count()
    save_threads = save_threads or min(workers * 2, 32)
    ext          = _ext(fmt)
    opts         = dict(SAVE_OPTS[fmt])
    if fmt in ("jpg", "webp"):
        opts["quality"] = quality

    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and any(output_dir.glob(f"*.{ext}")):
        return False  # já convertido

    try:
        with fitz.open(str(pdf_path)) as doc:
            total = doc.page_count
    except Exception as e:
        print(f"{prefix}[ERRO ao abrir] {e}")
        return False

    if total == 0:
        print(f"{prefix}[AVISO] 0 páginas, pulando.")
        return False

    zoom = dpi / 72.0
    t0   = time.perf_counter()
    done = 0

    try:
        with ThreadPoolExecutor(max_workers=save_threads) as save_pool:
            futs = []
            with mp.Pool(processes=workers,
                         initializer=_init_worker,
                         initargs=(str(pdf_path), zoom)) as pool:
                for page_idx, w, h, raw in pool.imap_unordered(
                    _render_page, range(total), chunksize=1
                ):
                    out = output_dir / f"page_{page_idx + 1:04d}.{ext}"
                    futs.append(save_pool.submit(_save_page,
                                                 (page_idx, w, h, raw, out, opts)))

            for fut in as_completed(futs):
                fut.result()
                done += 1
                print(f"\r{prefix}{done}/{total} págs", end="", flush=True)

    except Exception as e:
        print(f"\n{prefix}[ERRO] {e}")
        return False

    elapsed = time.perf_counter() - t0
    print(f"\r{prefix}{done}/{total} págs  {elapsed:.1f}s  ({total/elapsed:.0f} pág/s)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSA PASTA (sequencial, um PDF por vez com todos os cores)
# ══════════════════════════════════════════════════════════════════════════════

def process_directory(
    root: Path,
    dpi: int             = DPI_DEFAULT,
    fmt: str             = FORMAT_DEFAULT,
    quality: int         = QUALITY_DEFAULT,
    workers: int | None  = None,
    save_threads: int | None = None,
) -> None:
    workers      = workers      or mp.cpu_count()
    save_threads = save_threads or min(workers * 2, 32)
    ext          = _ext(fmt)

    out_root = root.parent / f"{root.name} ({fmt})"
    all_pdfs = sorted(root.rglob("*.pdf"))

    if not all_pdfs:
        print("Nenhum PDF encontrado.")
        return

    skip_count = sum(
        1 for p in all_pdfs
        if any((out_root / p.relative_to(root).parent / p.stem).glob(f"*.{ext}"))
    )

    print(f"\n[batch] {root.name}/")
    print(f"  Formato        : {fmt.upper()}  quality={quality if fmt != 'png' else 'N/A'}")
    print(f"  DPI            : {dpi}")
    print(f"  PDFs total     : {len(all_pdfs)}")
    print(f"  Já convertidos : {skip_count}  (serão pulados)")
    print(f"  A processar    : {len(all_pdfs) - skip_count}")
    print(f"  Workers CPU    : {workers}  (todos por PDF)")
    print(f"  Output         : {out_root}/\n")

    t_start   = time.perf_counter()
    converted = 0
    skipped   = 0
    errors    = 0

    for i, pdf_path in enumerate(all_pdfs, 1):
        rel        = pdf_path.relative_to(root)
        output_dir = out_root / rel.parent / pdf_path.stem
        prefix     = f"  [{i}/{len(all_pdfs)}] "

        # Verifica skip antes de imprimir
        if any(output_dir.glob(f"*.{ext}")):
            skipped += 1
            continue

        print(f"[{i}/{len(all_pdfs)}] {rel}")
        ok = pdf_to_img(
            pdf_path, output_dir,
            dpi=dpi, fmt=fmt, quality=quality,
            workers=workers, save_threads=save_threads,
            skip_existing=False, prefix=prefix,
        )
        if ok:
            converted += 1
        else:
            errors += 1

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─'*50}")
    print(f"  Concluído em {elapsed/60:.1f} min")
    print(f"  Convertidos  : {converted}")
    print(f"  Pulados      : {skipped}")
    print(f"  Erros        : {errors}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF → imagem  |  arquivo único ou pasta recursiva"
    )
    parser.add_argument("input", help="PDF ou pasta")
    parser.add_argument("--dpi",     type=int, default=DPI_DEFAULT)
    parser.add_argument("--format",  type=str, default=FORMAT_DEFAULT,
                        choices=["jpg", "png", "webp"],
                        help="Formato de saída (padrão: jpg)")
    parser.add_argument("--quality", "-q", type=int, default=QUALITY_DEFAULT,
                        help="Qualidade JPEG/WebP 1-95 (padrão: 85)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Workers render CPU (padrão: cpu_count())")
    parser.add_argument("--save-threads", "-s", type=int, default=None)
    args = parser.parse_args()

    target = Path(args.input).resolve()
    if not target.exists():
        print(f"Erro: não encontrado: {target}")
        raise SystemExit(1)

    if target.is_dir():
        process_directory(target, dpi=args.dpi, fmt=args.format,
                          quality=args.quality, workers=args.workers,
                          save_threads=args.save_threads)
    else:
        output_dir = target.parent / target.stem
        print(f"\n[pdf_to_img] {target.name}")
        pdf_to_img(target, output_dir, dpi=args.dpi, fmt=args.format,
                   quality=args.quality, workers=args.workers,
                   save_threads=args.save_threads, skip_existing=False, prefix="  ")
        print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
