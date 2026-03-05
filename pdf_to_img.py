#!/usr/bin/env python3
"""
PDF → PNG → GLM-OCR → Markdown

Pipeline:
  data/teste/*.pdf  →  data_md/<nome>.md

Uso:
  python pdf_to_img.py                    # processa data/teste/
  python pdf_to_img.py arquivo.pdf
  python pdf_to_img.py pasta/
  python pdf_to_img.py --out outra_pasta
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


def _pip(*pkgs: str) -> None:
    os.system(f"{sys.executable} -m pip install -q {' '.join(pkgs)}")


try:
    import fitz
except ImportError:
    _pip("pymupdf"); import fitz

try:
    import torch
except ImportError:
    _pip("torch"); import torch

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    _pip("transformers"); from transformers import AutoProcessor, AutoModelForImageTextToText


# ── Suprime warnings MuPDF ────────────────────────────────────────────────────

def _silence_mupdf() -> None:
    try:
        fitz.TOOLS.mupdf_warnings(False)
    except Exception:
        pass

_silence_mupdf()


# ── Configuração ──────────────────────────────────────────────────────────────

MODEL_PATH = "zai-org/GLM-OCR"
DPI_DEFAULT = 150


# ── Carregamento do modelo (feito uma única vez) ───────────────────────────────

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        print("[GLM-OCR] Carregando modelo (primeira vez, pode demorar)...", flush=True)
        t0 = time.perf_counter()
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        print(f"[GLM-OCR] Modelo pronto em {time.perf_counter() - t0:.1f}s", flush=True)
    return _processor, _model


# ── OCR de uma imagem ─────────────────────────────────────────────────────────

def _ocr_image(img_path: str) -> str:
    """Roda GLM-OCR em uma imagem e retorna o texto extraído."""
    processor, model = _load_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_path},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=8192)

    text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text


# ── Pipeline: PDF → MD ────────────────────────────────────────────────────────

def pdf_to_md(pdf_path: Path, out_dir: Path, dpi: int = DPI_DEFAULT) -> Path | None:
    """
    Converte um PDF completo para um único arquivo .md via GLM-OCR.
    Cada página é separada por '---' no markdown.
    Retorna o caminho do .md gerado, ou None em caso de erro.
    """
    md_path = out_dir / (pdf_path.stem + ".md")

    if md_path.exists():
        print(f"  [skip] {md_path.name} já existe.")
        return md_path

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"  [ERRO ao abrir PDF] {e}")
        return None

    total = doc.page_count
    if total == 0:
        print("  [AVISO] 0 páginas, pulando.")
        doc.close()
        return None

    zoom = dpi / 72.0
    parts: list[str] = []

    # PNGs em pasta temporária (apagados automaticamente após o PDF)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(total):
            page = doc.load_page(i)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            img_path = os.path.join(tmpdir, f"page_{i + 1:04d}.png")
            pix.save(img_path)

            t0 = time.perf_counter()
            text = _ocr_image(img_path)
            elapsed = time.perf_counter() - t0

            print(f"  pág {i + 1}/{total}  {elapsed:.1f}s", flush=True)
            parts.append(f"<!-- página {i + 1} -->\n\n{text.strip()}")

    doc.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    print(f"  → {md_path}")
    return md_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF → PNG → GLM-OCR → Markdown"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/teste",
        help="PDF ou pasta com PDFs (padrão: data/teste)",
    )
    parser.add_argument(
        "--out",
        default="data_md",
        help="Pasta de saída dos .md (padrão: data_md)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI_DEFAULT,
        help=f"DPI da renderização do PDF (padrão: {DPI_DEFAULT})",
    )
    args = parser.parse_args()

    target = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()

    if not target.exists():
        print(f"Erro: não encontrado: {target}")
        raise SystemExit(1)

    pdfs = sorted(target.rglob("*.pdf")) if target.is_dir() else [target]

    if not pdfs:
        print("Nenhum PDF encontrado.")
        return

    print(f"\n[pdf_to_img]  {len(pdfs)} PDF(s)  →  {out_dir}/\n")

    t_start = time.perf_counter()
    ok = 0
    skipped = 0
    errors = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        result = pdf_to_md(pdf, out_dir, dpi=args.dpi)
        if result is None:
            errors += 1
        elif result.exists() and result.stat().st_mtime < time.time() - 1:
            skipped += 1
        else:
            ok += 1

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─' * 50}")
    print(f"  Concluído em {elapsed / 60:.1f} min")
    print(f"  Gerados  : {ok}")
    print(f"  Pulados  : {skipped}")
    print(f"  Erros    : {errors}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
