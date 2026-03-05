#!/usr/bin/env python3
"""
PDF → PNG → GLM-OCR → Markdown  (GPU-optimized)

Pipeline:
  data/teste/*.pdf  →  data_md/<nome>.md

Uso:
  python pdf_to_img.py                        # processa data/teste/
  python pdf_to_img.py arquivo.pdf
  python pdf_to_img.py pasta/
  python pdf_to_img.py --out outra_pasta
  python pdf_to_img.py --batch 8 --tokens 2048
"""

import argparse
import os
import sys
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
    from PIL import Image
except ImportError:
    _pip("pillow"); from PIL import Image

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

MODEL_PATH      = "zai-org/GLM-OCR"
DPI_DEFAULT     = 150
BATCH_DEFAULT   = 4      # páginas por batch (ajuste conforme VRAM disponível)
TOKENS_DEFAULT  = 2048   # max_new_tokens; 8192 é excessivo para OCR


# ── Carregamento do modelo (feito uma única vez) ───────────────────────────────

_processor = None
_model     = None


def _load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model

    print("[GLM-OCR] Carregando modelo...", flush=True)
    t0 = time.perf_counter()

    _processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Tenta Flash Attention 2 (requer flash-attn instalado); cai para sdpa
    for attn in ("flash_attention_2", "sdpa", "eager"):
        try:
            _model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,   # bfloat16 explícito — mais rápido e estável na RTX
                device_map="cuda",
                attn_implementation=attn,
            )
            print(f"[GLM-OCR] attn_implementation={attn}", flush=True)
            break
        except Exception:
            continue

    _model.eval()  # desativa dropout, batch norm em modo treino, etc.

    device = next(_model.parameters()).device
    print(f"[GLM-OCR] Modelo pronto em {time.perf_counter() - t0:.1f}s | device={device}", flush=True)
    return _processor, _model


# ── Conversão Pixmap → PIL sem I/O de disco ───────────────────────────────────

def _pix_to_pil(pix: fitz.Pixmap) -> Image.Image:
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ── OCR em batch ──────────────────────────────────────────────────────────────

def _ocr_batch(pil_images: list, max_new_tokens: int) -> list[str]:
    """
    Processa um batch de PIL Images de uma só vez na GPU.
    Retorna lista de strings com o texto extraído, na mesma ordem.
    """
    processor, model = _load_model()

    # 1. Formata o template de chat para cada imagem (sem tokenizar ainda)
    texts = []
    for img in pil_images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "Text Recognition:"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

    # 2. Tokeniza + processa imagens em batch (padding para igualar comprimentos)
    inputs = processor(
        text=texts,
        images=pil_images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    # 3. Inferência — torch.inference_mode evita criação de grafo de gradiente
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy — mais rápido e determinístico para OCR
        )

    # 4. Decodifica apenas os tokens novos (descarta o prompt)
    prompt_len = inputs["input_ids"].shape[1]
    return [
        processor.decode(gen[prompt_len:], skip_special_tokens=True).strip()
        for gen in generated_ids
    ]


# ── Pipeline: PDF → MD ────────────────────────────────────────────────────────

def pdf_to_md(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = DPI_DEFAULT,
    batch_size: int = BATCH_DEFAULT,
    max_new_tokens: int = TOKENS_DEFAULT,
) -> Path | None:
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

    # 1. Renderiza todas as páginas para PIL Images em memória (sem disco)
    pil_pages: list[Image.Image] = []
    for i in range(total):
        page = doc.load_page(i)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            alpha=False,
            colorspace=fitz.csRGB,
        )
        pil_pages.append(_pix_to_pil(pix))
    doc.close()

    # 2. OCR em batches
    parts: list[str] = []
    for batch_start in range(0, total, batch_size):
        batch = pil_pages[batch_start : batch_start + batch_size]
        t0 = time.perf_counter()
        results = _ocr_batch(batch, max_new_tokens)
        elapsed = time.perf_counter() - t0

        for j, text in enumerate(results):
            page_num = batch_start + j + 1
            print(
                f"  pág {page_num}/{total}  "
                f"({elapsed / len(batch):.1f}s/pág  batch={len(batch)})",
                flush=True,
            )
            parts.append(f"<!-- página {page_num} -->\n\n{text}")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    print(f"  → {md_path}")
    return md_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → GLM-OCR → Markdown (GPU-optimized)")
    parser.add_argument(
        "input", nargs="?", default="data/teste",
        help="PDF ou pasta com PDFs (padrão: data/teste)",
    )
    parser.add_argument(
        "--out", default="data_md",
        help="Pasta de saída dos .md (padrão: data_md)",
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI_DEFAULT,
        help=f"DPI da renderização do PDF (padrão: {DPI_DEFAULT})",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_DEFAULT,
        help=f"Páginas por batch de GPU (padrão: {BATCH_DEFAULT}; aumente se tiver VRAM sobrando)",
    )
    parser.add_argument(
        "--tokens", type=int, default=TOKENS_DEFAULT,
        help=f"max_new_tokens por página (padrão: {TOKENS_DEFAULT})",
    )
    args = parser.parse_args()

    target  = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()

    if not target.exists():
        print(f"Erro: não encontrado: {target}")
        raise SystemExit(1)

    pdfs = sorted(target.rglob("*.pdf")) if target.is_dir() else [target]

    if not pdfs:
        print("Nenhum PDF encontrado.")
        return

    print(f"\n[pdf_to_img]  {len(pdfs)} PDF(s)  →  {out_dir}/")
    print(f"              batch={args.batch}  dpi={args.dpi}  tokens={args.tokens}\n")

    t_start = time.perf_counter()
    ok = skipped = errors = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        result = pdf_to_md(pdf, out_dir, dpi=args.dpi, batch_size=args.batch, max_new_tokens=args.tokens)
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
