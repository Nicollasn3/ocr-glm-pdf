#!/bin/bash
# =============================================================================
#  Setup RunPod — PDF → GLM-OCR → Markdown
#  Testado em: RunPod PyTorch 2.x  |  CUDA 11.8 / 12.1
#
#  Uso:
#    bash setup_runpod.sh
#    bash setup_runpod.sh --skip-model    # pula download do modelo
# =============================================================================

set -euo pipefail

SKIP_MODEL=false
for arg in "$@"; do
  [[ "$arg" == "--skip-model" ]] && SKIP_MODEL=true
done

# ── Cores para output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }

# ── 1. Sistema ────────────────────────────────────────────────────────────────
info "Atualizando pacotes do sistema..."
apt-get update -qq
apt-get install -y -qq \
  git \
  wget \
  curl \
  libgl1 \
  libglib2.0-0 \
  libgomp1

# ── 2. Python / pip ───────────────────────────────────────────────────────────
info "Atualizando pip..."
python -m pip install --upgrade pip --quiet

# ── 3. PyTorch (mantém a versão CUDA já presente no RunPod) ──────────────────
info "Verificando PyTorch + CUDA..."
python - <<'EOF'
import sys
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  PyTorch {torch.__version__}  |  CUDA disponível: {cuda}")
    if cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  PyTorch não encontrado — instalando...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
EOF

# ── 4. Pacotes principais ─────────────────────────────────────────────────────
info "Instalando dependências Python..."
pip install --quiet \
  pymupdf \
  pillow \
  numpy \
  accelerate \
  sentencepiece \
  protobuf \
  einops \
  timm

# ── 5. Transformers (versão dev — exigida pelo GLM-OCR) ──────────────────────
info "Instalando transformers (branch main)..."
pip install --quiet \
  "git+https://github.com/huggingface/transformers.git"

# ── 6. (Opcional) SDK oficial GLM-OCR ────────────────────────────────────────
# info "Instalando SDK oficial GLM-OCR..."
# pip install --quiet "git+https://github.com/zai-org/GLM-OCR"

# ── 7. Pre-download do modelo ─────────────────────────────────────────────────
if [ "$SKIP_MODEL" = false ]; then
  info "Baixando modelo zai-org/GLM-OCR (pode demorar ~10–20 min)..."
  python - <<'EOF'
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL = "zai-org/GLM-OCR"
print("  Baixando processor...")
AutoProcessor.from_pretrained(MODEL)
print("  Baixando modelo...")
AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
print("  Modelo em cache.")
EOF
else
  warn "Download do modelo pulado (--skip-model). Será baixado no primeiro uso."
fi

# ── 8. Verificação final ──────────────────────────────────────────────────────
info "Verificação final..."
python - <<'EOF'
import sys

checks = {
    "fitz (PyMuPDF)": "import fitz",
    "PIL (Pillow)":   "from PIL import Image",
    "numpy":          "import numpy",
    "torch":          "import torch",
    "transformers":   "from transformers import AutoProcessor, AutoModelForImageTextToText",
    "accelerate":     "import accelerate",
}

all_ok = True
for name, stmt in checks.items():
    try:
        exec(stmt)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name}  →  {e}", file=sys.stderr)
        all_ok = False

if not all_ok:
    sys.exit(1)
EOF

echo ""
info "Setup concluído! Para rodar:"
echo "  python pdf_to_img.py                   # processa data/teste/"
echo "  python pdf_to_img.py arquivo.pdf"
echo "  python pdf_to_img.py pasta/ --out data_md"
