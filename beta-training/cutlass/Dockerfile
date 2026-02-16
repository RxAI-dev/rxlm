# Bazowy obraz z CUDA 12.8 i cuDNN dla H100
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Ustawienie zmiennych środowiskowych
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalacja Python 3.12 i podstawowych narzędzi
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* get-pip.py

# Utworzenie folderu workspace z odpowiednimi uprawnieniami
RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

# Upgrade pip i instalacja podstawowych narzędzi
RUN pip install --upgrade pip setuptools wheel ninja

# Instalacja PyTorch 2.8.0 dla CUDA 12.8
RUN pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Instalacja RxLM 0.3.60 wraz z zależnościami
RUN pip install rxlm==0.3.60 transformers tokenizers huggingface_hub datasets tensorboard nltk

# Instalacja grouped_gemm z CUTLASS - wymuszamy architekturę H100 (9.0)
RUN GROUPED_GEMM_CUTLASS=1 \
    TORCH_CUDA_ARCH_LIST="9.0" \
    FORCE_CUDA=1 \
    pip install grouped_gemm --no-build-isolation

# Instalacja Flash Attention 2.8.1 ze skompilowanych źródeł
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    && pip install --no-dependencies --upgrade flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    && rm flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

RUN pip install git+https://github.com/KellerJordan/Muon

# Stwórz skrypt startowy, który najpierw instaluje Transformer Engine, potem uruchamia trening
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== Reactive AI Training Setup ==="\n\
echo "Checking Transformer Engine installation..."\n\
\n\
# Sprawdź czy Transformer Engine jest już zainstalowany\n\
if python -c "import transformer_engine" 2>/dev/null; then\n\
    echo "Transformer Engine already installed, skipping..."\n\
else\n\
    echo "Installing Transformer Engine 2.8.0"\n\
    echo "This may take 10-20 minutes..."\n\
    pip3 install --no-build-isolation transformer-engine[pytorch]==2.8.0\n\
    echo "Transformer Engine installed successfully!"\n\
fi\n\
\n\
echo ""\n\
echo "=== Starting RxT-Beta Training ==="\n\
echo ""\n\
\n\
# Uruchom trening\n\
exec torchrun --standalone --nnodes=1 --nproc-per-node=4 /workspace/training/beta.py "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Użyj skryptu startowego jako entrypoint
ENTRYPOINT ["/entrypoint.sh"]