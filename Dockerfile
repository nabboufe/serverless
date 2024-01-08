FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install safetensors==0.3.1 sentencepiece huggingface_hub \
        git+https://github.com/winglian/runpod-python.git@fix-generator-check ninja==1.11.1
RUN pip install exllamav2


COPY handler.py /data/handler.py
COPY __init.py__ /data/__init__.py

ENV PYTHONPATH=/data/exllama
ENV MODEL_REPO=""
ENV TOKENIZER_REPO=""
ENV HF_TOKEN=""
ENV MAX_SEQ_LEN=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/hub"

CMD [ "python", "-m", "handler" ]
