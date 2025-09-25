FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends     sox     ffmpeg     libsndfile1     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/voice-duplex-zh

COPY ../environment.yml /tmp/environment.yml

RUN conda env update -f /tmp/environment.yml &&     echo "conda activate voice-duplex-zh" >> /root/.bashrc

COPY . .

SHELL ["/bin/bash", "-lc"]

ENTRYPOINT ["/workspace/voice-duplex-zh/env/docker/entrypoint.sh"]
