FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive

# git: needed for pip-installing any VCS deps; otherwise the NGC PyTorch image
# already ships build-essential + python3-dev. Keep the image thin: the project
# Python env is created at runtime as a venv on $SCRATCH (the enroot squashfs is
# read-only at run time). See CSCS_SETUP.md.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
