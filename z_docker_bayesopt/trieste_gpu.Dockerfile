# This NVIDIA container uses TF 2.15 but is patched for Blackwell
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3
USER root

# Install uv for installation of other packages
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /bin/uv

# Install Trieste and other pins
# Note: tensorflow is pre-installed in the image.
RUN uv pip install --system \
    gpflow \
    setuptools==80.10.2 \
    ipython==8.38.0 \
    trieste==4.5.1 \
    trieste[plotting] \
    marimo \
    plotly \
    numba

# workspace is the working directory in the image
WORKDIR /workspace

EXPOSE 8080
