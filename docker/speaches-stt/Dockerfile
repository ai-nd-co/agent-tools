# syntax=docker/dockerfile:1.7
# speech-stt: Speaches as pinned in k3s-gpu-cluster, plus cuBLAS 12.9 for the RTX 50-series card.
# The base ships cuBLAS 12.6.4 (CUDA 12.6.3), which predates Blackwell; on the RTX 5070 Ti
# whisper large-v3 decodes 4-7x slower with it (measured 2026-10-21, gpu-speech). Nothing else
# changes: same Python stack, cuDNN, model and executor source (patch 433's hash guard still holds).
ARG BASE=ghcr.io/speaches-ai/speaches@sha256:5c6206a349e90b9a6782342917e72f84fc7cb60e8afd540f6aa625831ac1fd0f

# Unpack in a throwaway stage, so the 581 MB wheel itself never ships: only its two libraries do,
# in one layer on top of the unchanged base (which the node already has cached).
FROM ${BASE} AS cublas
USER root
# The exact wheel that was benchmarked, verified by hash at build time.
ADD --checksum=sha256:e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702 \
    https://files.pythonhosted.org/packages/py3/n/nvidia-cublas-cu12/nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl \
    /tmp/cublas.whl
RUN /home/ubuntu/speaches/.venv/bin/python -I -c "import zipfile,os; z=zipfile.ZipFile('/tmp/cublas.whl'); d='/opt/cublas-12.9'; os.makedirs(d); [open(os.path.join(d,os.path.basename(n)),'wb').write(z.read(n)) for n in z.namelist() if n.startswith('nvidia/cublas/lib/libcublas')]" \
 && chmod 0755 /opt/cublas-12.9 && chmod 0644 /opt/cublas-12.9/* \
 && test -s /opt/cublas-12.9/libcublas.so.12 && test -s /opt/cublas-12.9/libcublasLt.so.12

FROM ${BASE}
# Root-owned and read-only to the runtime user (uid 1000).
COPY --from=cublas --chown=0:0 /opt/cublas-12.9 /opt/cublas-12.9
# First on the loader path, ahead of the image's CUDA 12.6 copy (found through ld.so.cache); the
# base image's own entries are kept after it.
ENV LD_LIBRARY_PATH=/opt/cublas-12.9:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
