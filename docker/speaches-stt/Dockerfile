# syntax=docker/dockerfile:1.7
# speech-stt: Speaches as pinned in k3s-gpu-cluster, plus three additions:
#
# 1. cuBLAS 12.9 for the RTX 50-series card. The base ships cuBLAS 12.6.4 (CUDA 12.6.3), which
#    predates Blackwell; on the RTX 5070 Ti whisper large-v3 decodes 4-7x slower with it
#    (measured 2026-10-21, gpu-speech).
# 2. CTranslate2 4.8.3 instead of 4.5.0, for int8. On 4.5.0 with this cuBLAS, int8 fails in
#    word alignment with CUBLAS_STATUS_NOT_SUPPORTED; 4.8.3 runs it. Measured on 134 phone cuts
#    (2026-10-21): large-v3 int8 takes 2,575 MiB against 4,217 at float16, same speed, 1.4% of
#    words differ from float16. The compute type itself is the deployment's WHISPER__COMPUTE_TYPE.
# 3. gs_speech (this directory): admission control, a model allowlist, a language-ID-only
#    listener on a second port and an opt-in LLM punctuation pass (punct.py), started by `python -m gs_speech.serve`. It wraps Speaches'
#    classes at runtime and edits no Speaches file, so patch 433's hash guard on whisper.py holds.
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

# The CTranslate2 wheel, verified by hash; bind-mounted into the install below, so it never ships.
FROM ${BASE} AS ct2
USER root
ADD --checksum=sha256:7e161eb031fcf2a5d81ce3a1cd8be4954c7df758d96cfaba57aeecc69a0c00ae \
    https://files.pythonhosted.org/packages/b4/39/438c9236c57443099763789ee009d6d43a65fb58283163fb0d6e6dadacd7/ctranslate2-4.8.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
    /wheels/ctranslate2-4.8.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN chmod 0755 /wheels && chmod 0644 /wheels/*.whl

FROM ${BASE}
# Root-owned and read-only to the runtime user (uid 1000).
COPY --from=cublas --chown=0:0 /opt/cublas-12.9 /opt/cublas-12.9
# First on the loader path, ahead of the image's CUDA 12.6 copy (found through ld.so.cache); the
# base image's own entries are kept after it.
ENV LD_LIBRARY_PATH=/opt/cublas-12.9:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Into the image's own venv, as its owner (uid 1000), like the rest of the base's packages. No
# dependencies are resolved: numpy and pyyaml, all it needs, are already there.
RUN --mount=type=bind,from=ct2,source=/wheels,target=/wheels \
    uv pip install --python /home/ubuntu/speaches/.venv/bin/python --no-deps --no-index --no-cache \
      /wheels/ctranslate2-4.8.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
 && /home/ubuntu/speaches/.venv/bin/python -I -c "import ctranslate2; assert ctranslate2.__version__ == '4.8.3', ctranslate2.__version__"

# Root-owned and read-only to the runtime user, like the cuBLAS libraries.
COPY --chown=0:0 --chmod=0644 gs_speech/ /opt/gs-speech/gs_speech/
COPY --chown=0:0 --chmod=0644 tests/ /opt/gs-speech/tests/
USER root
RUN chmod 0755 /opt/gs-speech /opt/gs-speech/gs_speech /opt/gs-speech/tests
USER ubuntu
ENV PYTHONPATH=/opt/gs-speech
# The k3s entrypoint execs its own command; this is for `docker run`.
CMD ["python", "-m", "gs_speech.serve"]
