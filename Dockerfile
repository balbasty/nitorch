# Set the pytorch version. We set this once here so that both stages use the same
# pytorch version (with the same cuda and cudnn versions). This includes the image
# tag without '-devel' or '-runtime'. Stage 1 uses devel and stage 2 uses runtime.
ARG PYTORCH_TAG_PREFIX="1.12.1-cuda11.3-cudnn8"

FROM pytorch/pytorch:$PYTORCH_TAG_PREFIX-devel as builder
WORKDIR /opt/nitorch
COPY . .
RUN NI_COMPILED_BACKEND="C" python setup.py bdist_wheel

# Stage 2: Install pre-compiled package.
FROM pytorch/pytorch:$PYTORCH_TAG_PREFIX-runtime
COPY --from=builder /opt/nitorch/dist/ /opt/nitorch-wheel/
RUN python -m pip install --no-cache-dir /opt/nitorch-wheel/*
