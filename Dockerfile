# Set the pytorch version. We set this once here so that all stages use the same
# pytorch version (with the same cuda and cudnn versions). This includes the image
# tag without '-devel' or '-runtime'. The first stage uses devel and others use runtime.
ARG PYTORCH_TAG_PREFIX="1.12.1-cuda11.3-cudnn8"

# In this stage, build a nitorch wheel.
FROM pytorch/pytorch:${PYTORCH_TAG_PREFIX}-devel as builder
WORKDIR /opt/nitorch
COPY . .
RUN NI_COMPILED_BACKEND="C" \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" \
    python setup.py bdist_wheel

# In this stage, install python dependencies that are not in install_requires.
FROM pytorch/pytorch:${PYTORCH_TAG_PREFIX}-runtime as python-deps
RUN conda install --yes --quiet --freeze-installed --channel conda-forge \
        appdirs \
        matplotlib \
        nibabel \
        python-wget \
        scipy \
    # Clean up the caches so they are not part of the final image.
    && conda clean --all --yes \
    # Run first-time setup for matplotlib.
    && python -c 'import matplotlib'

# In this stage, set up the final image.
FROM python-deps
COPY --from=builder /opt/nitorch/dist/ /opt/nitorch-wheel/
RUN python -m pip install --no-cache-dir /opt/nitorch-wheel/*
