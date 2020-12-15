# parse pytorch version
TORCH_VERSION_MAJOR_MINOR=${torch}
TORCH_MAJOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f1)
TORCH_MINOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f2)
TORCH_PATCH=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f3)
TORCH_SHORT="${TORCH_MAJOR}${TORCH_MINOR}"
TORCH_REPO="https://download.pytorch.org/whl/torch_stable.html"

# torch 1.4 => pre-install numpy
# (it seems to only be an optional dependency in 1.4, but it breaks
#  later on if numpy is not present)
[ "$TORCH_SHORT" == "14" ] && pip install numpy

# install pytorch
pip install "torch==${TORCH_VERSION_MAJOR_MINOR}"
