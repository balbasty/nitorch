
# parse pytorch version
TORCH_VERSION_MAJOR_MINOR=${torch}
TORCH_MAJOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f1)
TORCH_MINOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f2)
TORCH_PATCH=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f3)
TORCH_SHORT="${TORCH_MAJOR}${TORCH_MINOR}"
TORCH_REPO="https://download.pytorch.org/whl/torch_stable.html"

# parse cuda version
CUDA_VERSION_MAJOR_MINOR=${cuda}
if [ -z "$CUDA_VERSION_MAJOR_MINOR" ]; then
  CUDA_SHORT="cpu"
elif [ "$CUDA_VERSION_MAJOR_MINOR" == "cpu" ]; then
  CUDA_SHORT="cpu"
else
  CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
  CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
  CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
  CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"
fi

# check compatibility
if [ "$TORCH_SHORT" == "17" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "110" ] || \
  [ "$CUDA_SHORT" == "102" ] || \
  [ "$CUDA_SHORT" == "101" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "16" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "102" ] || \
  [ "$CUDA_SHORT" == "101" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "15" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "102" ] || \
  [ "$CUDA_SHORT" == "101" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "14" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "101" ] || \
  [ "$CUDA_SHORT" == "100" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "13" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "101" ] || \
  [ "$CUDA_SHORT" == "100" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "12" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "100" ] || \
  [ "$CUDA_SHORT" == "92" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "11" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "100" ] || \
  [ "$CUDA_SHORT" == "90" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
elif [ "$TORCH_SHORT" == "10" ]; then
  [ "$CUDA_SHORT" == "cpu" ] || \
  [ "$CUDA_SHORT" == "100" ] || \
  [ "$CUDA_SHORT" == "90" ] || \
  [ "$CUDA_SHORT" == "80" ] || \
  { echo "Incompatible versions: pytorch ${TORCH_MAJOR}.${TORCH_MINOR} " \
  "and cuda ${CUDA_MAJOR}.${CUDA_MINOR}"; exit 1; }
fi

# for torch >= 1.5 CUDA 10.2 is the default version
# for torch <  1.5 CUDA 10.1 is the default version
# for torch <  1.3 CUDA 10.0 is the default version
# the corresponding wheels are not prepended with the version number
[ "$TORCH_SHORT" -lt "13" ] && [ "$CUDA_SHORT" == "100" ] && CUDA_SHORT=""
[ "$TORCH_SHORT" -lt "15" ] && [ "$CUDA_SHORT" == "101" ] && CUDA_SHORT=""
[ "$TORCH_SHORT" -ge "15" ] && [ "$CUDA_SHORT" == "102" ] && CUDA_SHORT=""
[ -n "${CUDA_SHORT}" ] && [ "${CUDA_SHORT}" != "cpu" ] && CUDA_SHORT="cu${CUDA_SHORT}"
[ -n "${CUDA_SHORT}" ] && CUDA_SHORT="+${CUDA_SHORT}"

# torch 1.4 => pre-install numpy
# (it seems to only be an optional dependency in 1.4, but it breaks
#  later on if numpy is not present)
[ "$TORCH_SHORT" == "14" ] && pip install numpy

# install pytorch
pip install "torch==${TORCH_VERSION_MAJOR_MINOR}${CUDA_SHORT}" -f "${TORCH_REPO}"
