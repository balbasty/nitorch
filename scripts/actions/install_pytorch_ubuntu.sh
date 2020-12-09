
# parse cuda version
CUDA_VERSION_MAJOR_MINOR=${cuda}
if [ ! -z "$CUDA_VERSION_MAJOR_MINOR" ]; then
  CUDA_SHORT="cpu"
else
  CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
  CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
  CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
  CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"
fi

# parse pytorch version
TORCH_VERSION_MAJOR_MINOR=${torch}
TORCH_MAJOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f1)
TORCH_MINOR=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f2)
TORCH_PATCH=$(echo "${TORCH_VERSION_MAJOR_MINOR}" | cut -d. -f3)
TORCH_SHORT="${TORCH_MAJOR}${TORCH_MINOR}"
TORCH_REPO="https://download.pytorch.org/whl/torch_stable.html"

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

# CUDA 10.2 is the default version so the corresponding wheels are not
# prepended with the version
if [ "$CUDA_SHORT" == "102" ]; then
  CUDA_SHORT=""
else
  CUDA_SHORT="+${CUDA_SHORT}"
fi

pip install "torch==${TORCH_VERSION_MAJOR_MINOR}${CUDA_SHORT}" -f "${TORCH_REPO}"

