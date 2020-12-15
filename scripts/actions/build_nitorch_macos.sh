sudo xcode-select -s /Applications/Xcode.app/Contents/Developer/

if [ -n "${TARGET_VERSION}" ]; then
  if [ "${TARGET_VERSION}" == "same" ]; then
    MACOS_MAJOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f1)
    MACOS_MINOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f2)
    TARGET_VERSION="${MACOS_MAJOR}.${MACOS_MINOR}"
  fi
  echo "TARGET MACOS VERSION: ${TARGET_VERSION}"

  MACOS_MAJOR=$(cut -d'.' -f1 <<< "${TARGET_VERSION}")
  MACOS_MINOR=$(cut -d'.' -f2 <<< "${TARGET_VERSION}")
  [ "${MACOS_MINOR}" -lt "10" ] && MINORSHORT="0${MACOS_MINOR}" || MINORSHORT="${MACOS_MINOR}"
  MACOS_SHORT="${MACOS_MAJOR}${MINORSHORT}"

  # 1) prepare compiler
  if [ "${MACOS_SHORT}" -le "1013" ]; then
    # activate xcode 9.4.1 so that we can compile against libstdc++
    # exists on 10.13 and 10.14
    sudo xcode-select -s /Applications/Xcode_9.4.1.app/Contents/Developer/
  fi
fi
XCODE_VERSION="$(xcodebuild -version)"
echo "XCODE VERSION: ${XCODE_VERSION}"

# 2) download and install openmp
XCODE_MAJOR=$(cut -d'.' -f1 <<< "${XCODE_VERSION}")
XCODE_MINOR=$(cut -d'.' -f2 <<< "${XCODE_VERSION}")
[ "${XCODE_MINOR}" -lt "10" ] && MINORSHORT="0${XCODE_MINOR}" || MINORSHORT="${XCODE_MINOR}"
XCODE_SHORT="${XCODE_MAJOR}${MINORSHORT}"
if [ "${XCODE_SHORT}" -le "1100" ]; then
  OMP_URL="openmp-7.1.0-darwin17-Release.tar.gz"
elif [ "${XCODE_SHORT}" -le "1104" ]; then
  OMP_URL="openmp-8.0.1-darwin17-Release.tar.gz"
elif [ "${XCODE_SHORT}" -le "1200" ]; then
  OMP_URL="openmp-9.0.1-darwin17-Release.tar.gz"
else
  OMP_URL="openmp-10.0.0-darwin17-Release.tar.gz"
fi
curl -O "https://mac.r-project.org/openmp/${OMP_URL}"
mkdir omp
tar fvx "$OMP_URL" -C omp
sudo cp -r omp/* /
rm -rf omp

# 3) install scipy ourselves because setuptools does a poor job
pip install scipy

# 4) build nitorch
[ -z "${BUILD_ACTION}" ] && BUILD_ACTION="install"
[ -n "${TARGET_VERSION}" ] && export MACOSX_DEPLOYMENT_TARGET="${TARGET_VERSION}"
python setup.py "${BUILD_ACTION}"
