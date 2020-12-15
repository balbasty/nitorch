[ -z "${TARGET_VERSION}" ] && TARGET_VERSION="same"
if [ "${TARGET_VERSION}" == "same" ]; then
  MACOS_MAJOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f1)
  MACOS_MINOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f2)
  TARGET_VERSION="${MACOS_MAJOR}.${MACOS_MINOR}"
fi
MACOS_MAJOR=$(cut -d'.' -f1 <<< "${TARGET_VERSION}")
MACOS_MINOR=$(cut -d'.' -f2 <<< "${TARGET_VERSION}")
[ "${MACOS_MINOR}" -lt "10" ] && MINORSHORT="0${MACOS_MINOR}" || MINORSHORT="${MACOS_MINOR}"
MACOS_SHORT="${MACOS_MAJOR}${MINORSHORT}"

## 1) prepare cSION: $(xcodebuild -verompiler
##echo "TARGET MACOS VERSION: ${TARGET_VERSION}"
##if [ "${MACOS_SHORT}" -le "1013" ]; then
##  # activate xcode 9.4.1 so that we can compile against libstdc++
##  # exists on 10.13 and 10.14
##  sudo xcode-select -s /Applications/Xcode_9.4.1.app/Contents/Developer/
##else
##  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer/
##fi
##echo "XCODE VERsion)"
export CC='gcc-8'
export CXX='g++-8'


# 2) install scipy ourselves because setuptools does a poor job
pip install scipy

# 3) build nitorch
[ -z "${BUILD_ACTION}" ] && BUILD_ACTION="install"
python setup.py "${BUILD_ACTION}"
