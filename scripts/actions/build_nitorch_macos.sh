if [ "${TARGET_VERSION}" == "same" ]; then
  MACOS_MAJOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f1)
  MACOS_MINOR=$(defaults read loginwindow SystemVersionStampAsString | cut -d'.' -f2)
  TARGET_VERSION="${MACOS_MAJOR}.${MACOS_MINOR}"
fi

# 1) prepare compiler
# activate xcode 9.4.1 so that we can compile against both libstdc++ and libc++
sudo xcode-select -s /Applications/Xcode_9.4.1.app/Contents/Developer/

# 2) install scipy ourselves because setuptools does a poor job
pip install scipy


# 3) build nitorch
python setup.py install
