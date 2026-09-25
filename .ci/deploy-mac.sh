#!/bin/sh -ex
# Package RPCS3 Metal (macOS 26+, Apple silicon, native Metal renderer).
#
# Environment:
#   BUILD_DIR                      build directory containing bin/rpcs3.app (default: build)
#   RPCS3_CODESIGN_IDENTITY        if set, sign with this identity using the hardened runtime and a secure timestamp
#                                  (required for notarization); otherwise the app is signed ad hoc
#   LVER, BUILD_ARTIFACTSTAGINGDIRECTORY, RELEASE_MESSAGE   set by CI (see .ci/build-mac.sh)
#
# No MoltenVK / Vulkan ICD / libvulkan is bundled: Metal is the only renderer of this fork.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTITLEMENTS="$ROOT/rpcs3/rpcs3.entitlements"
APP_NAME="RPCS3 Metal.app"

cd "${BUILD_DIR:-build}" || exit 1

cd bin

rm -rf "rpcs3.app/Contents/Frameworks/QtPdf.framework" \
"rpcs3.app/Contents/Frameworks/QtQml.framework" \
"rpcs3.app/Contents/Frameworks/QtQmlModels.framework" \
"rpcs3.app/Contents/Frameworks/QtQuick.framework" \
"rpcs3.app/Contents/Frameworks/QtVirtualKeyboard.framework" \
"rpcs3.app/Contents/Plugins/platforminputcontexts" \
"rpcs3.app/Contents/Plugins/virtualkeyboard" \
"rpcs3.app/Contents/Resources/git" || true

"$ROOT/.ci/optimize-mac.sh" rpcs3.app

# Download translations
mkdir -p "rpcs3.app/Contents/translations"
ZIP_URL="https://github.com/RPCS3/rpcs3_translations/releases/latest/download/RPCS3-languages.zip"
echo "Downloading translations from: $ZIP_URL"
if curl -fsSL --retry 3 --retry-delay 60 "$ZIP_URL" -o "translations.zip"; then
  echo "Successfully downloaded translations."
  if unzip -o translations.zip -d "rpcs3.app/Contents/translations" >/dev/null 2>&1; then
    rm -f translations.zip
  else
    echo "Failed to extract translations.zip. Continuing without translations."
    rm -f translations.zip
  fi
else
  echo "Warning: Failed to download translations. Skipping..."
fi

# Copy Qt translations manually (qt-downloader Qt used by .ci/build-mac.sh, or Homebrew Qt used by build-macos.sh)
if [ -z "${QT_TRANS:-}" ]; then
  if [ -n "${WORKDIR:-}" ] && [ -d "$WORKDIR/qt-downloader/${QT_VER:-}/clang_64/translations" ]; then
    QT_TRANS="$WORKDIR/qt-downloader/$QT_VER/clang_64/translations"
  elif command -v qtpaths6 >/dev/null 2>&1; then
    QT_TRANS="$(qtpaths6 --query QT_INSTALL_TRANSLATIONS 2>/dev/null || true)"
  elif command -v qtpaths >/dev/null 2>&1; then
    QT_TRANS="$(qtpaths --query QT_INSTALL_TRANSLATIONS 2>/dev/null || true)"
  elif command -v brew >/dev/null 2>&1; then
    QT_TRANS="$(brew --prefix qt 2>/dev/null || true)/share/qt/translations"
  fi
fi
if [ -n "${QT_TRANS:-}" ] && [ -d "$QT_TRANS" ]; then
  cp "$QT_TRANS"/qt_*.qm rpcs3.app/Contents/translations || true
  cp "$QT_TRANS"/qtbase_*.qm rpcs3.app/Contents/translations || true
  cp "$QT_TRANS"/qtmultimedia_*.qm rpcs3.app/Contents/translations || true
  rm -f rpcs3.app/Contents/translations/qt_help_*.qm || true
else
  echo "Warning: Qt translations not found. Skipping..."
fi

# Distinct bundle name so it can live next to an upstream RPCS3.app in /Applications
rm -rf "$APP_NAME"
mv rpcs3.app "$APP_NAME"

# Code signing (JIT, camera and microphone entitlements are required at runtime)
if [ -n "${RPCS3_CODESIGN_IDENTITY:-}" ]; then
  # Developer ID: hardened runtime + secure timestamp, ready for notarization
  # NOTE: "--deep" is deprecated by Apple but still signs all nested frameworks/dylibs here.
  codesign --force --deep --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" --sign "$RPCS3_CODESIGN_IDENTITY" "$APP_NAME"
else
  # Ad-hoc signature (users have to right-click -> Open or remove the quarantine attribute on first launch)
  codesign --force --deep --sign - --entitlements "$ENTITLEMENTS" "$APP_NAME"
fi
codesign --verify --deep --strict --verbose=2 "$APP_NAME"

echo "[InternetShortcut]" > Quickstart.url
echo "URL=https://rpcs3.net/quickstart" >> Quickstart.url
echo "IconIndex=0" >> Quickstart.url

ARCHIVE_DIR="${BUILD_ARTIFACTSTAGINGDIRECTORY:-$PWD}"
mkdir -p "$ARCHIVE_DIR"
if command -v 7z >/dev/null 2>&1; then
  ARCHIVE_FILEPATH="$ARCHIVE_DIR/rpcs3-metal-v${LVER:-local}_macos_arm64.7z"
  7z a -mx9 "$ARCHIVE_FILEPATH" "$APP_NAME" Quickstart.url
else
  # 7-Zip not installed (e.g. local builds): fall back to a zip that preserves the code signature
  ARCHIVE_FILEPATH="$ARCHIVE_DIR/rpcs3-metal-v${LVER:-local}_macos_arm64.zip"
  ditto -c -k --sequesterRsrc --keepParent "$APP_NAME" "$ARCHIVE_FILEPATH"
fi
FILESIZE=$(stat -f %z "$ARCHIVE_FILEPATH")
SHA256SUM=$(shasum -a 256 "$ARCHIVE_FILEPATH" | awk '{ print $1 }')

cd ..
echo "${SHA256SUM};${FILESIZE}B" > "${RELEASE_MESSAGE:-GitHubReleaseMessage.txt}"
cd bin
