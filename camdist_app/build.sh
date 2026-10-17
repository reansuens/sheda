#!/usr/bin/env bash
# Build (and optionally install) the CamDist Cam APK:  ./build.sh install
set -e
export JAVA_HOME=/opt/android-studio/jbr
export ANDROID_HOME="$HOME/Android/Sdk"
cd "$(dirname "$0")"
./gradlew assembleDebug
APK=app/build/outputs/apk/debug/app-debug.apk
echo "APK: $APK"
[ "$1" = "install" ] && adb install -r "$APK" && adb shell am start -n com.camdist.app/.MainActivity
