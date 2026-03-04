# build.py
# Packages the Continuous Auth app into a standalone executable
# Run: python build.py
#
# On Windows: produces dist/ContinuousAuth/ContinuousAuth.exe
# On macOS:   produces dist/ContinuousAuth.app

import subprocess
import sys
import platform
import plistlib
import os

# Python 3.11 is used for macOS builds — it has stable PyInstaller support
# and all dependencies (numpy, sklearn, PyQt6, pynput) pre-installed.
# Python 3.14 has known PyInstaller compatibility issues with numpy bundling.
PYTHON_MACOS = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"

def main():
    system = platform.system()
    print(f"Building for {system}...")

    python = PYTHON_MACOS if system == "Darwin" else sys.executable

    # base pyinstaller command
    cmd = [
        python, "-m", "PyInstaller",
        "--name", "ContinuousAuth",
        "--windowed",           # no console window (GUI app)
        "--noconfirm",          # overwrite previous build without asking
        "--clean",              # clean cache before building
        # hidden imports that pyinstaller sometimes misses
        "--hidden-import", "window",
        "--hidden-import", "policy",
        "--hidden-import", "db",
        "--hidden-import", "capture",
        "--hidden-import", "features",
        "--hidden-import", "train_model",
        "--hidden-import", "app_log",
        "--hidden-import", "logger",
        "--hidden-import", "sklearn.ensemble._forest",
        "--hidden-import", "sklearn.ensemble._iforest",
        "--hidden-import", "sklearn.svm._classes",
        "--hidden-import", "sklearn.utils._typedefs",
        "--hidden-import", "sklearn.neighbors._partition_nodes",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "pynput.keyboard._win32",
        "--hidden-import", "pynput.mouse._win32",
        "--hidden-import", "pynput.keyboard._darwin",
        "--hidden-import", "pynput.mouse._darwin",
        # collect sklearn metadata that pyinstaller needs
        "--collect-submodules", "sklearn",
        # numpy 2.x requires full collection (binaries + data) to avoid
        # "PyCapsule_Import could not import module datetime" crash
        "--collect-all", "numpy",
    ]

    if system == "Darwin":
        # macOS needs an Info.plist for permissions
        cmd += ["--osx-bundle-identifier", "com.continuousauth.app"]
        # Quartz is imported conditionally in capture.py — pyinstaller won't detect it automatically
        cmd += ["--hidden-import", "Quartz", "--hidden-import", "_datetime"]

    # add the main script
    cmd.append("main.py")

    print("Running PyInstaller...")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0 and system == "Darwin":
        # Patch Info.plist with required privacy usage descriptions.
        # Without these, macOS silently kills the app on launch when it tries
        # to monitor keyboard/mouse input via pynput.
        plist_path = "dist/ContinuousAuth.app/Contents/Info.plist"
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        plist["NSAccessibilityUsageDescription"] = (
            "ContinuousAuth needs Accessibility access to monitor mouse movements "
            "and clicks for continuous authentication."
        )
        plist["NSInputMonitoringUsageDescription"] = (
            "ContinuousAuth needs Input Monitoring access to track typing patterns "
            "for continuous authentication."
        )
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)
        print("  Patched Info.plist with privacy usage descriptions.")

        # Re-sign with ad-hoc identity after patching Info.plist,
        # otherwise macOS rejects the bundle due to invalid signature.
        subprocess.run([
            "codesign", "--deep", "--force", "--sign", "-",
            "dist/ContinuousAuth.app"
        ], check=True)
        print("  Re-signed app bundle.")

    if result.returncode == 0:
        print("\nBuild successful!")
        if system == "Windows":
            print("  Output: dist/ContinuousAuth/ContinuousAuth.exe")
        elif system == "Darwin":
            print("  Output: dist/ContinuousAuth.app")
            print("\n  IMPORTANT: On macOS you need to grant permissions:")
            print("    System Settings > Privacy & Security > Accessibility")
            print("    System Settings > Privacy & Security > Input Monitoring")
            print("    Add ContinuousAuth.app to both lists")
        print("\n  Note: the auth_log.db file will be created in the same")
        print("  directory as the executable when the app first runs.")
    else:
        print(f"\n✗ Build failed with code {result.returncode}")
        print("  Check the output above for errors.")
        print("  Common fix: pip install pyinstaller --break-system-packages")

if __name__ == "__main__":
    main()