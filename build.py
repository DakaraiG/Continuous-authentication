# build.py
# Packages the Continuous Auth app into a standalone executable
# Run: python build.py
#
# On Windows: produces dist/ContinuousAuth/ContinuousAuth.exe
# On macOS:   produces dist/ContinuousAuth.app

import subprocess
import sys
import platform

def main():
    system = platform.system()
    print(f"Building for {system}...")

    # base pyinstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
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
    ]

    if system == "Darwin":
        # macOS needs an Info.plist for permissions
        cmd += ["--osx-bundle-identifier", "com.continuousauth.app"]

    # add the main script
    cmd.append("main.py")

    print("Running PyInstaller...")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)

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