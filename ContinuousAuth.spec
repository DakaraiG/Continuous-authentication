# ContinuousAuth.spec
# PyInstaller spec file for the Continuous Auth app
# You can use this instead of build.py if you need more control:
#   pyinstaller ContinuousAuth.spec

import sys
import platform

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'window',
        'policy',
        'db',
        'capture',
        'features',
        'train_model',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._iforest',
        'sklearn.svm._classes',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree._utils',
        'sklearn.calibration',
        'sklearn.preprocessing._data',
        'pynput.keyboard._win32',
        'pynput.mouse._win32',
        'pynput.keyboard._darwin',
        'pynput.mouse._darwin',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',    # not needed at runtime, only for evaluation
        'pandas',        # not needed at runtime
        'tkinter',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ContinuousAuth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # GUI app, no terminal window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ContinuousAuth',
)

# macOS .app bundle
if platform.system() == 'Darwin':
    app = BUNDLE(
        coll,
        name='ContinuousAuth.app',
        bundle_identifier='com.continuousauth.app',
        info_plist={
            'NSAppleEventsUsageDescription': 'ContinuousAuth needs accessibility access to monitor keyboard and mouse input.',
            'NSAccessibilityUsageDescription': 'ContinuousAuth needs accessibility access to monitor keyboard and mouse input for behavioural authentication.',
        },
    )
