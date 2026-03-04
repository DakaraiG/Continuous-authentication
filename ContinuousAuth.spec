# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['window', 'policy', 'db', 'capture', 'features', 'train_model', 'app_log', 'logger', 'sklearn.ensemble._forest', 'sklearn.ensemble._iforest', 'sklearn.svm._classes', 'sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes', 'sklearn.tree._utils', 'pynput.keyboard._win32', 'pynput.mouse._win32', 'pynput.keyboard._darwin', 'pynput.mouse._darwin', 'Quartz', '_datetime']
hiddenimports += collect_submodules('sklearn')
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
app = BUNDLE(
    coll,
    name='ContinuousAuth.app',
    icon=None,
    bundle_identifier='com.continuousauth.app',
)
