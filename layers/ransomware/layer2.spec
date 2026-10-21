# PyInstaller spec for the FINAL Layer 2 build:   pyinstaller layer2.spec
# One-folder build (more reliable with antivirus than one-file). Bundles the
# config, trained models + manifest, dashboard pages and static files.
from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("src") + ["waitress", "sklearn.ensemble._forest", "sklearn.ensemble._iforest"]

a = Analysis(
    ["main.py"],
    pathex=["."],
    datas=[("config", "config"), ("models", "models"),
           ("src/dashboard/templates", "src/dashboard/templates"),
           ("src/dashboard/static", "src/dashboard/static")],
    hiddenimports=hidden,
    excludes=["matplotlib", "seaborn", "pytest", "tkinter"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NeuraShieldLayer2", console=True, uac_admin=False)
coll = COLLECT(exe, a.binaries, a.datas, name="NeuraShieldLayer2")
