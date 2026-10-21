# Building the NeuraShield Windows installer

One command turns this source tree into one offline installer:

```powershell
python installer\build_release.py          # envs + all exes + installer
```

Output: `installer\Output\NeuraShield-Setup-<version>.exe`.

The build runs **on Windows x64 with Python 3.11** (the layers' pinned wheels —
torch 2.6, tensorflow 2.16 — are for 3.11). Install **Inno Setup 6** for the last step.
Optional: put the Npcap offline installer at `installer\prerequisites\npcap-installer.exe`
so the Network Guardian can capture packets (the installer skips it if absent).

## What `build_release.py` does

1. **Preflight** – every model the product needs is present (ZTA Siamese checkpoint and
   scaler, content text/video/audio models, ransomware classifier).
2. **Isolated environments** – `scripts\bootstrap_envs.py` creates `runtimes\<id>\` for
   gateway, core, network, ransomware, content. They cannot share one environment
   (numpy 2.x vs 1.x, protobuf, tensorflow vs torch pins).
3. **Freeze** – `installer\build_all.py` builds each component with *its own* environment's
   PyInstaller (one-folder builds):

   ```
   dist\gateway\NeuraShield\NeuraShield.exe              desktop app + console + supervisor
   dist\core\NeuraShieldCore\NeuraShieldCore.exe          Zero-Trust platform (Layer 3)
   dist\network\NeuraShieldNetwork\NeuraShieldNetwork.exe Network Guardian (Layer 1)
   dist\ransomware\NeuraShieldRansom\NeuraShieldRansom.exe Ransomware Killer (Layer 2)
   dist\content\NeuraShieldContent\NeuraShieldContent.exe Content Threat (Layer 4)
   ```
4. **Check** the five exes exist.
5. **Compile** `installer\neurashield.iss`.

Individual steps: `python installer\build_all.py content` (one component),
`python installer\build_release.py --skip-envs`, `--no-installer`.
`NEURASHIELD_BUILD_OUT=<dir>` builds somewhere other than the repo root.

## Installed layout (what the code expects)

```
<Program Files>\NeuraShield\
  gateway\NeuraShield.exe        <- the app; install root = its parent folder's parent
  core\NeuraShieldCore\NeuraShieldCore.exe
  network|ransomware|content\<Name>\<Name>.exe     (only the components chosen)
  config\layers.json, product.json                 (writable by the user)
  data\                                            (writable: admin record, audit, logs, per-layer data)
```

`config\layers.json` names each layer's exe (`frozen_exe`); the supervisor starts it
with the shared API key in its environment. In a source run it starts the layer's
Python entry point instead - same wiring.

## Install-time behaviour

* The installer shows the consent notice, then lets the user pick components
  (Zero-Trust core is always installed).
* It writes `product.json` with **consent not accepted**. Consent is given once, when the
  administrator account is created in the console; nothing starts before that.
* Auto-start at sign-in is a `Run` registry entry launching `NeuraShield.exe --background`
  (console + tray, no window). The tray's *Open* / *Quit* work from there.
* Secrets: the platform key and signing secret are DPAPI-protected; key files are
  restricted to the signed-in user.

## When something fails

* Each layer's output is in `data\logs\<layer>.log`; the app's in `data\logs\app.log`.
* **Console → Self-check** shows "Shared API key": it proves every layer accepts the one
  platform key and rejects any other.
* A model that cannot load logs a line starting `✗` in `content.log`; the content layer
  loads its video model from the `.h5` weights directly, so a Keras version mismatch no
  longer breaks it.
