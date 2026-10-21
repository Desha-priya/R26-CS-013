; NeuraShield — Inno Setup installer script
; Build:  iscc installer\neurashield.iss   (after installer\build_all.py)
; Produces one OFFLINE installer: installer\Output\NeuraShield-Setup-1.0.0.exe

#define AppName "NeuraShield"
#define AppVersion "1.0.0"
#define AppPublisher "R26-CS-013 Project Team"
#define RootDir ".."

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\NeuraShield
DefaultGroupName=NeuraShield
OutputDir=Output
OutputBaseFilename=NeuraShield-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
DisableProgramGroupPage=yes
WizardStyle=modern
; Consent is shown as a required page before installation (see [Code]).

[Types]
Name: "full"; Description: "Full protection (all four layers)"
Name: "recommended"; Description: "Recommended (Zero-Trust + Ransomware + Content)"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "core"; Description: "Zero-Trust core, platform and console (required)"; Types: full recommended custom; Flags: fixed
Name: "ransomware"; Description: "Ransomware Killer (Layer 2)"; Types: full recommended custom
Name: "content"; Description: "Content Threat Detection — phishing, deepfake, voice (Layer 4)"; Types: full recommended custom
Name: "network"; Description: "Network Guardian (Layer 1) — requires Npcap and administrator rights"; Types: full custom

[Files]
; ONE executable carries the whole product: the app, the console, the supervisor and
; every layer (core/ZTA, network, ransomware, content). The supervisor starts a layer
; by launching this same exe as "NeuraShield.exe --run-layer <id>". Layout:
;   {app}\gateway\NeuraShield.exe  (+ _internal\ with all layers, libraries and models)
;   {app}\config\layers.json, product.json      {app}\data\  (writable state + logs)
; The app derives the install root from the exe's location ({app}\gateway\..).
; The component choices below decide which layers are ENABLED by default.
Source: "{#RootDir}\dist\gateway\NeuraShield\*"; DestDir: "{app}\gateway"; Flags: recursesubdirs ignoreversion; Components: core
; product.json is written in [Code] below (never ship the developer's copy: it may
; carry consent that was given on the dev machine).
Source: "{#RootDir}\config\*"; DestDir: "{app}\config"; Excludes: "product.json"; Flags: recursesubdirs ignoreversion; Components: core
; Npcap offline installer, run on demand
Source: "prerequisites\npcap-installer.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist; Components: network

[Dirs]
; Protected runtime data (ACLs tightened in [Code] after install)
Name: "{commonappdata}\NeuraShield"; Permissions: admins-full
Name: "{commonappdata}\NeuraShield\secrets"; Permissions: admins-full
; The desktop app runs as the signed-in user, not as admin, and writes its state
; (admin record, audit log, layer data, logs, product.json) under the install
; root. Secret files inside are locked to their owner by the supervisor (icacls).
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\config"; Permissions: users-modify

[Icons]
Name: "{group}\NeuraShield Console"; Filename: "{app}\gateway\NeuraShield.exe"
Name: "{group}\Uninstall NeuraShield"; Filename: "{uninstallexe}"
Name: "{autodesktop}\NeuraShield Console"; Filename: "{app}\gateway\NeuraShield.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
; Npcap (packet-capture driver for the Network Guardian) if that layer was selected.
; The free Npcap edition has no silent mode (/S is OEM-only), so its own installer
; window is shown. Npcap's licence: free for personal/educational use; commercial
; redistribution needs an Npcap OEM licence (npcap.com/oem).
Filename: "{tmp}\npcap-installer.exe"; StatusMsg: "Installing Npcap (packet capture driver) - follow its window..."; Components: network; Flags: waituntilterminated skipifdoesntexist
; (No Windows service: the exe is a normal program, not a service binary, so
; "sc create" produced a service that could never start - error 1053. Auto-start
; is done with a per-machine Run entry below; it launches the app in --background
; mode: console + tray, no window, and NOTHING monitors until consent is given.)
; Open the console for first-run admin setup
Filename: "{app}\gateway\NeuraShield.exe"; Description: "Open the NeuraShield console"; Flags: postinstall nowait skipifsilent

[Registry]
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "NeuraShield"; ValueData: """{app}\gateway\NeuraShield.exe"" --background"; Flags: uninsdeletevalue; Components: core

[UninstallRun]
; The console owns every layer (job object), so ending it ends them all.
Filename: "{sys}\taskkill.exe"; Parameters: "/F /T /IM NeuraShield.exe"; Flags: runhidden; RunOnceId: "KillApp"

[Code]
var
  ConsentPage: TOutputMsgMemoWizardPage;
  ConsentAccepted: TNewCheckBox;

procedure InitializeWizard;
begin
  ConsentPage := CreateOutputMsgMemoPage(wpWelcome,
    'Consent & Privacy Notice', 'Please read before installing',
    'NeuraShield protects THIS computer. To do so it analyses, on this machine only:',
    'NeuraShield — Consent & Privacy Notice'#13#10#13#10 +
    'What is processed (locally, never sent to any outside service):'#13#10 +
    '  - Keystroke TIMING (not the characters you type) for Zero-Trust checks.'#13#10 +
    '  - A webcam snapshot for liveness, only when a check is required.'#13#10 +
    '  - File activity in your folders, to detect and roll back ransomware.'#13#10 +
    '  - Network flow metadata (not packet contents).'#13#10 +
    '  - Email / call content you opt in to, for phishing and deepfake checks.'#13#10#13#10 +
    'Your rights:'#13#10 +
    '  - Zero-Trust is on by default; you may decline it below.'#13#10 +
    '  - Mailbox and call monitoring are separate opt-ins inside the console.'#13#10 +
    '  - You can withdraw consent at any time; all actions are recorded in a'#13#10 +
    '    tamper-evident audit log you can inspect and export.'#13#10#13#10 +
    'All processing stays on this computer. Nothing leaves the machine.');

  ConsentAccepted := TNewCheckBox.Create(ConsentPage);
  ConsentAccepted.Parent := ConsentPage.Surface;
  ConsentAccepted.Top := ConsentPage.RichEditViewer.Top + ConsentPage.RichEditViewer.Height + 8;
  ConsentAccepted.Width := ConsentPage.SurfaceWidth;
  ConsentAccepted.Caption := 'I have read and accept the consent and privacy notice.';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = ConsentPage.ID then
  begin
    if not ConsentAccepted.Checked then
    begin
      MsgBox('You must accept the consent notice to install NeuraShield.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigFile: String;
  Content: AnsiString;
begin
  if CurStep = ssPostInstall then
  begin
    // Record the user's component choices so the console's first-run setup
    // starts from what was installed. Consent is NOT recorded here: it is given
    // once, in the console, when the administrator account is created. Until
    // then "accepted" is false and the supervisor starts nothing at all.
    ConfigFile := ExpandConstant('{app}\config\product.json');
    Content :=
      '{'#13#10 +
      '  "product_name": "NeuraShield",'#13#10 +
      '  "version": "1.0.0",'#13#10 +
      '  "console": { "host": "127.0.0.1", "port": 9000 },'#13#10 +
      '  "platform_url": "http://127.0.0.1:8000",'#13#10 +
      '  "enabled_layers": {'#13#10 +
      '    "zta": true,'#13#10 +
      '    "network": ' + LowerCase(BoolToStr(IsComponentSelected('network'))) + ','#13#10 +
      '    "ransomware": ' + LowerCase(BoolToStr(IsComponentSelected('ransomware'))) + ','#13#10 +
      '    "content": ' + LowerCase(BoolToStr(IsComponentSelected('content'))) + #13#10 +
      '  },'#13#10 +
      '  "consent": { "accepted": false, "accepted_at": null }'#13#10 +
      '}'#13#10;
    SaveStringToFile(ConfigFile, Content, False);
  end;
end;
