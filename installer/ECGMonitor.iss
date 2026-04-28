#ifndef MyAppName
  #define MyAppName "ECG Monitor"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "ECGMonitor.exe"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "ECG Monitor"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "2.0.0"
#endif
#ifndef MyAppChannel
  #define MyAppChannel "stable"
#endif
#ifndef MyAppDistDir
  #define MyAppDistDir "..\dist\ECGMonitor"
#endif
#ifndef MyAppOutputDir
  #define MyAppOutputDir "..\dist_installer"
#endif
#ifndef MyAppURL
  #define MyAppURL "https://example.com"
#endif
; NOTE: This installer packages the recommended PyInstaller ONEDIR build output:
;       dist\ECGMonitor\ECGMonitor.exe + dist\ECGMonitor\_internal\...

[Setup]
AppId={{B29D5C9D-67E6-4F8C-8EF0-DBE8E2F0C5EA}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}
LicenseFile=EULA.txt
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir={#MyAppOutputDir}
OutputBaseFilename=Setup_{#MyAppName}_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
ChangesEnvironment=no
SetupLogging=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Main app folder (PyInstaller ONEDIR)
Source: "{#MyAppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Runtime folders created/used by the app
Name: "{app}\logs"
Name: "{app}\reports"
Name: "{app}\offline_queue"
Name: "{app}\recordings"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
; ECGMonitor.exe is currently built with `--uac-admin` (requires elevation).
; Use the `runas` verb so the post-install launch works (UAC prompt).
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Verb: "runas"; Flags: shellexec nowait postinstall skipifsilent
