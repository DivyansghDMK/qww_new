#ifndef MyAppName
  #define MyAppName "ECGMonitor"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "Deckmount Electronics"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "ECGMonitor.exe"
#endif
#ifndef MyAppDistDir
  #define MyAppDistDir "..\\dist\\ECGMonitor"
#endif
#ifndef MyAppOutputDir
  #define MyAppOutputDir "..\\dist\\installers"
#endif

[Setup]
AppId={{9DCE38F3-0F26-4A68-9A56-7A6A27312361}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir={#MyAppOutputDir}
OutputBaseFilename={#MyAppName}_Setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Dirs]
Name: "{localappdata}\\Deckmount\\ECGMonitor"
Name: "{localappdata}\\Deckmount\\ECGMonitor\\reports"
Name: "{localappdata}\\Deckmount\\ECGMonitor\\logs"
Name: "{localappdata}\\Deckmount\\ECGMonitor\\offline_queue"
Name: "{localappdata}\\Deckmount\\ECGMonitor\\temp"

[Files]
Source: "{#MyAppDistDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"; WorkingDir: "{localappdata}\\Deckmount\\ECGMonitor"
Name: "{autodesktop}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"; WorkingDir: "{localappdata}\\Deckmount\\ECGMonitor"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{localappdata}\\Deckmount\\ECGMonitor"; Flags: nowait postinstall skipifsilent
