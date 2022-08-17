# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
$install_dir = $args[0]

if ( Get-Command "conda" -ErrorAction SilentlyContinue ) {
    Write-Host "WARNING: conda is already installed and available on command line."
}

if ($args.Count -ne 1) {
    Write-Host "You did not say where to install conda"
    Write-Host "Usage: powershell install_conda.ps1 /path/to/install"
    exit
}

$install_dir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($install_dir) # why resolving a path is so hard
$install_dir = $install_dir -replace "/","\"
Write-Host Installing conda to $install_dir
New-Item $install_dir -ItemType Directory -Force

## Download installer
$AllProtocols = [System.Net.SecurityProtocolType]'Ssl3,Tls,Tls11,Tls12'
[System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols
$SCRIPT="Miniforge3-Windows-x86_64.exe"
$URL="https://github.com/conda-forge/miniforge/releases/latest/download/$SCRIPT"
Invoke-WebRequest -Uri $URL -OutFile "$SCRIPT"

## Install
Write-Host "Installing conda locally"
# The -Wait is important
Write-Host Running Start-Process -Wait -FilePath "$SCRIPT" -ArgumentList "/S","/AddToPath=0","/RegisterPython=0","/D=${install_dir}"
Start-Process -Wait -FilePath "$SCRIPT" -ArgumentList "/S","/AddToPath=0","/RegisterPython=0","/D=${install_dir}"
if( -not $? ) { Write-Host "Failed to install conda"; exit -1 }
Remove-Item "$SCRIPT"
