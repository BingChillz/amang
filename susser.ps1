# Generate a random string of 7 alphabetic characters for the folder name
$folderName = -join ((65..90) + (97..122) | Get-Random -Count 7 | ForEach-Object {[char]$_})
New-Item -ItemType Directory -Path $folderName | Out-Null
Set-Location -Path $folderName
$folderPath = (Get-Location).Path
New-Item -ItemType Directory -Path "extensionne" | Out-Null
Set-Location -Path "extensionne"

# Download and unzip the required files
Invoke-WebRequest -Uri "https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip" -OutFile "propeller.zip"
Expand-Archive -Path "propeller.zip" -DestinationPath .
Remove-Item -Path "propeller.zip"

Invoke-WebRequest -Uri "https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip" -OutFile "thottathukiduven-v2.zip"
Expand-Archive -Path "thottathukiduven-v2.zip" -DestinationPath .
Remove-Item -Path "thottathukiduven-v2.zip"

# Additional downloads
Invoke-WebRequest -Uri "https://github.com/jswanner/DontF-WithPaste/archive/refs/heads/master.zip" -OutFile "paste.zip"
Expand-Archive -Path "paste.zip" -DestinationPath .
Remove-Item -Path "paste.zip"

Invoke-WebRequest -Uri "https://github.com/brian-girko/always-active/archive/refs/heads/master.zip" -OutFile "window.zip"
Expand-Archive -Path "window.zip" -DestinationPath .
Remove-Item -Path "window.zip"

# Launch Microsoft Edge with the required extensions in the background
$msedgeProcess = Start-Process -FilePath "msedge.exe" `
    -ArgumentList "--user-data-dir=$folderPath", "--load-extension=$($folderPath)\extensionne\Propeller-main,$($folderPath)\extensionne\thottathukiduven-v2-main,$($folderPath)\extensionne\DontF-WithPaste-master,$($folderPath)\extensionne\always-active-master\v3", "--no-first-run" `
    -PassThru

# Background job to wait for Edge to close and clean up the directory
Start-Job -ScriptBlock {
    param ($folderPath, $msedgeProcessId)
    
    # Wait for the Edge process to exit
    do {
        Start-Sleep -Seconds 2
    } while (Get-Process -Id $msedgeProcessId -ErrorAction SilentlyContinue)

    # Move out of the directory to avoid locking issues
    Set-Location -Path "$env:USERPROFILE"
    
    # Clean up the user data directory
    Write-Host "Cleaning up..."
    Remove-Item -Path $folderPath -Recurse -Force
    Write-Host "User data directory deleted."
} -ArgumentList $folderPath, $msedgeProcess.Id
