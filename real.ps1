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

# Function to clean up the user data directory
function Cleanup {
    Write-Host "Cleaning up..."
    Remove-Item -Path $folderPath -Recurse -Force
    Write-Host "User data directory deleted."
}

# Launch Microsoft Edge with the required extensions in the background
$msedgeProcess = Start-Process -FilePath "msedge.exe" `
    -ArgumentList "--user-data-dir=$folderPath", "--load-extension=$($folderPath)\extensionne\Propeller-main,$($folderPath)\extensionne\thottathukiduven-v2-main", "--no-first-run" `
    -PassThru

# Wait for Edge to close
$msedgeProcess.WaitForExit()

# Run cleanup after Edge closes
Cleanup
