# PowerShell script to download and install FFmpeg
Write-Host "Downloading FFmpeg..." -ForegroundColor Cyan

$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$downloadPath = "$env:TEMP\ffmpeg.zip"
$extractPath = "C:\ffmpeg"

# Download FFmpeg
try {
    Write-Host "Downloading from $ffmpegUrl" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath -UseBasicParsing
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "Error downloading: $_" -ForegroundColor Red
    Write-Host "Please download manually from: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
    exit 1
}

# Extract
try {
    Write-Host "Extracting to $extractPath..." -ForegroundColor Yellow
    if (-not (Test-Path $extractPath)) {
        New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
    }
    Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    Write-Host "Extraction complete!" -ForegroundColor Green
} catch {
    Write-Host "Error extracting: $_" -ForegroundColor Red
    exit 1
}

# Find ffmpeg.exe
$ffmpegBin = Get-ChildItem -Path $extractPath -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if ($ffmpegBin) {
    $ffmpegDir = $ffmpegBin.DirectoryName
    Write-Host "Found FFmpeg at: $ffmpegDir" -ForegroundColor Green
    
    # Add to PATH for current session
    $env:PATH += ";$ffmpegDir"
    
    Write-Host "`nFFmpeg installed! Testing..." -ForegroundColor Cyan
    & "$ffmpegDir\ffmpeg.exe" -version | Select-Object -First 1
    
    Write-Host "`nTo make permanent, add this to your PATH:" -ForegroundColor Yellow
    Write-Host "$ffmpegDir" -ForegroundColor White
} else {
    Write-Host "Could not find ffmpeg.exe" -ForegroundColor Red
}

Write-Host "`nDone! You can now run the batch translator." -ForegroundColor Green

