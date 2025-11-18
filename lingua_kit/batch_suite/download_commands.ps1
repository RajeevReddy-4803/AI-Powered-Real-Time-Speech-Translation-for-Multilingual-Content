# PowerShell script to download language-specific files using HuggingFace CLI
# Since you're on Windows, use this instead of .sh file

Write-Host "📥 Downloading audio dataset files for 11 languages..." -ForegroundColor Cyan

$languages = @("Hindi", "Punjabi", "Marathi", "Kannada", "Telugu", "Tamil", "Gujarati", "Malayalam", "Bengali", "Odia", "Urdu")

foreach ($lang in $languages) {
    Write-Host "`n📥 Downloading $lang..." -ForegroundColor Yellow
    huggingface-cli download ai4bharat/indicvoices_r --repo-type dataset `
        --local-dir ".\datasets_cache\$lang" `
        --include "$lang/*.parquet"
}

Write-Host "`n✅ Download complete! Files saved in .\datasets_cache\" -ForegroundColor Green

