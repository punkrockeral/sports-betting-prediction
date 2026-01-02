# Descarga segura del dataset de Kaggle
$USERNAME = "punkrockeral"
$KEY = "61fa2fa4c66c6320a434156ac6c60142"  # ← ¡Clave real de kaggle.json!
$URL = "https://www.kaggle.com/api/v1/datasets/download/martj42/football-match-results-with-betting-odds"

# Descargar con sintaxis correcta
& curl.exe -u "${USERNAME}:${KEY}" -L -o "football-odds-dataset.zip" $URL

# Verificar descarga
if ((Get-Item "football-odds-dataset.zip").Length -lt 100000) {
    Write-Host "❌ ERROR: La descarga falló. ¿Clave de API correcta?"
    exit 1
}

# Crear carpeta y extraer
$dir = "data/processed"
mkdir -Force $dir
Expand-Archive -Path "football-odds-dataset.zip" -DestinationPath $dir -Force

# Renombrar
if (Test-Path "$dir/results.csv") {
    Move-Item "$dir/results.csv" "$dir/football_matches_with_odds.csv" -Force
    Write-Host "✅ ¡Listo! Dataset en data/processed/football_matches_with_odds.csv"
} else {
    Write-Host "⚠️  No se encontró results.csv"
}