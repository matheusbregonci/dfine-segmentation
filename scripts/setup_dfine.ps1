<#
.SYNOPSIS
    Clona o repositório D-FINE-seg e instala as dependências com uv.

.DESCRIPTION
    O D-FINE-seg é um framework baseado em repositório (não um pacote pip).
    Este script clona-o em ./D-FINE-seg e roda `uv sync`. Os scripts wrappers
    deste projeto (train.py, predict.py, etc.) detectam esse clone automaticamente.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_dfine.ps1
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $ProjectRoot "D-FINE-seg"
$RepoUrl = "https://github.com/ArgoHA/D-FINE-seg.git"

# 1. Verifica git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git não encontrado no PATH. Instale o Git para Windows."
}

# 2. Clona (ou atualiza)
if (Test-Path (Join-Path $Target ".git")) {
    Write-Host "D-FINE-seg já clonado em $Target — atualizando..." -ForegroundColor Cyan
    git -C $Target pull --ff-only
} else {
    Write-Host "Clonando D-FINE-seg em $Target ..." -ForegroundColor Cyan
    git clone $RepoUrl $Target
}

# 3. Instala dependências com uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning "uv não encontrado. Instale-o: https://docs.astral.sh/uv/getting-started/installation/"
    Write-Warning "Depois rode dentro de $Target :  uv sync"
} else {
    Write-Host "Instalando dependências com uv sync ..." -ForegroundColor Cyan
    Push-Location $Target
    try { uv sync } finally { Pop-Location }
}

Write-Host "`nPronto. D-FINE-seg disponível em: $Target" -ForegroundColor Green
Write-Host "Os wrappers detectam esse caminho automaticamente." -ForegroundColor Green
