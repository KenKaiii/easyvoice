# EasyVoice Universal Installer (PowerShell)
# One-command installation for Windows
# Usage: irm [URL]/install-easyvoice.ps1 | iex

param(
    [switch]$Verbose = $false
)

# Set error action
$ErrorActionPreference = "Stop"

# Repository information
$RepoUrl = "https://raw.githubusercontent.com/your-username/easyvoice/main"
$ProjectName = "EasyVoice"

# Helper functions
function Write-Header {
    Write-Host ""
    Write-Host "┌─────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│       🎤 EasyVoice Installer       │" -ForegroundColor Cyan
    Write-Host "│   Lightweight Voice Agent CLI      │" -ForegroundColor Cyan
    Write-Host "└─────────────────────────────────────┘" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-PythonVersion {
    Write-Step "Checking Python installation..."
    
    try {
        $pythonCmd = Get-Command python -ErrorAction Stop
        $version = & python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
        $versionCheck = & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python $version found"
            return $true
        } else {
            Write-Error "Python 3.10+ required, found $version"
            return $false
        }
    }
    catch {
        try {
            $pythonCmd = Get-Command python3 -ErrorAction Stop
            $version = & python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
            $versionCheck = & python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Python3 $version found"
                return $true
            } else {
                Write-Error "Python 3.10+ required, found $version"
                return $false
            }
        }
        catch {
            Write-Error "Python not found"
            Write-Step "Please install Python 3.10+ from https://python.org"
            return $false
        }
    }
}

function Install-EasyVoice {
    Write-Step "Downloading PowerShell installer..."
    
    try {
        $tempDir = New-TemporaryFile | %{ rm $_; mkdir $_ }
        $installerPath = Join-Path $tempDir "install.ps1"
        
        Invoke-WebRequest -Uri "$RepoUrl/install.ps1" -OutFile $installerPath -UseBasicParsing
        
        Write-Step "Running install.ps1..."
        & PowerShell -ExecutionPolicy Bypass -File $installerPath
        
        # Cleanup
        Remove-Item $tempDir -Recurse -Force
        
        return $true
    }
    catch {
        Write-Error "Failed to download or run installer: $($_.Exception.Message)"
        return $false
    }
}

function Test-Installation {
    Write-Step "Verifying installation..."
    
    try {
        $version = & easyvoice --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "EasyVoice installed successfully! Version: $version"
            return $true
        }
    }
    catch {
        # Command not found
    }
    
    Write-Warning "easyvoice command not found in PATH"
    Write-Step "You may need to restart your terminal or add Python Scripts to PATH"
    return $false
}

function Show-NextSteps {
    Write-Host ""
    Write-Host "🎉 Installation Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Quick Start:" -ForegroundColor Cyan
    Write-Host "  easyvoice                    # Start interactive mode"
    Write-Host "  easyvoice chat               # Start text chat"
    Write-Host "  easyvoice ask `"question`"     # Ask single question"
    Write-Host "  easyvoice --help             # Show all options"
    Write-Host ""
    Write-Host "Optional Audio Dependencies:" -ForegroundColor Cyan
    Write-Host "  For voice features, install audio dependencies:"
    Write-Host "  pip install 'easyvoice[audio]'"
    Write-Host ""
    Write-Host "Configuration:" -ForegroundColor Cyan
    Write-Host "  Set OPENAI_API_KEY for OpenAI models"
    Write-Host "  Or configure Ollama for local models"
    Write-Host ""
    Write-Host "Need help? Check the documentation or run 'easyvoice --help'" -ForegroundColor Yellow
}

function Main {
    Write-Header
    
    # Check Python
    if (-not (Test-PythonVersion)) {
        Write-Step "Please install Python 3.10+ and try again"
        exit 1
    }
    
    # Install EasyVoice
    if (-not (Install-EasyVoice)) {
        Write-Error "Installation failed"
        exit 1
    }
    
    # Verify installation
    if (Test-Installation) {
        Show-NextSteps
    } else {
        Write-Error "Installation verification failed"
        Write-Step "Please check the output above for errors"
        Write-Step "You can try running 'pip install --user .' manually"
        exit 1
    }
}

# Run main function
Main