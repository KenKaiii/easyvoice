# EasyVoice Universal Installer (PowerShell)
# One-command installation for Windows
# Usage: irm [URL]/install-easyvoice.ps1 | iex

param(
    [switch]$Verbose = $false
)

# Set error action
$ErrorActionPreference = "Stop"

# Repository information
$RepoUrl = "https://raw.githubusercontent.com/KenKaiii/easyvoice/main"
$ProjectName = "EasyVoice"

# Helper functions
function Write-Header {
    Write-Host ""
    Write-Host "┌─────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│       🎤 EasyVoice Installer       │" -ForegroundColor Cyan
    Write-Host "│   Lightweight Voice Agent CLI      │" -ForegroundColor Cyan
    Write-Host "│        Created by Ken Kai           │" -ForegroundColor Cyan
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

function Set-OpenAIAPI {
    Write-Host ""
    Write-Step "Setting up OpenAI API for seamless experience..."
    
    # Check if API key is already set
    if ($env:OPENAI_API_KEY) {
        Write-Success "OpenAI API key already configured"
        return $true
    }
    
    Write-Host ""
    Write-Host "EasyVoice uses OpenAI for the best voice AI experience." -ForegroundColor Cyan
    Write-Host "You'll need an OpenAI API key (get one at: https://platform.openai.com/api-keys)" -ForegroundColor Yellow
    Write-Host ""
    
    # Prompt for API key
    $apiKey = Read-Host "Enter your OpenAI API key [or press Enter to skip]"
    
    if ($apiKey) {
        # Set environment variable for current session
        $env:OPENAI_API_KEY = $apiKey
        
        # Add to PowerShell profile for persistence
        try {
            $profilePath = $PROFILE.CurrentUserAllHosts
            if (-not (Test-Path $profilePath)) {
                New-Item -ItemType File -Path $profilePath -Force | Out-Null
            }
            Add-Content -Path $profilePath -Value "`n`$env:OPENAI_API_KEY = `"$apiKey`""
            Write-Success "OpenAI API key saved to PowerShell profile"
        }
        catch {
            Write-Warning "API key set for this session only"
            Write-Step "Add '`$env:OPENAI_API_KEY = `"$apiKey`"' to your PowerShell profile"
        }
        
        return $true
    }
    else {
        Write-Warning "Skipping API key setup - you can set it later with:"
        Write-Step "`$env:OPENAI_API_KEY = `"your-key-here`""
        return $false
    }
}

function Test-Installation {
    Write-Step "Verifying installation..."
    
    try {
        $version = & easyvoice --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "EasyVoice installed successfully! Version: $version"
            
            # Test basic functionality if API key is set
            if ($env:OPENAI_API_KEY) {
                Write-Step "Testing OpenAI connection..."
                try {
                    $testResult = & timeout 10 easyvoice ask "Hello" 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "OpenAI integration working!"
                    }
                    else {
                        Write-Warning "OpenAI test failed - check your API key and internet connection"
                    }
                }
                catch {
                    Write-Warning "OpenAI test failed - check your API key and internet connection"
                }
            }
            
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
    Write-Host "🎉 EasyVoice is Ready!" -ForegroundColor Green
    Write-Host ""
    
    if ($env:OPENAI_API_KEY) {
        Write-Host "✅ OpenAI API configured - you're all set!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Try it now:" -ForegroundColor Cyan
        Write-Host "  easyvoice                    # Start interactive mode"
        Write-Host "  easyvoice chat               # Start text chat"
        Write-Host "  easyvoice ask `"Hello!`"       # Ask your first question"
    }
    else {
        Write-Host "⚠️  OpenAI API key needed for full functionality" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To get started:" -ForegroundColor Cyan
        Write-Host "  1. Get an API key at: https://platform.openai.com/api-keys"
        Write-Host "  2. Run: `$env:OPENAI_API_KEY = `"your-key-here`""
        Write-Host "  3. Then try: easyvoice ask `"Hello!`""
    }
    
    Write-Host ""
    Write-Host "All Commands:" -ForegroundColor Cyan
    Write-Host "  easyvoice                    # Interactive mode with menu"
    Write-Host "  easyvoice chat               # Start text chat"  
    Write-Host "  easyvoice ask `"question`"     # Ask single question"
    Write-Host "  easyvoice --help             # Show all options"
    Write-Host ""
    Write-Host "Optional Voice Features:" -ForegroundColor Cyan
    Write-Host "  For voice conversations, install audio dependencies:"
    Write-Host "  pip install 'easyvoice[audio]'"
    Write-Host ""
    Write-Host "🎤 Welcome to EasyVoice!" -ForegroundColor Green
    Write-Host "Created by Ken Kai - AI Developer" -ForegroundColor Cyan
    Write-Host "Follow more AI projects: Ken Kai does AI" -ForegroundColor Cyan
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
    
    # Setup OpenAI API for seamless experience
    Set-OpenAIAPI | Out-Null
    
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