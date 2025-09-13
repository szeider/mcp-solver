# MCP Solver Setup Guide

Supports Windows, macOS and Linux.

Tested with:

- macOS Sequoia 15.3.2
- Windows 11 Pro Version 10.0.26100 Build 26100
- Ubuntu 24.04.2 LTS

Other versions may also work, as long as the required tools (Python 3.11+, pipx, uv, MiniZinc, etc.) are available or can be installed.

---

## Step 1: Install Python 3.11+

### macOS (via Homebrew)

```bash
brew install python
```

Verify:

```bash
python3 --version
```

---

### Windows (via Microsoft Store)

```powershell
winget install --id Python.Python.3.13
```

Verify:

```powershell
python --version
pip --version
```

---

### Linux (Ubuntu)

Ubuntu 24.04 ships with Python 3.12.3 which is fully compatible with MCP Solver. You can use the system-provided version without installing Python manually.

Verify:

```bash
python3 --version
```

---

## Step 2: Install `pipx` and `uv`

### macOS

```bash
brew install pipx
pipx ensurepath
pipx install uv
```

---

### Windows (PowerShell)

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
pipx install uv
```

If needed, add to PATH:

```
%USERPROFILE%\AppData\Roaming\Python\Python313\Scripts
```

---

### Linux

On Debian and Ubuntu, `pipx` is available as a system package and can be installed via APT. However, the `uv` tool is not part of the standard repositories and should be installed via `pipx`.

```bash
sudo apt install pipx
pipx ensurepath
pipx install uv
```

`uv` will be installed locally to `~/.local/bin`. Make sure this directory is included in your `PATH`.

Restart your shell if needed and ensure `uv` is in your PATH.

---

## Step 3: Set up the MCP Solver project

### Clone project

```bash
mkdir -p ~/projects/mcp-solver
cd ~/projects/mcp-solver
git clone https://github.com/szeider/mcp-solver.git .
```

### Create and activate virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
uv pip install -e ".[all]"
```

This installs the MCP Solver in editable mode.

### Install additional solvers

```bash
uv pip install "z3-solver>=4.12.1"
uv pip install python-sat
```

---

## Step 4: Install MiniZinc

### macOS

- Download from: [https://www.minizinc.org/software.html](https://www.minizinc.org/software.html)
- Install `.dmg`, move app to Applications
- Add to `.env`:

```env
PATH=/Applications/MiniZincIDE.app/Contents/Resources:$PATH
```

### Windows

- Install to: `C:\Program Files\MiniZinc`
- Add to PATH via system environment variables

### Linux

- Download from: [https://www.minizinc.org/software.html](https://www.minizinc.org/software.html)
- Extract and move binaries to `/opt/minizinc`, or use a package if available
- Add to `.env`:

```env
PATH=/opt/minizinc:$PATH
```

Ensure `minizinc --version` works in the shell.

## Step 5: Run verification tests

Run the following tests to verify your MCP Solver setup:

```bash
uv run test-setup-mzn
uv run test-setup-z3
uv run test-setup-pysat
uv run test-setup-maxsat
uv run test-setup-client
uv run test-setup-asp
```

## Step 6: Test Client Setup

The client requires an **API key** from an LLM provider, unless you run a local model.

### Anthropic API Key (Default)

By default, the client uses Anthropic Claude. Make sure you have an [**Anthropic API key**](https://www.anthropic.com/api). The preferred method is to add it to a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-...
```

Alternatively, you can export it as an environment variable:

#### macOS / Linux

```bash
export ANTHROPIC_API_KEY=sk-...
```

#### Windows (PowerShell)

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

### Using Other LLM Providers

The client supports multiple LLM providers through the `--mc` model code flag. The syntax follows this pattern:

```
XY:model                     # For cloud providers
LM:model@url                 # For local models via LM Studio (basic)
LM:model(param=value)@url    # For local models with parameters
```

You can also use parameters to configure local models:

```
LM:model(format=json)@url                      # Request JSON output
LM:model(temp=0.7)@url                         # Set temperature to 0.7
LM:model(format=json,temp=0.7,max_tokens=1000)@url  # Multiple parameters
```

Where `XY` is a two-letter code representing the platform:
- `OA`: OpenAI
- `AT`: Anthropic
- `OR`: OpenRouter
- `GO`: Google (Gemini)
- `LM`: LM Studio (local models)

Examples:
```
OA:gpt-4.1-2025-04-14
AT:claude-3-7-sonnet-20250219
OR:google/gemini-2.5-pro-preview
```

For providers other than Anthropic, you'll need to add the corresponding API key to your `.env` file:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-...
# No API key needed for LM Studio
```

Test the client setup:

```bash
uv run test-setup-client
```

You can now run a problem description 

```bash
# MiniZinc mode
uv run run-test mzn --problem <path/to/problem.md>

# PySAT mode
uv run run-test pysat --problem <path/to/problem.md>

# MaxSAT mode
uv run run-test maxsat --problem <path/to/problem.md>

# Z3 mode
uv run run-test z3 --problem <path/to/problem.md>

# ASP mode
uv run run-test asp --problem <path/to/problem.md>
```



Some problem descriptions are provided in `tests/problems`

---

## Step 7: Claude Desktop Setup

### macOS

Download from: [https://claude.ai/download](https://claude.ai/download)

### Windows

Download from: [https://claude.ai/download](https://claude.ai/download)

### Linux (unofficial workaround)

You can use the community wrapper from [aaddrick/claude-desktop-debian](https://github.com/aaddrick/claude-desktop-debian):

```bash
git clone https://github.com/aaddrick/claude-desktop-debian.git
cd claude-desktop-debian
sudo ./build-deb.sh
sudo dpkg -i /path/to/claude-desktop_0.9.0_amd64.deb
```

Start Claude Desktop. Note that the splash screen may misleadingly show 'Claude for Windows' even on Linux. For best results, set Google Chrome as your default browser before launching. Alternatively, you can log in using your email address and a token. Login may require a few attempts regardless of method.

```bash
xdg-settings set default-web-browser google-chrome.desktop # optionally
/usr/bin/claude-desktop --no-sandbox
```

---

### Configure `claude_desktop_config.json`

In the examples below, replace "mcp-solver-mzn" with "mcp-solver-pysat", "mcp-solver-maxsat", "mcp-solver-z3", or "mcp-solver-asp" depending on the mode you want to run the MCP Solver in.

#### macOS

The config file is located at `~/Library/Application\ Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/stefanszeider/git/mcp-solver",
        "run",
        "mcp-solver-mzn"
      ]
    }
  }
}
```

#### Windows (example path)

The config file is located at `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "cmd.exe",
      "args": [
        "/C",
        "cd C:\\Users\\AC Admin\\build\\mcp-solver && uv run mcp-solver-mzn"
      ]
    }
  }
}
```

#### Linux (example path)

The config file is located at `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "/bin/bash",
      "args": [
        "-c",
        "cd /path/to/mcp-solver && uv run mcp-solver-mzn"
      ]
    }
  }
}
```

---

### Usage:

- We strongly recommend using the [Claude Pro](https://claude.ai/) subscription to run the Claude 3.7 Sonnet.  
- When you start Claude Desktop (version > 0.8.0), you should see a *Plus Icon* and to the right of it a *Settings Slicer Icon*.
- When you click the Plus Icon, you should see "Add from MCP Solver", follow this, and add the instructions prompt to your conversations. 
- When you click the Settings Slider Icon, you can access all the tools of the MCP Solver, and enable/disable them individually; we recommend having all enabled. 
- Now you are ready to type your query to the MCP solver.