# MCP Solver Setup Guide

Supports Windows, macOS and Linux.

Tested with:

- macOS Sequoia 15.3.2
- Windows 11 Pro Version 10.0.26100 Build 26100
- Ubuntu 24.04.2 LTS

Other versions may also work, as long as the required tools (Python 3.11+, pipx, uv, MiniZinc, etc.) are available or can be installed.

---

## Step 1: Install Python 3.13+

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
git clone --branch z3 https://github.com/szeider/mcp-solver.git .
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
uv pip install -e .
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

### Python bindings

```bash
uv pip install minizinc rich langchain_mcp_adapters langchain_google_genai
```

## Step 5: Run verification tests

Before running the tests, make sure your Claude API key is set. You can either export it as an environment variable or define it in a `.env` file.

### macOS / Linux

```bash
export ANTHROPIC_API_KEY=sk-...
```

### Windows (PowerShell)

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

Or add to `.env` file:

```env
ANTHROPIC_API_KEY=sk-...
```

Run the following tests to verify your full MCP Solver setup:

```bash
uv run test-setup-mzn
uv run test-setup-z3
uv run test-setup-pysat
uv run test-setup-client
```

All tests should pass.

---

## Step 6: Claude Desktop Setup

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

#### macOS

The config file is located at `~/Library/Application\ Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "/path/to/mcp-solver/.venv/bin/python",
      "args": [
        "-m",
        "mcp_solver"
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
      "command": "C:\\path\\to\\mcp-solver\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "mcp_solver"
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
      "command": "/path/to/mcp-solver/.venv/bin/python",
      "args": [
        "-m",
        "mcp_solver.core.server"
      ]
    }
  }
}
```

---

You're now ready to use the solver! 