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

### Clone the main barnch of the project

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
uv pip install -e ."[all]"
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
uv run test-setup-client
```

## Step 6: Test Client Setup

If you want to run the test client, make sure you have an [**Anthropic API key**](https://www.anthropic.com/api). You can either export it as an environment variable or define it in a `.env` file.

### macOS / Linux

```bash
export ANTHROPIC_API_KEY=sk-...
```

### Windows (PowerShell)

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

Or add to an `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-...
```

Test the client setup:

```bash
uv run test-setup-client
```

You can now run a problem description 

```bash
# MiniZinc mode
uv run test-client --query <query_file>.md

# PySAT mode
uv run test-client-pysat --query <query_file>.md

# Z3 mode
uv run test-client-z3 --query <query_file>.md
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

In the examples below, replace "mcp-solver-mzn" with "mcp-solver-pysat" or "mcp-solver-z3" depeding on the mode you want to run the MCP Solver in.

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
- We strongly recommend using the [Claude Pro](https://claude.ai/) subscription to access Claude 3.7 Sonnet model capabilities.
- Upon launching Claude Desktop, verify that the interface displays a hammer symbol indicating at least 6 available tools.
- Locate the electrical plug symbol in the interface. Select this icon, then choose "choose an integration" followed by "MCP Solver instructions." This action should result in the attachment of a prompt file named `instructions.txt`.
- Once these prerequisites are satisfied, you may proceed to submit your query to the MCP solver.
