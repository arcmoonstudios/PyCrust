# PyCrust Launcher (v4.0.2 Orchestrator)

This repository contains the `pycrust.py` bootstrap script. This script is a lightweight launcher that, when run, will set up the "Xyn's Decruster" environment and fetch the powerful **PyDecruster Engine** (from a separate, private repository) to analyze and assist with your Rust projects.

## Prerequisites

1. **Git CLI**: Ensure `git` is installed and accessible in your system's PATH.
   * Verify with: `git --version`

2. **Python**: Python 3.8 or newer is recommended.
   * Verify with: `python --version` or `python3 --version`

3. **GitHub Personal Access Token (PAT)**: The PyDecruster Engine is hosted in a private repository (`arcmoonstudios/pyDecruster`). To allow `pycrust.py` to clone it, you need to:

   * Generate a GitHub Personal Access Token (Classic or Fine-Grained).
   * **Permissions for Fine-Grained PAT**:
     * Repository access: "Only select repositories" -> `arcmoonstudios/pyDecruster`
     * Repository permissions: **Contents** -> **Read-only`.
   * Set this token as an environment variable named `PYDECRUSTER_TOKEN`.

## Quick Start

1. **Set `PYDECRUSTER_TOKEN`**: Follow the instructions above to set your GitHub PAT.

2. **Download the Launcher Script (`pycrust.py`)**:

   Open your terminal and run:

   * **PowerShell:**

     ```powershell
     curl.exe -L "https://raw.githubusercontent.com/arcmoonstudios/pycrust-launcher/Main/pycrust.py" -o pycrust.py
     ```

   * **bash/zsh:**

     ```bash
     curl -L "https://raw.githubusercontent.com/arcmoonstudios/pycrust-launcher/Main/pycrust.py" -o pycrust.py
     ```

3. **Run PyCrust**:

   Navigate to your Rust project's root and execute:

   ```powershell
   python pycrust.py

   # Or target a specific path:
   # python path/to/pycrust.py path/to/your/rust_project
   ```

4. **Set Up Python Environment for PyDecruster Engine**:

   After the first run (which clones `pyDecruster` and copies `reqenv.py`),:

   * Navigate to the project root (alongside `pycrust.py`).
   * Run `python reqenv.py`. This will:
     * Create a virtual environment named `PycRustEnv`.
     * Provide activation instructions.
     * After activation, rerun `python reqenv.py` to install dependencies.

   **Example:**

   ```powershell
   cd /path/to/your/rust_project
   python reqenv.py
   # Then (PowerShell):
   # . .\PycRustEnv\Scripts\activate.ps1
   # (PycRustEnv) python reqenv.py
   ```

5. **Use PyCrust**:

   With `PycRustEnv` activated:

   ```powershell
   python pycrust.py --scan
   python pycrust.py --cli
   python pycrust.py --help
   ```

## How It Works

`pycrust.py` is a bootstrap script that, on first run, will:

1. Create `./Xyn's Decruster/` in your Rust project.
2. Use `PYDECRUSTER_TOKEN` to sparse-clone `arcmoonstudios/pyDecruster` into that directory.
3. Copy setup files (`reqenv.py`, `README.md`) for the PyDecruster environment.
4. Use the local engine on subsequent runs.

This minimizes initial download and protects core IP.

## Troubleshooting

* **`PYDECRUSTER_TOKEN` not set**: Ensure the environment variable is set.
* **`git` not found**: Install `git` and add to PATH.
* **Cloning fails**:
  * Verify `PYDECRUSTER_TOKEN` has correct scopes and access.
  * Check your internet connection.
* **`reqenv.py` issues**:
  * Ensure Python's `venv` module is available.
  * Follow the shell-specific activation instructions.
  * Manually install dependencies (e.g., PyCUDA) if pip fails.

## License

This launcher script (`pycrust.py`) is distributed under MIT. The PyDecruster Engine has its own license in its repository.
