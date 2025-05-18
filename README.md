# PyCrust: Advanced Rust Project Analysis & Refactoring Framework

This repository contains the `pycrust.py` orchestrator script that bootstraps the powerful **Xyn's Decruster** framework for analyzing and enhancing Rust projects.

## Why This Approach?

Originally, users directly cloned the `pyDecruster` repository, but this caused numerous issues:

1. Many users faced setup problems without the proper environment configuration
2. Direct cloning bypassed essential initialization steps needed for proper framework operation
3. Several users reported functionality issues due to incorrect configuration

After receiving multiple issue reports, I revamped the entire approach to ensure a seamless experience. The current architecture uses this `pycrust.py` script as a specialized orchestrator that:

1. Automatically sets up the correct directory structure
2. Handles dependency management through an integrated virtual environment
3. Configures the framework correctly with zero user interaction required
4. Properly initializes all components needed for the advanced analysis engine

This approach allows me to continue actively developing the core `pyDecruster` engine while ensuring everyone who uses it gets a properly configured installation that just works.

## Implementation Architecture Evolution

The framework architecture evolved in response to systematic deployment issues:

```markdown
HISTORICAL_DEPLOYMENT_PATTERN:
│
├── PHASE_1: Direct Repository Access [DEPRECATED]
│   └── Users directly cloned pyDecruster repository
│       └── ERROR: Framework initialization failures
│           └── ISSUE: Missing environment configuration
│           └── ISSUE: Improper directory structures
│           └── ISSUE: Unresolved dependency chains
│
└── PHASE_2: Orchestrated Deployment [CURRENT]
    └── Two-phase initialization process
        ├── Bootstrap: Clone PyCrust repository
        └── Orchestration: Execute pycrust.py
            └── Runtime sparse-cloning of framework components
            └── Automated environment preparation
            └── Proper initialization sequence enforcement
```

This architectural approach ensures controlled deployment with precise initialization sequencing, preventing the configuration failures reported by multiple users in the previous direct access pattern.

## System Prerequisites

```python
REQUIRED_COMPONENTS = {
    "git": {
        "verification_command": "git --version",
        "minimum_version": "2.20.0",
        "purpose": "Repository management and sparse checkout operations"
    },
    "python": {
        "verification_command": "python --version || python3 --version",
        "minimum_version": "3.8.0",
        "purpose": "Execution environment for framework orchestration"
    }
}
```

## Implementation Procedure

### Deployment Protocol

```bash
# PHASE 1: Bootstrap Repository Acquisition

git clone https://github.com/arcmoonstudios/PyCrust.git
cd PyCrust

# PHASE 2: Framework Orchestration Execution

python pycrust.py
```

The orchestrator performs these critical operations:

1. Creates `./Xyn's Decruster/` in the target project root
2. Sparse-clones core framework components from the private repository
3. Deploys the operational `pycrust.py` script to the project root (from `pyDecruster/operational/pycrust.py`)
4. Sets up a dedicated virtual environment with precise dependency resolution

### Framework Utilization

After initialization, all framework operations are performed through the root `pycrust.py`:

```bash
cd /path/to/your/rust_project
python pycrust.py --scan          # Execute comprehensive project scan
python pycrust.py --cli           # Initialize interactive analysis console
python pycrust.py --force-setup   # Force environment reconstruction
python pycrust.py --help          # View complete command reference
```

## Technical Architecture

```markdown
TARGET_PROJECT_STRUCTURE:
│
├── [Your Rust Project Files]
│
├── pycrust.py                     # Operational interface (auto-deployed)
│
└── Xyn's Decruster/               # Framework environment
    ├── pyDecruster/               # Core engine (sparse-cloned)
    │   ├── pydecruster/           # Implementation modules
    │   ├── README.md              # Engine documentation
    │   └── requirements.txt       # Dependency specifications
    │
    ├── PYDECRUSTER_ENGINE_README.md  # Copied documentation
    └── pycrust_orchestrator.log   # Operation logs
```

## Deployment Verification

The deployment is successfully completed when:

1. `Xyn's Decruster/` directory exists in project root
2. `pycrust.py` operational script is present in project root
3. Framework initialization completes with success message:

```text
======================================================================
Xyn's Decruster framework Installation is now Complete!
======================================================================
```

## Implementation Troubleshooting

```python
ERROR_RESOLUTION_MATRIX = {
    "git_not_found": {
        "detection": "CommandError: Command 'git' not found",
        "resolution": "Install Git and ensure it's available in system PATH"
    },
    "python_environment_issues": {
        "detection": "venv module not available",
        "resolution": "Install Python with venv support (standard in Python 3.8+)"
    },
    "framework_initialization_failure": {
        "detection": "Failed to set up PyDecruster engine",
        "resolution": "Run with --verbose flag and check logs at ./Xyn's Decruster/pycrust_orchestrator.log"
    }
}
```

## License Configuration

The PyCrust orchestrator (`pycrust.py`) is distributed under MIT license, with independent licensing for the core PyDecruster engine defined within its repository.
