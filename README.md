# LIFEcontrol

LIFE Mission – High-Fidelity Translational Dynamics (Truth Model)

## Complete Setup (for new repository clones)

When you first clone this repository, follow these steps to get everything working:

### 1. Download SPICE kernels

```bash
python scripts/setup_kernels.py
# or: ./scripts/setup_kernels.py
```

This downloads the three required SPICE kernels (~114 MB total) to `data/spice_kernels/`:
- **naif0012.tls** (leapseconds)
- **de440.bsp** (planetary ephemeris, ~114 MB)
- **gm_de440.tpc** (GM values matched to DE440)

### 2. Install runtime dependencies

```bash
pip install numpy spiceypy scipy
```

**Note:** Without these packages, `python main.py` will fail with import errors.

### 3. Verify setup (optional)

Check that everything is properly configured:

```bash
python scripts/check_setup.py
# or: ./scripts/check_setup.py
```

### 4. Run the mission propagator

```bash
python main.py
```

This loads the kernels and demonstrates a 1-day trajectory propagation from the Sun–Earth L2.

---

## Quick Start

### 1. Download SPICE kernels

Run the downloader directly from the repository root:

```bash
python setup_kernels.py
```

This automatically downloads the three required SPICE kernels to `data/spice_kernels/`:
- **naif0012.tls** (leapseconds)
- **de440.bsp** (planetary ephemeris, ~114 MB)
- **gm_de440.tpc** (GM values matched to DE440)

The script will skip any kernels already present and show progress for large downloads.

### 2. Install runtime dependencies (optional)

If you want to run the mission propagator (`main.py`), install the required scientific packages:

```bash
pip install numpy spiceypy scipy
```

### 3. Run the mission propagator

```bash
python main.py
```

This loads the kernels and demonstrates a 1-day trajectory propagation from the Sun–Earth L2.

---

## What Gets Downloaded/Created

After running the setup steps:

- **`data/spice_kernels/`** directory is created with:
  - `naif0012.tls` (~1 KB) - leapseconds kernel
  - `de440.bsp` (~114 MB) - planetary ephemeris
  - `gm_de440.tpc` (~1 KB) - gravitational parameters

- **Python packages** installed (if step 2 is run):
  - `numpy` - numerical arrays
  - `spiceypy` - SPICE toolkit interface
  - `scipy` - ODE integration

---

## What This Automation Does

The repository automates the SPICE kernel download step:

- `scripts/setup_kernels.py` downloads required kernels from NAIF servers
- `scripts/check_setup.py` verifies that setup is complete
- No Python package install is required for kernel download
- Runtime dependencies (`numpy`, `spiceypy`, `scipy`) must be installed separately
- Users may install `numpy`, `spiceypy`, and `scipy` themselves if they want to run `main.py`.

---

## Manual Kernel Download (if needed)

If you prefer to download manually or the script fails:

1. **naif0012.tls** – https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
2. **de440.bsp** – https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp
3. **gm_de440.tpc** – https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc

Save them to `data/spice_kernels/`

---

## Project Structure

```
LIFEcontrol/
├── main.py                          # Mission propagator (entry point)
├── scripts/
│   ├── setup_kernels.py             # Kernel downloader
│   └── check_setup.py               # Setup verification script
├── pyproject.toml                   # Build configuration
├── README.md                        # This file
├── data/
│   └── spice_kernels/               # Downloaded kernels go here
│       ├── naif0012.tls
│       ├── de440.bsp
│       └── gm_de440.tpc
└── results/                         # Output trajectories
```

---

## Development

This repository does not require a package install for kernel download.

Use the downloader script directly:

```bash
python setup_kernels.py
```
