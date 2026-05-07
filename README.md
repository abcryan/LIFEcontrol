# LIFEcontrol

LIFE Mission – High-Fidelity Translational Dynamics (Truth Model)

## Complete Setup (for new repository clones)

When you first clone this repository, follow these steps to get everything working:

### 1. Download SPICE kernels

```bash
python3 scripts/setup_kernels.py
# or: ./scripts/setup_kernels.py
```

This downloads the three required SPICE kernels (~114 MB total) to `data/spice_kernels/`:
- **naif0012.tls** (leapseconds)
- **de440.bsp** (planetary ephemeris, ~114 MB)
- **gm_de440.tpc** (GM values matched to DE440)

### 2. Install runtime dependencies

```bash
pip install numpy spiceypy scipy
# or: use your preferred package manager (e.g. conda) to install numpy, spiceypy, and scipy
```

**Note:** Without these packages, `python main.py` will fail with import errors.

### 3. Verify setup (optional)

Check that everything is properly configured:

```bash
python3 scripts/check_setup.py
# or: ./scripts/check_setup.py
```

### 4. Run the mission propagator

```bash
python3 main.py
```

This loads the kernels and demonstrates a 1-day trajectory propagation from the Sun–Earth L2.

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

