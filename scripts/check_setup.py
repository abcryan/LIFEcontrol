#!/usr/bin/env python3
"""
LIFE Mission Setup Checker

This script verifies that all required components are properly set up.
"""

import sys
from pathlib import Path


def check_kernels() -> bool:
    """Check if SPICE kernels are present."""
    kernel_dir = Path("data/spice_kernels")
    kernels = ["naif0012.tls", "de440.bsp", "gm_de440.tpc"]

    if not kernel_dir.exists():
        print("✗ Kernel directory missing: data/spice_kernels/")
        return False

    missing = []
    for kernel in kernels:
        if not (kernel_dir / kernel).exists():
            missing.append(kernel)

    if missing:
        print(f"✗ Missing kernels: {', '.join(missing)}")
        return False

    print("✓ All SPICE kernels present")
    return True


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    packages = ["numpy", "spiceypy", "scipy"]
    missing = []

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"✗ Missing Python packages: {', '.join(missing)}")
        print("  Install with: pip install numpy spiceypy scipy")
        return False

    print("✓ All Python dependencies installed")
    return True


def main() -> int:
    """Main setup check."""
    print("\n" + "="*50)
    print("LIFE Mission – Setup Check")
    print("="*50 + "\n")

    kernels_ok = check_kernels()
    deps_ok = check_dependencies()

    print()

    if kernels_ok and deps_ok:
        print("✓ Setup complete! You can run:")
        print("  python main.py")
        return 0
    else:
        print("✗ Setup incomplete. Run:")
        if not kernels_ok:
            print("  python setup_kernels.py")
        if not deps_ok:
            print("  pip install numpy spiceypy scipy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
