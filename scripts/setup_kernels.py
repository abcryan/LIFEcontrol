"""
SPICE kernel downloader for LIFE mission.

This script automates downloading the required SPICE kernels from the NAIF servers.
"""

import sys
import urllib.request
from pathlib import Path


# SPICE kernel definitions: (filename, URL)
KERNELS = [
    (
        "naif0012.tls",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    ),
    (
        "de440.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
    ),
    (
        "gm_de440.tpc",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
    ),
]


def get_kernel_dir() -> Path:
    """Get the data/spice_kernels directory, creating it if needed."""
    kernel_dir = Path(__file__).parent.parent / "data" / "spice_kernels"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    return kernel_dir


def download_kernel(filename: str, url: str, target_dir: Path) -> bool:
    """
    Download a single kernel file.

    Returns True if successful or file already exists, False otherwise.
    """
    target_path = target_dir / filename

    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"✓ {filename:<20s} (already present, {size_mb:.1f} MB)")
        return True

    print(f"⬇ Downloading {filename:<15s} from NAIF servers...", end=" ", flush=True)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "lifecontrol/0.1.0"},
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192 * 10

            with open(target_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and filename.endswith(".bsp"):
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(
                            f"\r⬇ Downloading {filename:<15s} {mb_done:.1f}/{mb_total:.1f} MB ({percent:.0f}%)...",
                            end=" ",
                            flush=True,
                        )

        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"✓ ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print("✗")
        print(f"  Error: {e}", file=sys.stderr)
        if target_path.exists():
            target_path.unlink()
        return False


def verify_kernels(kernel_dir: Path) -> bool:
    """Check that all required kernels are present."""
    all_present = True
    for filename, _ in KERNELS:
        if not (kernel_dir / filename).exists():
            print(f"✗ Missing: {filename}", file=sys.stderr)
            all_present = False
    return all_present


def main() -> int:
    """Main entry point for kernel setup."""
    print("\n" + "=" * 60)
    print("LIFE Mission – SPICE Kernel Setup")
    print("=" * 60 + "\n")

    kernel_dir = get_kernel_dir()
    print(f"Kernel directory: {kernel_dir}\n")

    if verify_kernels(kernel_dir):
        print("✓ All required kernels are already downloaded.")
        print("\nSetup complete! You can now run:")
        print("  python main.py\n")
        return 0

    print("Starting kernel download...\n")

    failed = []
    for filename, url in KERNELS:
        if not download_kernel(filename, url, kernel_dir):
            failed.append(filename)

    print()

    if failed:
        print("=" * 60)
        print("✗ Download failed for the following kernels:")
        for f in failed:
            print(f"  - {f}")
        print("=" * 60)
        print("\nPlease check your internet connection and try again:")
        print("  python setup_kernels.py\n")
        return 1

    if verify_kernels(kernel_dir):
        print("=" * 60)
        print("✓ All kernels downloaded successfully!")
        print("=" * 60)
        print("\nSetup complete! You can now run:")
        print("  python main.py\n")
        return 0

    print("=" * 60)
    print("✗ Verification failed: Not all kernels are present.")
    print("=" * 60 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
