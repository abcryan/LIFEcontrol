"""Unit tests for quaternion ↔ CTM conversions.

Run with:  pytest test_quaternion.py -v
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

# Add the project root to sys.path so 'utils.coordinate_trafos' resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # ../../ from this file
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

# Adjust the import to wherever your functions actually live, e.g.:
# from lifecontrol.core.attitude import Quat_to_CTM, CTM_to_Quat
from utils.coordinate_trafos.CTM_to_Quat import CTM_to_Quat
from utils.coordinate_trafos.Quat_to_CTM import Quat_to_CTM


# ── helpers ──────────────────────────────────────────────────────────────────
def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q       # canonicalize sign (q and -q are equal)


# ── Test 1: Quat → CTM → Quat ────────────────────────────────────────────────
def test_quat_ctm_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q  = _random_unit_quat(rng)
        q2 = CTM_to_Quat(Quat_to_CTM(q))
        if q2[0] < 0:
            q2 = -q2
        assert np.linalg.norm(q - q2) < 1e-12


# ── Test 2: CTM → Quat → CTM ─────────────────────────────────────────────────
def test_ctm_quat_ctm_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q  = _random_unit_quat(rng)
        C  = Quat_to_CTM(q)              # start from a guaranteed-valid rotation
        C2 = Quat_to_CTM(CTM_to_Quat(C))
        assert np.max(np.abs(C - C2)) < 1e-12


# ── Test 3: known case — 90° rotation about z-axis ───────────────────────────
def test_known_rotation_z90():
    s = np.sqrt(2) / 2
    q = np.array([s, 0.0, 0.0, s])       # 90° about +z, scalar-first
    C = Quat_to_CTM(q)
    C_expected = np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])
    assert np.max(np.abs(C - C_expected)) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])