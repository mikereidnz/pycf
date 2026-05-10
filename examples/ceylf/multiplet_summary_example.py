#!/usr/bin/env python3
"""
Example: multiplet-aware energy summary output.

This example shows how to:
1. Build and diagonalize a Ce:YLF Hamiltonian
2. Attach absolute experimental energies via ExData
3. Define user multiplet boundaries with 1-based end levels
4. Print `gen_summary()` output including per-multiplet diagnostics

Notes
-----
- Multiplet boundaries are inclusive end levels, e.g. [3, 8, 14] means:
  1-3, 4-8, and 9-14.
- Multiplet diagnostics use absolute-energy assignments only.
- If `e_shift=True`, both level rows and multiplet diagnostics use shifted levels.
"""

from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM


def main() -> None:
    # Load matrix elements from this example directory.
    t = ImportSLJM(str(Path(__file__).parent / "matel" / "f1cf"))

    # Ce:YLF starting coefficients (same baseline as other Ce:YLF examples).
    coeff = {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
    }

    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64], label="Ce:YLF")
    h.set_coeff(coeff)
    h.diag()

    # Absolute experimental energies: (level_index, energy_cm^-1), 1-based levels.
    ex_abs = np.array(
        [
            [2, 0.0],
            [3, 216.0],
            [8, 2216.0],
            [9, 2312.8],
            [12, 2428.8],
            [14, 3157.8],
        ],
        dtype=float,
    )
    exdata = cfl.ExData(ex_abs, "A")

    # Multiplet boundaries (1-based inclusive end levels).
    h.set_multiplet_end_levels([3, 9, 14])

    print(
        h.gen_summary(
            ex=exdata,
            e_shift=False,
            max_levels=20,
            nstates=2,
        )
    )


if __name__ == "__main__":
    main()
