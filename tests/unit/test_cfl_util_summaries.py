"""Unit tests for the formatting / summary helpers in ``pycf.cfl_util``.

These functions are pure-Python text-formatting utilities that take numerical
inputs (eigenvalues, eigenvectors, fitted coefficients, fake ExData objects)
and produce human-readable summary strings.  They were previously untested
beyond a handful of cases hidden inside integration tests.

The tests here probe expected behaviour rather than just shape:

* ``ex_parse_abs`` / ``ex_parse_diff``: feed minimal duck-typed ExData stand-ins
  with hand-computed ``parsed_ex`` arrays and assert the returned arrays match
  exactly.  Both ``sl_index=False`` (index-based) and ``sl_index=True``
  (state-label) code paths are exercised, plus error paths.
* ``gen_e_summary`` / ``gen_e_summary_trunc`` / ``gen_sh_summary`` /
  ``gen_fit_summary``: build small synthetic inputs and assert the returned
  strings contain the expected headings, the right number of data rows, the
  numeric values that were passed in, and that derived quantities such as the
  ``sigma`` field are computed correctly.
* ``print_as_fortran_array`` / ``print_as_c_array``: capture stdout via
  ``capsys`` and assert structural and numeric correctness.
* The pycf-banner helpers (``gen_pycf_summary``, ``print_pycf_details``,
  ``print_completed_str``) are smoke-tested.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest

from pycf import cfl_util

# ---------------------------------------------------------------------------
# ex_parse_abs / ex_parse_diff
# ---------------------------------------------------------------------------


def _make_index_exdata(*, n_a, n_d, e, la=None, ild=None, fld=None):
    """Build a minimal duck-typed ExData stand-in for ``sl_index = False``."""
    return SimpleNamespace(
        n_a=n_a,
        n_d=n_d,
        e=np.asarray(e, dtype=float),
        la=np.asarray(la if la is not None else [], dtype=float),
        ild=np.asarray(ild if ild is not None else [], dtype=float),
        fld=np.asarray(fld if fld is not None else [], dtype=float),
        sl_index=False,
        a_states=[],
        id_states=[],
        fd_states=[],
    )


def _make_label_exdata(*, n_a, n_d, e, a_states=None, id_states=None, fd_states=None):
    """Build a minimal duck-typed ExData stand-in for ``sl_index = True``."""
    return SimpleNamespace(
        n_a=n_a,
        n_d=n_d,
        e=np.asarray(e, dtype=float),
        la=np.array([], dtype=float),
        ild=np.array([], dtype=float),
        fld=np.array([], dtype=float),
        sl_index=True,
        a_states=a_states if a_states is not None else [],
        id_states=id_states if id_states is not None else [],
        fd_states=fd_states if fd_states is not None else [],
    )


class TestExParseAbs:
    def test_empty(self):
        ex = _make_index_exdata(n_a=0, n_d=0, e=[])
        out = cfl_util.ex_parse_abs(ex, np.eye(3), [(0, 0)] * 3)
        assert out.size == 0

    def test_index_mode_unsorted_input_is_sorted(self):
        # Indices are 1-based on the user side.  ex_parse_abs sorts by the
        # index column.
        ex = _make_index_exdata(n_a=3, n_d=0, e=[300.0, 100.0, 200.0], la=[3, 1, 2])
        out = cfl_util.ex_parse_abs(ex, np.eye(4), [(0, 0)] * 4)
        # After sort: indices 1, 2, 3 with energies 100, 200, 300.
        np.testing.assert_array_equal(out[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(out[:, 1], [100.0, 200.0, 300.0])

    def test_index_mode_with_diff_part_uses_only_n_a(self):
        # ``ex.e`` typically holds abs values first, then diff values.
        # ex_parse_abs must read only the first n_a entries.
        ex = _make_index_exdata(n_a=2, n_d=2, e=[10.0, 20.0, 999.0, 999.0], la=[1, 2])
        out = cfl_util.ex_parse_abs(ex, np.eye(3), [(0, 0)] * 3)
        np.testing.assert_array_equal(out[:, 1], [10.0, 20.0])

    def test_label_mode_picks_principal_component(self):
        # Two eigenvectors:
        #   col 0: dominant on basis state 1 (label (1,))
        #   col 1: dominant on basis state 0 (label (0,))
        z = np.array([[0.1, 0.99], [0.99, 0.1]])
        labels = [(0,), (1,)]
        # Experimental absolute level with state label (1,) and energy 50.
        ex = _make_label_exdata(n_a=1, n_d=0, e=[50.0], a_states=[(1,)])
        out = cfl_util.ex_parse_abs(ex, z, labels)
        # The principal component of column 0 has label (1,); so index 0.
        np.testing.assert_array_equal(out[:, 0], [0])
        np.testing.assert_array_equal(out[:, 1], [50.0])

    def test_label_mode_missing_label_raises(self):
        z = np.eye(2)
        labels = [(0,), (1,)]
        ex = _make_label_exdata(n_a=1, n_d=0, e=[42.0], a_states=[(7,)])
        with pytest.raises(RuntimeError, match="not found"):
            cfl_util.ex_parse_abs(ex, z, labels)


class TestExParseDiff:
    def test_empty(self):
        ex = _make_index_exdata(n_a=0, n_d=0, e=[])
        out = cfl_util.ex_parse_diff(ex, np.eye(3), [(0, 0)] * 3)
        assert out.size == 0

    def test_index_mode_uses_lexsort_initial_then_final(self):
        # ex.e: n_a=2 abs values then n_d=3 diff values.
        ex = _make_index_exdata(
            n_a=2,
            n_d=3,
            e=[10.0, 20.0, 100.0, 200.0, 50.0],
            ild=[3, 1, 1],
            fld=[5, 3, 2],
        )
        out = cfl_util.ex_parse_diff(ex, np.eye(6), [(0, 0)] * 6)
        # After lexsort (primary key: ild = col 0):
        #   ild=1, fld=2, e=50.0
        #   ild=1, fld=3, e=200.0
        #   ild=3, fld=5, e=100.0
        np.testing.assert_array_equal(out[:, 0], [1, 1, 3])
        np.testing.assert_array_equal(out[:, 1], [2, 3, 5])
        np.testing.assert_array_equal(out[:, 2], [50.0, 200.0, 100.0])

    def test_label_mode_two_diffs(self):
        z = np.eye(3)
        labels = [(0,), (1,), (2,)]
        ex = _make_label_exdata(
            n_a=0,
            n_d=2,
            e=[33.0, 77.0],
            id_states=[(0,), (2,)],
            fd_states=[(1,), (0,)],
        )
        out = cfl_util.ex_parse_diff(ex, z, labels)
        np.testing.assert_array_equal(out[:, 0], [0, 2])
        np.testing.assert_array_equal(out[:, 1], [1, 0])
        np.testing.assert_array_equal(out[:, 2], [33.0, 77.0])

    def test_label_mode_missing_initial_label_raises(self):
        z = np.eye(2)
        labels = [(0,), (1,)]
        ex = _make_label_exdata(n_a=0, n_d=1, e=[10.0], id_states=[(99,)], fd_states=[(0,)])
        with pytest.raises(RuntimeError, match="Initial-state label"):
            cfl_util.ex_parse_diff(ex, z, labels)

    def test_label_mode_missing_final_label_raises(self):
        z = np.eye(2)
        labels = [(0,), (1,)]
        ex = _make_label_exdata(n_a=0, n_d=1, e=[10.0], id_states=[(0,)], fd_states=[(99,)])
        with pytest.raises(RuntimeError, match="Final-state label"):
            cfl_util.ex_parse_diff(ex, z, labels)


# ---------------------------------------------------------------------------
# gen_e_summary
# ---------------------------------------------------------------------------


def _diag_eigenpair(eigenvalues):
    """Return (w, z) for a diagonal, identity-eigenvector model."""
    n = len(eigenvalues)
    return np.asarray(eigenvalues, dtype=float), np.eye(n)


class TestGenESummary:
    def test_no_ex_contains_levels_and_label_key(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0])
        labels = [(0, 0), (0, 1), (0, 2)]
        s = cfl_util.gen_e_summary(w, z, labels, "JM")
        assert "Energy level summary" in s
        assert "Label key: JM" in s
        # Each level number must appear in the leftmost column.
        for level in (1, 2, 3):
            assert "\n{0:<6}".format(level) in s or s.startswith("{0:<6}".format(level))
        # Theory eigenvalues must be present.
        for energy in ("0.0000", "100.0000", "200.0000"):
            assert energy in s

    def test_with_ex_array_shows_residuals(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0])
        labels = [(0, 0), (0, 1), (0, 2)]
        ex = np.array([[1, 0.0], [3, 198.0]], dtype=float)
        s = cfl_util.gen_e_summary(w, z, labels, "JM", ex=ex)
        assert "Experiment" in s
        # Difference for level 3: 198 - 200 = -2.
        assert "-2.0000" in s
        # Levels with no experimental data must show the placeholder.
        assert "--" in s

    def test_e_shift_zeroes_minimum(self):
        w, z = _diag_eigenpair([100.0, 200.0])
        labels = [(0, 0), (0, 1)]
        s = cfl_util.gen_e_summary(w, z, labels, "JM", e_shift=True)
        # The minimum (100) should now print as 0.0000, and the shift line
        # should announce the magnitude.
        assert "0.0000" in s
        assert "Energy level shift: -100.0000" in s

    def test_chi2_with_ndof_computes_sigma(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        s = cfl_util.gen_e_summary(w, z, labels, "JM", chi2=4.0, ndof=4, weighting=1.0)
        # sigma = sqrt(4 / (1 * 4)) = 1.0
        assert "sigma = 1.0000" in s
        assert "weighted chi2 = 4.0000" in s

    def test_chi2_with_ndof_zero_shows_NA(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        s = cfl_util.gen_e_summary(w, z, labels, "JM", chi2=4.0, ndof=0, weighting=1.0)
        assert "sigma = N/A" in s

    def test_chi2_with_ndof_missing_weighting_raises(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        with pytest.raises(ValueError, match="weight argument"):
            cfl_util.gen_e_summary(w, z, labels, "JM", chi2=1.0, ndof=1)

    def test_duplicate_ex_indices_raise(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        ex = np.array([[1, 0.0], [1, 50.0]], dtype=float)
        with pytest.raises(ValueError, match="duplicate entries"):
            cfl_util.gen_e_summary(w, z, labels, "JM", ex=ex)

    def test_multiplet_stats_no_ex_data_prints_barycenter_only(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0, 300.0])
        labels = [(0, i) for i in range(4)]
        s = cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=[2, 4])
        assert "Multiplet" in s
        assert "barycenter =" in s
        assert "sigma_total =" not in s
        assert "sigma_crystal_field =" not in s

    def test_multiplet_stats_with_ex_data_and_e_shift(self):
        w, z = _diag_eigenpair([100.0, 200.0, 300.0, 400.0])
        labels = [(0, i) for i in range(4)]
        ex = np.array([[1, 1.0], [2, 98.0], [3, 205.0], [4, 305.0]], dtype=float)
        s = cfl_util.gen_e_summary(
            w,
            z,
            labels,
            "JM",
            ex=ex,
            e_shift=True,
            multiplet_end_levels=[2, 4],
        )
        assert "Multiplet" in s
        assert "barycenter =" in s
        assert "shift =" in s
        assert "sigma_total =" in s
        assert "sigma_crystal_field =" in s

    def test_multiplet_stats_respects_max_levels(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0, 300.0])
        labels = [(0, i) for i in range(4)]
        s = cfl_util.gen_e_summary(
            w,
            z,
            labels,
            "JM",
            ex=np.array([[1, 0.0], [2, 101.0]]),
            max_levels=2,
            multiplet_end_levels=[2, 4],
        )
        # Only the first multiplet boundary (2) is displayed when max_levels=2.
        assert s.count("Multiplet") == 1
        assert "  1-  2]" in s

    def test_multiplet_end_levels_validation(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0])
        labels = [(0, i) for i in range(3)]
        with pytest.raises(ValueError, match="strictly increasing"):
            cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=[2, 2])
        with pytest.raises(ValueError, match=">= 1"):
            cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=[0, 2])
        with pytest.raises(ValueError, match="<= number of levels"):
            cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=[4])
        with pytest.raises(TypeError, match="sequence"):
            cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=3)
        with pytest.raises(TypeError, match="integer values"):
            cfl_util.gen_e_summary(w, z, labels, "JM", multiplet_end_levels=[2.5])


# ---------------------------------------------------------------------------
# gen_e_summary_trunc
# ---------------------------------------------------------------------------


class TestGenESummaryTrunc:
    def test_empty_returns_empty_string(self):
        ex = _make_index_exdata(n_a=0, n_d=0, e=[])
        w, z = _diag_eigenpair([0.0, 100.0])
        out = cfl_util.gen_e_summary_trunc(w, z, [(0, 0), (0, 1)], "JM", ex, "Test")
        assert out == ""

    def test_abs_only(self):
        w, z = _diag_eigenpair([0.0, 50.0, 100.0, 150.0])
        labels = [(0, i) for i in range(4)]
        ex = _make_index_exdata(n_a=2, n_d=0, e=[2.0, 105.0], la=[1, 3])
        s = cfl_util.gen_e_summary_trunc(w, z, labels, "JM", ex, "TruncAbs")
        assert "TruncAbs summary" in s
        # Only levels 1 and 3 should appear; levels 2 and 4 should not.
        assert "Label key: JM" in s
        # The two experimental energies are exactly 2.0 and 105.0.
        assert "2.0000" in s
        assert "105.0000" in s
        # Difference for level 1: 2 - 0 = 2; for level 3: 105 - 100 = 5.
        assert "5.0000" in s

    def test_diff_only(self):
        w, z = _diag_eigenpair([0.0, 50.0, 100.0])
        labels = [(0, i) for i in range(3)]
        ex = _make_index_exdata(n_a=0, n_d=1, e=[48.0], ild=[1], fld=[2])
        s = cfl_util.gen_e_summary_trunc(w, z, labels, "JM", ex, "TruncDiff")
        assert "TruncDiff summary" in s
        assert "48" in s
        # The theoretical difference (level 2 - level 1) is 50, residual = -2.
        assert "-2" in s

    def test_chi2_with_ndof_zero_shows_NA(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        ex = _make_index_exdata(n_a=1, n_d=0, e=[1.0], la=[1])
        s = cfl_util.gen_e_summary_trunc(
            w,
            z,
            labels,
            "JM",
            ex,
            "Tr",
            chi2=2.0,
            ndof=0,
            weighting=1.0,
        )
        assert "sigma = N/A" in s

    def test_chi2_with_ndof_missing_weighting_raises(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        ex = _make_index_exdata(n_a=1, n_d=0, e=[0.0], la=[1])
        with pytest.raises(ValueError, match="weight argument"):
            cfl_util.gen_e_summary_trunc(w, z, labels, "JM", ex, "Tr", chi2=1.0, ndof=1)


# ---------------------------------------------------------------------------
# gen_sh_summary
# ---------------------------------------------------------------------------


class _FakeSpinH:
    """Minimal stand-in for the SpinH object expected by ``gen_sh_summary``."""

    def __init__(self, interactions):
        self.interactions = list(interactions)


class TestGenShSummary:
    def test_default_heading(self):
        sh = _FakeSpinH(["zeeman"])
        param = [np.diag([2.0, 2.0, 2.0])]
        s = cfl_util.gen_sh_summary(param, sh)
        assert "Spin Hamiltonian summary" in s
        assert "zeeman interaction" in s
        assert "Theory (abs. value)" in s

    def test_custom_name(self):
        sh = _FakeSpinH(["hyperfine"])
        param = [np.diag([100.0, 100.0, 200.0])]
        s = cfl_util.gen_sh_summary(param, sh, name="Site 1")
        assert "Site 1 summary" in s
        assert "hyperfine interaction" in s

    def test_with_experimental_shx(self):
        sh = _FakeSpinH(["zeeman"])
        param = [np.diag([2.0, 2.0, 2.0])]
        shx = {"zeeman": np.diag([2.5, 2.5, 2.5])}
        s = cfl_util.gen_sh_summary(param, sh, shx=shx)
        assert "Experiment (abs. value)" in s
        assert "Difference" in s

    def test_chi2_with_ndof_computes_sigma(self):
        sh = _FakeSpinH(["zeeman"])
        param = [np.diag([2.0, 2.0, 2.0])]
        # sigma = sqrt((4/2) / 2) = sqrt(1) = 1.0
        s = cfl_util.gen_sh_summary(
            param, sh, chi2=np.array([4.0]), ndof=2, weighting={"zeeman": 2.0}
        )
        assert "sigma = 1.0000" in s

    def test_chi2_with_ndof_zero_shows_NA(self):
        sh = _FakeSpinH(["zeeman"])
        param = [np.diag([2.0, 2.0, 2.0])]
        s = cfl_util.gen_sh_summary(
            param, sh, chi2=np.array([4.0]), ndof=0, weighting={"zeeman": 1.0}
        )
        assert "sigma = N/A" in s

    def test_chi2_with_ndof_missing_weighting_raises(self):
        sh = _FakeSpinH(["zeeman"])
        param = [np.diag([2.0, 2.0, 2.0])]
        with pytest.raises(ValueError, match="weight argument"):
            cfl_util.gen_sh_summary(param, sh, chi2=np.array([1.0]), ndof=1)


# ---------------------------------------------------------------------------
# gen_fit_summary
# ---------------------------------------------------------------------------


class _FakeFitObj:
    """Minimal iterable stand-in for *FitRunner objects."""

    def __init__(self, coeff):
        self.coeff = coeff

    def __iter__(self):
        return iter(self.coeff)


class TestGenFitSummary:
    def test_basic_output(self):
        coeff = {"EAVG": 1000.0, "C20": 300.0}
        fit_obj = _FakeFitObj({"EAVG": 1010.0 + 0j, "C20": 290.0 + 0j})
        s = cfl_util.gen_fit_summary(
            coeff,
            fit_obj,
            method="nlopt_bobyqa",
            fmin=1.5,
            n_obs=10,
            n_param=2,
        )
        assert "Fitting summary" in s
        assert "EAVG" in s
        assert "C20" in s
        assert "Number of observables: 10" in s
        assert "Number of real-valued parameters: 2" in s
        assert "method:" in s
        assert "nlopt_bobyqa" in s

    def test_complex_param_increments_cov_index_by_two(self):
        # The function distinguishes real vs complex coefficients when stepping
        # through the covariance diagonal.  An off-by-one would corrupt the
        # uncertainty for the second parameter.
        coeff = {"C44": -1000.0 + 500j, "C20": 300.0}
        fit_obj = _FakeFitObj({"C44": -1000.0 + 500j, "C20": 300.0 + 0j})
        # diagonal: var(re C44)=4, var(im C44)=9, var(C20)=16
        cov = np.diag([4.0, 9.0, 16.0])
        s = cfl_util.gen_fit_summary(
            coeff,
            fit_obj,
            method="nlopt_bobyqa",
            fmin=0.0,
            n_obs=5,
            n_param=3,
            covar=cov,
        )
        # The uncertainty on C20 should be sqrt(16) = 4, not sqrt(9) = 3.
        # The complex C44 uncertainty must be reported as a complex with
        # magnitudes sqrt(4)=2 and sqrt(9)=3.
        assert "Covariance matrix" in s
        # Complex C44 std-dev is reported as "2+3j"; C20 std-dev is "4".
        assert "2+3j" in s
        assert " 4\n" in s or " 4 " in s

    def test_basinhopping_renames_retval(self):
        coeff = {"EAVG": 1.0}
        fit_obj = _FakeFitObj({"EAVG": 1.0 + 0j})
        s = cfl_util.gen_fit_summary(
            coeff,
            fit_obj,
            method="basinhopping",
            fmin=0.0,
            n_obs=1,
            n_param=1,
            retval=42,
        )
        assert "naccept:" in s
        assert "42" in s
        assert "retval" not in s


# ---------------------------------------------------------------------------
# gen_all_coeff_summary
# ---------------------------------------------------------------------------


class TestGenAllCoeffSummary:
    def test_parameter_order_matches_domain_priority(self):
        all_coeff = {
            "ZZ": 1.0,
            "Q2": 1.0,
            "A2": 1.0,
            "M2": 1.0,
            "MY": 1.0,
            "C44": 1.0,
            "C20": 1.0,
            "PTOT": 1.0,
            "MTOT": 1.0,
            "ZETA": 1.0,
            "T8": 1.0,
            "T2": 1.0,
            "GAMMA": 1.0,
            "BETA": 1.0,
            "ALPHA": 1.0,
            "F9": 1.0,
            "FTOT": 1.0,
            "F6": 1.0,
            "F4": 1.0,
            "F2": 1.0,
            "EAVG": 1.0,
            "MX": 1.0,
            "MZ": 1.0,
            "A": 1.0,
            "Q": 1.0,
            "C2": 1.0,
            "c4": 1.0,
            "C64": 1.0,
            "MABC": 1.0,
            "X10": 1.0,
            "X2": 1.0,
        }
        s = cfl_util.gen_all_coeff_summary(all_coeff)

        lines = s.splitlines()
        dash_idx = next(i for i, line in enumerate(lines) if set(line) == {"-"})
        params = [line.split()[0] for line in lines[dash_idx + 1 :] if line.strip()]

        assert params == [
            "EAVG",
            "F2",
            "F4",
            "F6",
            "FTOT",
            "F9",
            "ALPHA",
            "BETA",
            "GAMMA",
            "T2",
            "T8",
            "ZETA",
            "MTOT",
            "PTOT",
            "C2",
            "c4",
            "C20",
            "C44",
            "C64",
            "MX",
            "MY",
            "MZ",
            "M2",
            "MABC",
            "A",
            "A2",
            "Q",
            "Q2",
            "X2",
            "X10",
            "ZZ",
        ]


# ---------------------------------------------------------------------------
# print_as_fortran_array / print_as_c_array
# ---------------------------------------------------------------------------


class TestPrintAsArray:
    def test_fortran_real_only(self, capsys):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        cfl_util.print_as_fortran_array(a)
        out = capsys.readouterr().out.strip()
        # Fortran/column-major: outer braces only, comma-separated.
        assert out.startswith("{")
        assert out.endswith("};")
        assert "1.0" in out and "4.0" in out
        # 4 values implies exactly 3 commas inside the braces.
        assert out.count(",") == 3

    def test_fortran_with_imaginary(self, capsys):
        a = np.array([[1 + 2j, 3 - 4j]])
        cfl_util.print_as_fortran_array(a)
        out = capsys.readouterr().out
        assert "1.0+2.0*I" in out
        assert "3.0-4.0*I" in out

    def test_c_real_only(self, capsys):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        cfl_util.print_as_c_array(a)
        out = capsys.readouterr().out.strip()
        # C/row-major: nested braces.
        assert "{1.0,2.0}" in out
        assert "{3.0,4.0}" in out
        assert out.endswith("};")

    def test_c_zero_real_part_normalised(self, capsys):
        a = np.array([[0.0 + 1j]])
        cfl_util.print_as_c_array(a)
        out = capsys.readouterr().out
        # The zero real part is normalised to integer 0.
        assert "0+1.0*I" in out


# ---------------------------------------------------------------------------
# Banner / print helpers (smoke tests)
# ---------------------------------------------------------------------------


class TestBannerHelpers:
    def test_gen_pycf_summary_includes_input_file_section(self, tmp_path):
        # gen_pycf_summary opens the caller's source file via inspect.stack().
        # When called from within a pytest test, the caller is this very test
        # file, so the function will succeed and the returned text should
        # include both the input-file header and pycf details.
        s = cfl_util.gen_pycf_summary()
        assert "Input file" in s
        assert "pycf details" in s
        assert "pycf revision" in s

    def test_print_pycf_details_writes_to_stdout(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfl_util.print_pycf_details()
        out = buf.getvalue()
        assert "pycf revision" in out
        assert "Calculation started at:" in out

    def test_print_completed_str_writes_to_stdout(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfl_util.print_completed_str()
        out = buf.getvalue()
        assert "Calculation completed at:" in out


# ---------------------------------------------------------------------------
# Error-path coverage for shared summary helpers
# ---------------------------------------------------------------------------


class TestSummaryErrorPaths:
    """Cover defensive raises that have historically been untested."""

    def test_gen_e_summary_rejects_max_levels_zero(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        with pytest.raises(ValueError, match="max_levels must be >= 1"):
            cfl_util.gen_e_summary(w, z, labels, "JM", max_levels=0)

    def test_gen_e_summary_rejects_max_levels_negative(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        with pytest.raises(ValueError, match="max_levels must be >= 1"):
            cfl_util.gen_e_summary(w, z, labels, "JM", max_levels=-3)

    def test_gen_e_summary_duplicate_ex_indices_raises(self):
        w, z = _diag_eigenpair([0.0, 100.0, 200.0])
        labels = [(0, i) for i in range(3)]
        # Duplicate level index 1 — should be flagged before any formatting.
        ex = _make_index_exdata(
            n_a=2, n_d=0, e=[0.0, 0.0, 999.0, 999.0], la=[1, 1]
        )
        with pytest.raises(ValueError, match="duplicate entries"):
            cfl_util.gen_e_summary(w, z, labels, "JM", ex=ex)

    def test_gen_e_summary_ndof_without_weighting_raises(self):
        """``_format_summary_footer`` requires ``weighting`` when ndof is given."""
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        ex = _make_index_exdata(n_a=1, n_d=0, e=[0.0, 999.0], la=[1])
        with pytest.raises(ValueError, match="weight argument needs to be provided"):
            cfl_util.gen_e_summary(w, z, labels, "JM", ex=ex, chi2=1.0, ndof=1)

    def test_gen_e_summary_trunc_ndof_without_weighting_raises(self):
        w, z = _diag_eigenpair([0.0, 100.0])
        labels = [(0, 0), (0, 1)]
        ex = _make_index_exdata(n_a=1, n_d=0, e=[0.0, 999.0], la=[1])
        with pytest.raises(ValueError, match="weight argument needs to be provided"):
            cfl_util.gen_e_summary_trunc(
                w, z, labels, "JM", ex, "Test", chi2=1.0, ndof=1
            )
