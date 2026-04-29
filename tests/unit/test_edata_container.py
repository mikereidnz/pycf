"""Tests for the EData container in pycf.cfl_util."""

from __future__ import annotations

import numpy as np
import pytest

from pycf.cfl_util import EData, gen_edata_summary


def _make_arr(n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=EData.DTYPE)
    for i in range(n):
        arr[i]["h_index"] = i % 2
        arr[i]["h_label"] = f"H{i % 2}"
        arr[i]["kind"] = "A" if i % 2 == 0 else "D"
        arr[i]["i_lo"] = i + 1
        arr[i]["i_hi"] = i + 2 if i % 2 else 0
        arr[i]["e_calc"] = float(i)
        arr[i]["e_obs"] = float(i) + 0.5
        arr[i]["weight"] = 1.0 + 0.25 * i
        arr[i]["residual"] = arr[i]["e_calc"] - arr[i]["e_obs"]
        arr[i]["wresidual"] = np.sqrt(arr[i]["weight"]) * arr[i]["residual"]
    return arr


def test_dtype_field_order():
    expected = (
        "h_index",
        "h_label",
        "kind",
        "i_lo",
        "i_hi",
        "e_calc",
        "e_obs",
        "weight",
        "residual",
        "wresidual",
    )
    assert EData.DTYPE.names == expected


def test_empty_constructor_zero_rows():
    e = EData.empty(0)
    assert len(e) == 0
    assert e.chi2() == 0.0
    assert "empty" in repr(EData.empty(0)) or "n=0" in repr(EData.empty(0))


def test_empty_constructor_n_rows():
    e = EData.empty(5)
    assert len(e) == 5
    assert e.arr.shape == (5,)
    # All rows zero-initialised.
    assert np.all(e.arr["e_calc"] == 0.0)


def test_empty_negative_raises():
    with pytest.raises(ValueError):
        EData.empty(-1)


def test_constructor_rejects_wrong_dtype():
    bad = np.zeros(3, dtype=np.float64)
    with pytest.raises(TypeError):
        EData(bad)


def test_constructor_rejects_2d():
    arr = np.zeros((2, 3), dtype=EData.DTYPE)
    with pytest.raises(ValueError):
        EData(arr)


def test_constructor_rejects_non_ndarray():
    with pytest.raises(TypeError):
        EData([1, 2, 3])  # type: ignore[arg-type]


def test_chi2_matches_manual_sum():
    arr = _make_arr(4)
    e = EData(arr)
    expected = float(np.sum(arr["weight"] * arr["residual"] ** 2))
    assert e.chi2() == pytest.approx(expected)


def test_chi2_zero_when_perfect_fit():
    arr = _make_arr(3)
    arr["e_obs"] = arr["e_calc"]
    arr["residual"] = 0.0
    arr["wresidual"] = 0.0
    e = EData(arr)
    assert e.chi2() == 0.0


def test_getitem_returns_row():
    arr = _make_arr(2)
    e = EData(arr)
    assert int(e[0]["i_lo"]) == 1
    assert str(e[1]["kind"]) == "D"


def test_to_str_contains_header_and_chi2():
    e = EData(_make_arr(2))
    s = e.to_str()
    assert "kind" in s and "e_calc" in s
    assert "chi2" in s
    assert "N = 2" in s


def test_to_str_empty():
    s = EData.empty(0).to_str()
    assert s == "EData (empty)"


def test_to_str_max_rows_truncates():
    e = EData(_make_arr(10))
    s = e.to_str(max_rows=3)
    assert "more rows" in s
    # Three data rows + header(1) + sep(1) + truncation(1) + sep(1) + chi2(1)
    # = 8 lines minimum.
    assert len(s.splitlines()) >= 7
    # And full version is longer.
    assert len(e.to_str().splitlines()) > len(s.splitlines())


def test_repr_includes_size_and_chi2():
    e = EData(_make_arr(3))
    r = repr(e)
    assert "n=3" in r
    assert "chi2" in r


def test_gen_edata_summary_matches_to_str():
    e = EData(_make_arr(4))
    assert gen_edata_summary(e) == e.to_str()
    assert gen_edata_summary(e, precision=2) == e.to_str(precision=2)


def test_gen_edata_summary_type_error():
    with pytest.raises(TypeError):
        gen_edata_summary([])  # type: ignore[arg-type]
