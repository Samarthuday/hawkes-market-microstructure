import numpy as np
import pandas as pd

from poisson_diagnostics import calculate_autocorrelation, calculate_fano_factor


def test_autocorrelation_near_zero_for_iid_counts():
    """
    Regression test for a real bug: calculate_autocorrelation used
    to multiply two pandas Series sliced from an IntervalIndex-
    indexed series directly, which aligns by index LABEL rather than
    by position. On i.i.d. data (true autocorrelation ~0 at every
    lag), the old code reported ~0.998 at every lag regardless of
    the input. The fix converts to a plain NumPy array first.
    """

    rng = np.random.default_rng(0)
    values = rng.poisson(lam=5, size=2000).astype(float)

    edges = np.arange(0, len(values) + 1) * 10
    series = pd.Series(
        values,
        index=pd.IntervalIndex.from_arrays(
            edges[:-1], edges[1:], closed="left"
        )
    )

    for lag in (1, 2, 5, 10):
        acf = calculate_autocorrelation(series, lag)
        assert abs(acf) < 0.15, (
            f"lag {lag} autocorrelation {acf} is suspiciously large "
            "for i.i.d. data -- the IntervalIndex alignment bug may "
            "have regressed."
        )


def test_fano_factor_excludes_partial_trailing_window():
    """
    Regression test for a real bug: calculate_fano_factor computed
    `num_windows` as the number of *complete* windows, but
    np.bincount's minlength only sets a floor on the output length,
    not a ceiling -- an incomplete trailing window used to be
    silently included as an extra, short-exposure bin. The fix drops
    events falling in that trailing partial window before counting.
    """

    # Deterministic counts and positions, so the window boundaries
    # are exactly under our control -- calculate_fano_factor measures
    # elapsed time from event_times.min(), so the first event must
    # sit at exactly t=0 for windows to line up with n_per_window.
    n_per_window = [3, 5, 2, 4, 6, 1, 3, 5, 2, 4]

    times = []
    for k, count in enumerate(n_per_window):
        times.extend(10 * k + np.linspace(0, 9.9, count))
    times[0] = 0.0

    # A few extra events in an incomplete trailing window [100, 110).
    times.extend([100.5, 101.2, 102.9])

    event_times = pd.Series(pd.to_datetime(sorted(times), unit="s"))

    fano = calculate_fano_factor(event_times, window_size=10)

    expected_mean = np.mean(n_per_window)
    expected_variance = np.var(n_per_window)
    expected_fano = expected_variance / expected_mean

    assert abs(fano - expected_fano) < 1e-9, (
        f"got {fano}, expected {expected_fano} -- if this regresses, "
        "the trailing partial window is leaking back into the Fano "
        "factor calculation."
    )
