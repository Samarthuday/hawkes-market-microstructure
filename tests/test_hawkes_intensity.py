import numpy as np

from hawkes import (
    calculate_hawkes_intensity_series,
    calculate_hawkes_intensity_series_fast,
)


def test_naive_and_recursive_intensity_match():
    """
    The O(n^2) direct intensity calculation and the O(n) recursive
    calculation must agree exactly (up to floating point): they are
    two implementations of the same mathematical quantity.
    """

    event_times = np.array([0.5, 1.0, 1.3, 2.7, 2.75, 5.0, 5.01, 9.3])
    mu, alpha, beta = 1.0, 0.5, 2.0

    naive = np.array(
        calculate_hawkes_intensity_series(event_times, mu, alpha, beta)
    )
    fast = calculate_hawkes_intensity_series_fast(
        event_times, mu, alpha, beta
    )

    np.testing.assert_allclose(naive, fast, rtol=1e-10, atol=1e-12)


def test_intensity_increases_right_after_an_event_cluster():

    event_times = np.array([1.0, 1.01, 1.02])
    mu, alpha, beta = 1.0, 5.0, 1.0

    intensities = calculate_hawkes_intensity_series_fast(
        event_times, mu, alpha, beta
    )

    # Each successive event occurs while the previous events'
    # excitation has barely decayed, so intensity should climb.
    assert intensities[1] > intensities[0]
    assert intensities[2] > intensities[1]
