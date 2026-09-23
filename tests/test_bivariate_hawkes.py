import numpy as np

from bivariate_hawkes import calculate_shared_beta_bases


def test_simultaneous_buy_and_sell_do_not_excite_each_other():
    """
    Regression test for a real bug: events sharing the exact same
    timestamp (a real occurrence -- a buy-initiated and sell-initiated
    trade can print in the same microsecond) used to be processed one
    at a time in concatenation order, so whichever side was
    concatenated first would spuriously appear to excite the other at
    Delta t = 0. The fix processes tied events as a batch: every event
    in a tie group sees the same pre-group state, and none of them can
    see each other.
    """

    buy_times = np.array([1.0, 2.0])
    sell_times = np.array([1.0])
    beta = 1.0

    base_buy_at_buy, base_sell_at_buy, base_buy_at_sell, base_sell_at_sell = (
        calculate_shared_beta_bases(buy_times, sell_times, beta)
    )

    # The buy at t=1 must not see the simultaneous sell at t=1, and
    # vice versa -- both pre-event bases at t=1 must be exactly zero.
    assert base_sell_at_buy[0] == 0.0
    assert base_buy_at_sell[0] == 0.0

    # The later buy at t=2 must see BOTH the t=1 buy and the t=1 sell,
    # each decayed by exp(-beta * 1.0).
    expected_decayed = np.exp(-beta * 1.0)
    assert abs(base_buy_at_buy[1] - expected_decayed) < 1e-12
    assert abs(base_sell_at_buy[1] - expected_decayed) < 1e-12


def test_tie_handling_is_symmetric_regardless_of_concatenation_order():
    """
    Swapping which side is treated as "buy" vs "sell" for a
    simultaneous pair must not change which one appears to excite the
    other -- neither should, in either labeling.
    """

    tied_time = 5.0
    beta = 2.0

    buy_times = np.array([tied_time])
    sell_times = np.array([tied_time])

    result_a = calculate_shared_beta_bases(buy_times, sell_times, beta)
    result_b = calculate_shared_beta_bases(sell_times, buy_times, beta)

    for arr in list(result_a) + list(result_b):
        assert np.all(arr == 0.0)
