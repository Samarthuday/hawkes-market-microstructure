import numpy as np
from scipy.optimize import minimize

from data_loader import load_trade_data
from trade_processing import process_trade_data

# Modeling simplification, documented up front: all four kernels
# (buy->buy, sell->buy, buy->sell, sell->sell) share the SAME decay
# rate beta, and only the four excitation strengths (alpha_BB,
# alpha_BS, alpha_SB, alpha_SS) differ. A fully general model would
# give each of the 4 kernels its own independent beta, but the
# exponential-kernel Hawkes recursion's O(n) trick only works because
# a *single* decaying state can be updated incrementally as events
# arrive; with 4 independent betas, computing the log-intensity sum
# would need 4 separate O(n) passes with no shared state, which is a
# reasonable extension but was out of scope for this pass. The shared
# common decay rate. See calculate_branching_matrix for the resulting
# 2x2 excitation matrix and its stability condition.


def extract_side_event_times(processed_data):
    """
    Split trades into buy-initiated and sell-initiated event streams
    using the aggressor side implied by `is_buyer_maker`:

        is_buyer_maker == False -> the buyer is the taker
                                    -> buy-initiated trade
        is_buyer_maker == True  -> the seller is the taker
                                    -> sell-initiated trade

    Each side is deduplicated to unique timestamps independently
    (mirroring trade_processing.extract_event_times), but both sides
    are converted to elapsed seconds from the SAME global start time
    so the two streams share one time axis.
    """

    global_start = processed_data["timestamp"].min()

    def side_event_times(mask):
        timestamps = (
            processed_data.loc[mask, "timestamp"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        elapsed = (timestamps - global_start).dt.total_seconds()
        return elapsed.to_numpy()

    buy_times = side_event_times(~processed_data["is_buyer_maker"])
    sell_times = side_event_times(processed_data["is_buyer_maker"])

    return buy_times, sell_times


def calculate_shared_beta_bases(buy_times, sell_times, beta):
    """
    Compute, for every buy event and every sell event, the pre-event
    "base" (alpha-free) excitation contributed by prior buy events and
    by prior sell events, using a single O(n_buy + n_sell) merged pass
    with the standard Hawkes recursive-decay trick:

        base_source(t) = Sum_{t_j^source < t} exp(-beta * (t - t_j^source))

    Because all four kernels share beta, base_buy(t) and base_sell(t)
    can be tracked with two running scalars while sweeping through the
    time-merged buy/sell timeline once, then multiplied by whichever
    alpha applies once we know the target side:

        lambda_B(t) = mu_B + alpha_BB * base_buy(t) + alpha_BS * base_sell(t)
        lambda_S(t) = mu_S + alpha_SB * base_buy(t) + alpha_SS * base_sell(t)

    Returns four arrays aligned with buy_times/sell_times:
        base_buy_at_buy, base_sell_at_buy   (aligned with buy_times)
        base_buy_at_sell, base_sell_at_sell (aligned with sell_times)
    """

    n_buy = len(buy_times)
    n_sell = len(sell_times)

    all_times = np.concatenate([buy_times, sell_times])
    # False = buy, True = sell, in the same concatenated order as
    # all_times (buy block first, then sell block).
    is_sell = np.concatenate([
        np.zeros(n_buy, dtype=bool),
        np.ones(n_sell, dtype=bool)
    ])

    order = np.argsort(all_times, kind="stable")
    sorted_times = all_times[order]
    # Plain Python list, not a numpy array: this is accessed one
    # scalar element at a time in the tight loop below, and native
    # list/bool indexing has far less per-access overhead there than
    # repeated single-element numpy indexing.
    sorted_is_sell = is_sell[order].tolist()

    base_buy_pre_sorted = np.empty(len(sorted_times))
    base_sell_pre_sorted = np.empty(len(sorted_times))

    running_buy = 0.0
    running_sell = 0.0
    previous_time = sorted_times[0]

    n_merged = len(sorted_times)
    i = 0

    while i < n_merged:

        t = sorted_times[i]
        delta_t = t - previous_time
        decay = np.exp(-beta * delta_t)

        running_buy *= decay
        running_sell *= decay

        # Events sharing the exact same timestamp t (a real, if rare,
        # occurrence: a buy-initiated and sell-initiated trade can
        # print in the same microsecond) are a "tie group". Every
        # event in the group must see the SAME pre-event state --
        # computed from strictly *before* t, not from partway through
        # processing the group -- otherwise whichever side happened
        # to be concatenated first would spuriously appear to excite
        # the other at Δt=0, breaking the t_j < t_i requirement and
        # introducing an arbitrary buy-vs-sell ordering asymmetry.
        #
        # The vast majority of groups have size 1 (ties are rare --
        # 63 out of ~221k events in the real dataset), so this counts
        # sides with a plain Python loop rather than numpy slicing:
        # slicing into small numpy sub-arrays and calling
        # count_nonzero on every single event (not just tied ones)
        # was roughly 4x slower overall than the pre-fix version that
        # only ever touched scalars.
        j = i
        sell_count = 0
        while j < n_merged and sorted_times[j] == t:
            if sorted_is_sell[j]:
                sell_count += 1
            j += 1

        base_buy_pre_sorted[i:j] = running_buy
        base_sell_pre_sorted[i:j] = running_sell

        running_sell += sell_count
        running_buy += (j - i) - sell_count

        previous_time = t
        i = j

    # Scatter back from sorted (merged) order into the original
    # concatenated [buy_times..., sell_times...] order.
    base_buy_pre = np.empty(len(sorted_times))
    base_sell_pre = np.empty(len(sorted_times))
    base_buy_pre[order] = base_buy_pre_sorted
    base_sell_pre[order] = base_sell_pre_sorted

    base_buy_at_buy = base_buy_pre[:n_buy]
    base_sell_at_buy = base_sell_pre[:n_buy]
    base_buy_at_sell = base_buy_pre[n_buy:]
    base_sell_at_sell = base_sell_pre[n_buy:]

    return base_buy_at_buy, base_sell_at_buy, base_buy_at_sell, base_sell_at_sell


def calculate_bivariate_log_likelihood(
    buy_times,
    sell_times,
    mu_B,
    mu_S,
    alpha_BB,
    alpha_BS,
    alpha_SB,
    alpha_SS,
    beta
):
    """
    Calculate the joint log-likelihood of the bivariate (buy, sell)
    Hawkes process:

        log L = Sum_buy log(lambda_B(t_i))
              + Sum_sell log(lambda_S(t_j))
              - integral_0^T lambda_B(t) dt
              - integral_0^T lambda_S(t) dt

    T is the later of the two streams' last event (both streams are
    observed over the same [0, T] window even though one side may
    have its last event slightly before the other).
    """

    base_buy_at_buy, base_sell_at_buy, base_buy_at_sell, base_sell_at_sell = (
        calculate_shared_beta_bases(buy_times, sell_times, beta)
    )

    lambda_B = mu_B + alpha_BB * base_buy_at_buy + alpha_BS * base_sell_at_buy
    lambda_S = mu_S + alpha_SB * base_buy_at_sell + alpha_SS * base_sell_at_sell

    log_intensity_sum = np.sum(np.log(lambda_B)) + np.sum(np.log(lambda_S))

    T = max(buy_times[-1], sell_times[-1])

    # Closed-form integral of a single exponential kernel's
    # contribution, summed across its source events -- shared between
    # both target equations since the decay rate beta is shared.
    buy_source_integral = np.sum(1 - np.exp(-beta * (T - buy_times))) / beta
    sell_source_integral = np.sum(1 - np.exp(-beta * (T - sell_times))) / beta

    integrated_B = (
        mu_B * T
        + alpha_BB * buy_source_integral
        + alpha_BS * sell_source_integral
    )
    integrated_S = (
        mu_S * T
        + alpha_SB * buy_source_integral
        + alpha_SS * sell_source_integral
    )

    return log_intensity_sum - integrated_B - integrated_S


def _negative_log_likelihood_bivariate(log_params, buy_times, sell_times):

    (
        log_mu_B, log_mu_S,
        log_alpha_BB, log_alpha_BS, log_alpha_SB, log_alpha_SS,
        log_beta
    ) = log_params

    params = np.exp(log_params)

    return -calculate_bivariate_log_likelihood(
        buy_times, sell_times, *params
    )


def estimate_bivariate_hawkes_parameters(buy_times, sell_times, beta0_grid=(10.0, 60.0)):
    """
    Estimate the 7 bivariate Hawkes parameters
    (mu_B, mu_S, alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta) by
    maximum likelihood, optimizing in log-space (so every parameter
    is automatically positive) from a small grid of beta starting
    points. Returns a scipy OptimizeResult whose `.x` is the 7
    parameters in their original (non-log) units.
    """

    buy_rate = len(buy_times) / max(buy_times[-1], sell_times[-1])
    sell_rate = len(sell_times) / max(buy_times[-1], sell_times[-1])

    best_result = None
    best_log_params = None

    for beta0 in beta0_grid:

        # Split each side's empirical rate roughly evenly between
        # self- and cross-excitation as a neutral starting guess.
        alpha0 = 0.2 * beta0

        initial_log_params = np.log([
            0.5 * buy_rate,
            0.5 * sell_rate,
            alpha0, alpha0, alpha0, alpha0,
            beta0
        ])

        result = minimize(
            _negative_log_likelihood_bivariate,
            initial_log_params,
            args=(buy_times, sell_times),
            method="BFGS"
        )

        if not np.isfinite(result.fun):
            continue

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_log_params = result.x

    if best_result is None:
        raise RuntimeError(
            "Bivariate Hawkes MLE failed to converge from every "
            "multi-start initial guess."
        )

    best_result.x = np.exp(best_log_params)

    return best_result


def _negative_log_likelihood_no_cross_excitation(log_params, buy_times, sell_times):
    """
    The restricted model used for the likelihood-ratio test below:
    alpha_BS = alpha_SB = 0 fixed (no cross-side excitation at all),
    leaving only mu_B, mu_S, alpha_BB, alpha_SS, beta free.
    """

    log_mu_B, log_mu_S, log_alpha_BB, log_alpha_SS, log_beta = log_params

    mu_B, mu_S, alpha_BB, alpha_SS, beta = np.exp(log_params)

    return -calculate_bivariate_log_likelihood(
        buy_times, sell_times,
        mu_B, mu_S,
        alpha_BB, 0.0, 0.0, alpha_SS,
        beta
    )


def estimate_no_cross_excitation_parameters(buy_times, sell_times, beta0_grid=(10.0, 60.0)):
    """
    Fit the restricted (same-side-only) model for the likelihood-
    ratio test in test_cross_excitation_significance.
    """

    buy_rate = len(buy_times) / max(buy_times[-1], sell_times[-1])
    sell_rate = len(sell_times) / max(buy_times[-1], sell_times[-1])

    best_result = None
    best_log_params = None

    for beta0 in beta0_grid:

        alpha0 = 0.2 * beta0

        initial_log_params = np.log([
            0.5 * buy_rate, 0.5 * sell_rate, alpha0, alpha0, beta0
        ])

        result = minimize(
            _negative_log_likelihood_no_cross_excitation,
            initial_log_params,
            args=(buy_times, sell_times),
            method="BFGS"
        )

        if not np.isfinite(result.fun):
            continue

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_log_params = result.x

    best_result.x = np.exp(best_log_params)

    return best_result


def test_cross_excitation_significance(buy_times, sell_times, full_log_likelihood):
    """
    Likelihood-ratio test of H0: alpha_BS = alpha_SB = 0 (no cross-
    side excitation at all) against the unrestricted bivariate model.
    Since the fitted cross-side alphas are small but the log-space
    parameterization cannot represent alpha = 0 exactly during the
    unrestricted fit, this restricted-vs-unrestricted comparison is
    the honest way to ask whether that small cross-side excitation is
    actually distinguishable from zero, rather than reading it off a
    parameterization that can only ever return values > 0.

        LR = 2 * (LL_full - LL_restricted) ~ chi2(df=2) under H0
    """

    from scipy.stats import chi2

    restricted_result = estimate_no_cross_excitation_parameters(
        buy_times, sell_times
    )
    restricted_log_likelihood = -restricted_result.fun

    lr_statistic = 2 * (full_log_likelihood - restricted_log_likelihood)
    p_value = chi2.sf(lr_statistic, df=2)

    return {
        "restricted_params": restricted_result.x,
        "restricted_log_likelihood": restricted_log_likelihood,
        "full_log_likelihood": full_log_likelihood,
        "lr_statistic": lr_statistic,
        "p_value": p_value,
    }


def calculate_branching_matrix(alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta):
    """
    Build the 2x2 excitation ("branching") matrix

        G = [[alpha_BB, alpha_BS],
             [alpha_SB, alpha_SS]] / beta

    G[i, j] is the expected number of side-i events directly
    triggered by one side-j event. The bivariate process is
    stationary/stable when the spectral radius of G is < 1
    (the natural multivariate generalization of the univariate
    n = alpha / beta < 1 condition).
    """

    return np.array([
        [alpha_BB, alpha_BS],
        [alpha_SB, alpha_SS]
    ]) / beta


def is_bivariate_stable(alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta):

    G = calculate_branching_matrix(alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(G)))

    return spectral_radius < 1, spectral_radius


if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )
    processed_data = process_trade_data(data)

    buy_times, sell_times = extract_side_event_times(processed_data)

    print("=" * 60)
    print("BUY/SELL BIVARIATE HAWKES MODEL")
    print("=" * 60)

    print(f"\nBuy-initiated events:  {len(buy_times):,}")
    print(f"Sell-initiated events: {len(sell_times):,}")
    print(
        f"Buy share: {len(buy_times) / (len(buy_times) + len(sell_times)):.4%}"
    )

    import time

    print("\nFitting bivariate Hawkes model via MLE...")
    start = time.time()
    result = estimate_bivariate_hawkes_parameters(buy_times, sell_times)
    fit_seconds = time.time() - start
    print(f"Fit completed in {fit_seconds:.2f}s")

    mu_B, mu_S, alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta = result.x

    print("\nEstimated parameters:")
    print(f"mu_B      = {mu_B:.6f}")
    print(f"mu_S      = {mu_S:.6f}")
    print(f"alpha_BB  = {alpha_BB:.6f}")
    print(f"alpha_BS  = {alpha_BS:.6f}")
    print(f"alpha_SB  = {alpha_SB:.6f}")
    print(f"alpha_SS  = {alpha_SS:.6f}")
    print(f"beta      = {beta:.6f} (shared decay rate)")

    G = calculate_branching_matrix(alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta)
    stable, spectral_radius = is_bivariate_stable(
        alpha_BB, alpha_BS, alpha_SB, alpha_SS, beta
    )

    print("\nBranching matrix G = [[n_BB, n_BS], [n_SB, n_SS]]:")
    print(G)
    print(f"\nSpectral radius: {spectral_radius:.6f}")
    print(f"Stable: {stable}")

    print("\nLog-likelihood:", -result.fun)

    print("\n" + "=" * 60)
    print("SAME-SIDE vs CROSS-SIDE EXCITATION")
    print("=" * 60)
    print(f"\nBuy -> Buy   (n_BB): {G[0, 0]:.4f}")
    print(f"Sell -> Buy  (n_BS): {G[0, 1]:.4f}")
    print(f"Buy -> Sell  (n_SB): {G[1, 0]:.4f}")
    print(f"Sell -> Sell (n_SS): {G[1, 1]:.4f}")

    same_side = (G[0, 0] + G[1, 1]) / 2
    cross_side = (G[0, 1] + G[1, 0]) / 2
    print(
        f"\nAverage same-side excitation:  {same_side:.4f}\n"
        f"Average cross-side excitation: {cross_side:.4f}"
    )
    if same_side > cross_side:
        print(
            "Same-side excitation is stronger: a buy is more likely to "
            "trigger another buy than a sell, and vice versa."
        )
    else:
        print(
            "Cross-side excitation is stronger: a trade on one side is "
            "more likely to trigger a trade on the OPPOSITE side "
            "(consistent with e.g. liquidity replenishment / mean "
            "reversion at the microstructure level)."
        )

    print("\n" + "=" * 60)
    print("LIKELIHOOD-RATIO TEST: IS CROSS-SIDE EXCITATION REAL?")
    print("=" * 60)
    print(
        "\nH0: alpha_BS = alpha_SB = 0 (no cross-side excitation at all)"
    )

    lr_result = test_cross_excitation_significance(
        buy_times, sell_times, full_log_likelihood=-result.fun
    )

    print(f"\nRestricted-model log-likelihood: {lr_result['restricted_log_likelihood']:.4f}")
    print(f"Full-model log-likelihood:        {lr_result['full_log_likelihood']:.4f}")
    print(f"LR statistic (df=2):              {lr_result['lr_statistic']:.4f}")
    print(f"p-value:                          {lr_result['p_value']:.6g}")

    if lr_result["p_value"] < 0.05:
        print(
            "\nThe unrestricted model fits significantly better than the "
            "no-cross-excitation model (p < 0.05): the small cross-side "
            "coefficients are statistically distinguishable from zero, "
            "even though they are economically tiny relative to the "
            "same-side coefficients."
        )
    else:
        print(
            "\nThe unrestricted model does NOT fit significantly better "
            "than the no-cross-excitation model: the data cannot rule "
            "out zero cross-side excitation."
        )
