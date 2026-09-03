import numpy as np
import pandas as pd


def hawkes_intensity(t, event_times, mu, alpha, beta):
    """
    Calculate the Hawkes process intensity at time t.

    For an exponential-kernel Hawkes process:

        λ(t) = μ + Σ_{t_i < t} α exp[-β(t - t_i)]

    where:

        μ     = baseline intensity
        α     = excitation parameter
        β     = decay parameter
        t_i   = previous event times

    Each previous event increases the current intensity by

        α exp[-β(t - t_i)]

    and this contribution decays exponentially as time passes.
    """

    # Select only events that occurred before time t.
    # The current event is excluded because we are calculating
    # the pre-event intensity λ(t^-).
    past_events = event_times[event_times < t]

    # Baseline intensity + contribution from all previous events.
    intensity = mu + np.sum(
        alpha * np.exp(-beta * (t - past_events))
    )

    return intensity


def calculate_hawkes_intensity_series(event_times, mu, alpha, beta):
    """
    Calculate the Hawkes intensity at every event time.

    Returns a sequence:

        [λ(t_1), λ(t_2), ..., λ(t_n)]

    where each intensity is calculated using only events
    occurring before that event.
    """

    # Store the intensity corresponding to each event time.
    intensities = []

    # Calculate the intensity separately for every event time.
    for t in event_times:

        # Calculate the pre-event Hawkes intensity λ(t^-).
        intensity = hawkes_intensity(
            t,
            event_times,
            mu,
            alpha,
            beta
        )

        # Add the calculated intensity to the list.
        intensities.append(intensity)

    return intensities


def create_hawkes_intensity_dataframe(event_times, mu, alpha, beta):
    """
    Create a DataFrame containing event times and their
    corresponding Hawkes intensities.
    """

    # Calculate the Hawkes intensity for every event time.
    intensities = calculate_hawkes_intensity_series(
        event_times,
        mu,
        alpha,
        beta
    )

    # Combine event times and intensities into a DataFrame.
    df = pd.DataFrame({
        "timestamp": event_times,
        "intensity": intensities
    })

    return df


if __name__ == "__main__":

    # Simple example event times for testing.
    event_times = np.array([
        1.0,
        2.0,
        4.0
    ])

    # Hawkes process parameters.
    #
    # μ     = baseline intensity
    # α     = strength of excitation
    # β     = rate at which excitation decays
    mu = 1.0
    alpha = 0.5
    beta = 1.0

    # Calculate intensity at t = 5.
    intensity = hawkes_intensity(
        5.0,
        event_times,
        mu,
        alpha,
        beta
    )

    print(intensity)

    # Create a DataFrame containing the intensity
    # associated with each event time.
    df = create_hawkes_intensity_dataframe(
        event_times,
        mu,
        alpha,
        beta
    )

    print(df)