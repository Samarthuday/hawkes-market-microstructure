import numpy as np
import pandas as pd


def hawkes_intensity(t, event_times, mu, alpha, beta):
    past_events = event_times[event_times < t]
    intensity = mu + np.sum(alpha * np.exp(-beta * (t - past_events)))
    return intensity

def calculate_hawkes_intensity_series(event_times, mu, alpha, beta): # intensities  → the list holding all results & intensity   → one result for the current t
    intensities = []
    for t in event_times:
        intensity = hawkes_intensity(t, event_times, mu, alpha, beta)
        intensities.append(intensity)
    return intensities

def create_hawkes_intensity_dataframe(event_times, mu, alpha, beta):
    intensities = calculate_hawkes_intensity_series(event_times, mu, alpha, beta)
    df = pd.DataFrame({"timestamp": event_times, "intensity": intensities})
    return df

if __name__ == "__main__":
    event_times = np.array([1.0, 2.0, 4.0])

    mu = 1.0
    alpha = 0.5
    beta = 1.0

    print(hawkes_intensity(5.0, event_times, mu, alpha, beta))

    df = create_hawkes_intensity_dataframe(
        event_times,
        mu,
        alpha,
        beta
    )

    print(df)