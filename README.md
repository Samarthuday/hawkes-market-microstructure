# Market Order Self-Excitation with Hawkes Processes

A quantitative finance research project studying the temporal structure, clustering, and self-excitation of market order arrivals using **Poisson and Hawkes point processes**.

The central question is:

> **Does the arrival of one market order increase the probability of subsequent orders?**

The project begins by testing whether market-order arrivals can reasonably be modeled as a homogeneous Poisson process. When the empirical data exhibits clustering and temporal dependence inconsistent with the Poisson assumption, a **Hawkes process** is introduced to model the resulting self-excitation.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Objectives](#objectives)
- [Data](#data)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Poisson Baseline](#poisson-baseline)
- [Inter-Arrival Times](#inter-arrival-times)
- [Coefficient of Variation](#coefficient-of-variation)
- [Event Counts and Fano Factor](#event-counts-and-fano-factor)
- [Autocorrelation](#autocorrelation)
- [Windowed Event Intensity](#windowed-event-intensity)
- [Hawkes Process](#hawkes-process)
- [Current Hawkes Implementation](#current-hawkes-implementation)
- [Initial Hawkes Example](#initial-hawkes-example)
- [Current Findings](#current-findings)
- [Methodology](#methodology)
- [Next Steps](#next-steps)
- [Research Interpretation](#research-interpretation)
- [Disclaimer](#disclaimer)

---

## Research Motivation

In a homogeneous Poisson process, events arrive independently at a constant rate:

$$
\lambda(t) = \mu
$$

where $\mu$ is the baseline event intensity.

Financial markets, however, often exhibit **order clustering**. A burst of trading activity can be followed by additional orders arriving at high frequency before activity eventually decays.

This behavior suggests that the probability of an order arriving may depend on previous orders.

A Hawkes process captures this behavior through a time-varying intensity:

$$
\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t-t_i)}
$$

where:

| Symbol | Meaning |
|---|---|
| $\mu$ | Baseline intensity |
| $\alpha$ | Excitation strength |
| $\beta$ | Decay rate |
| $t_i$ | Previous event times |

Each previous event temporarily increases the intensity, with its effect decaying exponentially over time.

---

## Objectives

The project is being developed in stages:

1. Load and process high-frequency trade data.
2. Extract unique market-order event times.
3. Calculate inter-arrival times.
4. Establish a homogeneous Poisson baseline.
5. Test the exponential inter-arrival assumption.
6. Measure dispersion using the coefficient of variation.
7. Measure event clustering using the Fano factor.
8. Analyze temporal dependence using autocorrelation.
9. Construct windowed event-count and intensity series.
10. Implement a Hawkes-process intensity.
11. Estimate Hawkes parameters from the observed data.
12. Evaluate whether the Hawkes model better explains market-order clustering.

---

## Data

The current analysis uses Binance BTC/USDT trade data:

```text
BTCUSDT-trades-2025-01.csv
```

The raw data contains fields including:

- `trade_id`
- `price`
- `quantity`
- `quote_quantity`
- `timestamp_us`
- `is_buyer_maker`
- `is_best_match`

For the current experiments, the first **1,000,000 trades** are loaded.

---

## Project Structure

```text
hawkes-market-microstructure/
│
├── data/
│   └── BTCUSDT-trades-2025-01.csv
│
├── src/
│   ├── data_loader.py
│   ├── trade_processing.py
│   ├── poisson.py
│   ├── poisson_diagnostics.py
│   └── hawkes.py
│
├── README.md
└── ...
```

| File | Description |
|---|---|
| `data_loader.py` | Responsible for loading the raw trade data into a pandas DataFrame. |
| `trade_processing.py` | Handles timestamp conversion and basic event-time processing. |
| `poisson.py` | Provides the basic homogeneous Poisson intensity estimate. |
| `poisson_diagnostics.py` | Contains the statistical diagnostics used to evaluate whether the observed event process resembles a Poisson process. |
| `hawkes.py` | Contains the initial implementation of the Hawkes-process intensity. |

---

## Data Processing

### Timestamp Conversion

The raw timestamps are provided in Unix microseconds. They are converted to UTC timestamps:

```python
data["timestamp"] = pd.to_datetime(
    data["timestamp_us"],
    unit="us",
    utc=True
)
```

Inter-arrival times are calculated as:

$$
\Delta t_i = t_i - t_{i-1}
$$

and converted from microseconds to seconds:

$$
\Delta t_{\text{seconds}} = \frac{\Delta t_{\text{microseconds}}}{10^6}
$$

### Event-Time Extraction

The analysis uses unique, chronologically ordered event timestamps. The processing pipeline:

1. Remove missing timestamps.
2. Remove duplicate timestamps.
3. Sort chronologically.
4. Reset the index.

This produces the event sequence

$$
t_1, t_2, \ldots, t_N
$$

used throughout the point-process analysis.

---

## Poisson Baseline

Before introducing self-excitation, the project establishes a homogeneous Poisson baseline.

For a Poisson process with constant intensity $\lambda$, the maximum-likelihood estimate is:

$$
\hat{\lambda} = \frac{N-1}{T}
$$

where

$$
T = t_N - t_1
$$

is the total observation time. The estimated intensity is expressed in events/second.

For the current dataset:

```text
Poisson intensity ≈ 3.764 events/second
```

The mean inter-arrival time is:

```text
≈ 0.265674 seconds
```

which agrees with:

$$
E[\Delta t] = \frac{1}{\lambda}
$$

and therefore:

```text
1 / λ ≈ 0.265674 seconds
```

This agreement provides a basic consistency check for the intensity calculation.

---

## Inter-Arrival Times

For a homogeneous Poisson process, inter-arrival times follow an exponential distribution:

$$
f(x) = \lambda e^{-\lambda x}, \qquad x \geq 0
$$

The project compares the empirical distribution of observed inter-arrival times with this theoretical exponential distribution. Both the empirical vs. theoretical PDF and the empirical vs. theoretical CDF are examined.

The empirical CDF is constructed as:

$$
\hat F(x) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\Delta t_i \leq x)
$$

---

## Coefficient of Variation

The coefficient of variation (CV) provides a scale-independent measure of dispersion:

$$
CV = \frac{\sigma_{\Delta t}}{\mu_{\Delta t}}
$$

For an exponential distribution:

$$
CV = 1
$$

Therefore, a value substantially different from 1 provides evidence that the inter-arrival process does not behave like an ideal Poisson process. The project calculates the CV of the observed inter-arrival times as part of the Poisson diagnostics.

---

## Event Counts and Fano Factor

To study clustering over different time scales, event times are divided into fixed windows of width $\Delta$. For each window:

$$
N_k = \text{number of events in window } k
$$

This produces an event-count time series:

```text
[0, 10)       143
[10, 20)      152
[20, 30)      125
[30, 40)       77
...
```

For a Poisson process, the number of events in a fixed interval follows:

$$
N(\Delta) \sim \text{Poisson}(\lambda \Delta)
$$

and therefore:

$$
E[N] = Var(N)
$$

The Fano factor is:

$$
F = \frac{Var(N)}{E[N]}
$$

For an ideal Poisson process, $F = 1$. Values substantially larger than 1 indicate **over-dispersion and clustering**.

### Observed Fano Factors

| Window Size (s) | Fano Factor |
| ---: | ---: |
| 1 | 13.08 |
| 2 | 17.30 |
| 5 | 25.29 |
| 10 | 36.44 |
| 20 | 53.22 |
| 50 | 84.66 |
| 100 | 136.34 |

The Fano factor is far above the Poisson benchmark of 1 across the tested window sizes. This provides strong evidence that the event process exhibits substantial clustering and is not well described by a homogeneous Poisson process.

---

## Autocorrelation

The event-count series is also analyzed for temporal dependence. For lag $k$, the autocorrelation measures the relationship between $N_t$ and $N_{t+k}$.

The current implementation calculates:

$$
\rho_k = \frac{\gamma_k}{\gamma_0}
$$

where $\gamma_k$ is the lag-$k$ autocovariance. A simpler pandas-based alternative is also provided:

```python
event_count_series.autocorr(lag=lag)
```

### Observed Autocorrelation

Using a 10-second event-count window:

```text
Lag  1: 0.998785
Lag  2: 0.997029
Lag  3: 0.996213
...
Lag 20: 0.994272
```

The event-count series exhibits extremely strong positive temporal dependence. This is inconsistent with independent Poisson increments and further motivates the use of a self-exciting point process.

---

## Windowed Event Intensity

The event-count series can be converted into an empirical intensity series. For a window of width $\Delta$:

$$
\hat{\lambda}_k = \frac{N_k}{\Delta}
$$

where $N_k$ is the number of events observed in window $k$. The resulting series represents the observed event intensity over time, allowing periods of elevated and reduced trading activity to be visualized directly.

---

## Hawkes Process

The project then moves from a constant-intensity Poisson model to a self-exciting Hawkes process. The Hawkes intensity is:

$$
\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t-t_i)}
$$

| Parameter | Role |
|---|---|
| $\mu$ (baseline intensity) | Background rate of events in the absence of recent activity |
| $\alpha$ (excitation parameter) | Controls how strongly each previous event increases future intensity |
| $\beta$ (decay parameter) | Controls how quickly the effect of a previous event disappears — a larger $\beta$ means faster decay |

---

## Current Hawkes Implementation

The current implementation evaluates:

```python
def hawkes_intensity(t, event_times, mu, alpha, beta):

    past_events = event_times[event_times < t]

    intensity = (
        mu
        + np.sum(
            alpha * np.exp(
                -beta * (t - past_events)
            )
        )
    )

    return intensity
```

For every time $t$, the function:

1. Finds all previous events.
2. Calculates the contribution of each previous event.
3. Applies exponential decay.
4. Adds all contributions to the baseline intensity.

The project also constructs an intensity series $\lambda(t_1), \lambda(t_2), \ldots, \lambda(t_N)$ and stores the result in a DataFrame containing:

```text
timestamp | intensity
```

---

## Initial Hawkes Example

For the initial implementation, the following parameters are used:

```python
mu = 1.0
alpha = 0.5
beta = 1.0
```

with example event times:

```python
event_times = np.array([1.0, 2.0, 4.0])
```

At a time $t = 5$:

$$
\lambda(5) = 1 + 0.5e^{-1(5-1)} + 0.5e^{-1(5-2)} + 0.5e^{-1(5-4)}
$$

which produces the corresponding Hawkes intensity.

> **Note:** These parameters are currently **illustrative** and have not yet been estimated from the market data.

---

## Current Findings

The preliminary analysis suggests that BTC/USDT trade arrivals exhibit strong temporal structure. The current diagnostics show:

- Mean inter-arrival time: $\approx 0.265674$ s
- Estimated Poisson intensity: $\approx 3.764$ events/s
- Fano factors substantially greater than 1
- Strong positive autocorrelation in event counts
- Event intensity varies substantially over time

The observed event process therefore shows behavior inconsistent with a simple homogeneous Poisson model. These results motivate the transition to a Hawkes-process framework.

Importantly, these diagnostics establish **evidence of clustering**, but do not by themselves prove that a Hawkes process is the uniquely correct model. Model fitting and validation are required.

---

## Methodology

```text
Raw Binance Trade Data
          │
          ▼
   Timestamp Processing
          │
          ▼
   Unique Event Times
          │
          ▼
    Inter-Arrival Times
          │
          ▼
   ┌──────────────────┐
   │ Poisson Baseline  │
   └──────────────────┘
          │
          ▼
   Distribution Tests
          │
          ├── Exponential PDF/CDF
          ├── Coefficient of Variation
          ├── Fano Factor
          ├── Event Counts
          ├── Autocorrelation
          └── Windowed Intensity
          │
          ▼
   Evidence of Clustering
          │
          ▼
    Hawkes Process
          │
          ▼
   Parameter Estimation
          │
          ▼
      Model Validation
```

---

## Next Steps

The next stage of the project will focus on fitting the Hawkes process to the observed event data rather than using manually selected parameters.

Planned work includes:

- [ ] Estimate $\mu, \alpha, \beta$ from observed event times.
- [ ] Implement Hawkes log-likelihood.
- [ ] Numerically optimize the likelihood.
- [ ] Calculate the branching ratio: $n = \dfrac{\alpha}{\beta}$
- [ ] Check the stability condition: $\dfrac{\alpha}{\beta} < 1$
- [ ] Compare fitted Hawkes intensity with empirical windowed intensity.
- [ ] Analyze the model residuals.
- [ ] Perform goodness-of-fit diagnostics.
- [ ] Compare Poisson and Hawkes models quantitatively.
- [ ] Study the persistence and magnitude of order-flow aftershocks.
- [ ] Extend the model to distinguish buy and sell order arrivals.

---

## Research Interpretation

The ultimate goal is not simply to fit a Hawkes process, but to quantify **market-order self-excitation**. A fitted model can potentially answer questions such as:

- How much does one order increase short-term order-arrival intensity?
- How quickly does this effect decay?
- What fraction of observed activity can be attributed to endogenous excitation?
- Does excitation differ across market conditions?
- Are buy and sell orders characterized by different excitation dynamics?

The project therefore treats the Hawkes process as a quantitative framework for studying **order-flow clustering and market microstructure dynamics**.

> **Working conclusion (current stage):** The data exhibits strong clustering, over-dispersion, and temporal dependence that are inconsistent with a simple homogeneous Poisson process and motivate a Hawkes-process model. This is a more defensible claim than asserting that self-excitation has been proven — that requires fitted parameters and validation, which are the subject of the next stage.

---

## Disclaimer

This repository is a quantitative research and modeling project. The results are exploratory and should not be interpreted as trading advice or as evidence of a directly exploitable trading strategy. Model assumptions, parameter estimates, and empirical conclusions should be validated using additional data and appropriate statistical tests.