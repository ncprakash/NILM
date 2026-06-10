"""Generate realistic synthetic aggregate power traces for demo mode."""
import numpy as np


APPLIANCE_PROFILES = {
    "kettle": {
        "power": 2200, "on_duration": (2, 5), "cycles_per_hour": 0.4,
        "rise_time": 3, "color": "#E63946",
    },
    "fridge": {
        "power": 150, "on_duration": (8, 20), "cycles_per_hour": 3,
        "rise_time": 5, "color": "#06A77D",
    },
    "washing_machine": {
        "power": 1800, "on_duration": (50, 90), "cycles_per_hour": 0.15,
        "rise_time": 10, "color": "#1D7DBC",
    },
    "dish_washer": {
        "power": 1500, "on_duration": (60, 120), "cycles_per_hour": 0.1,
        "rise_time": 15, "color": "#8338EC",
    },
}


def generate_demo_trace(hours=6, sample_period=6, seed=42):
    """Return dict of {appliance: watts_array} plus aggregate."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / sample_period)
    t = np.arange(n)

    traces = {}
    for app, prof in APPLIANCE_PROFILES.items():
        signal = np.zeros(n, dtype=np.float32)
        interval = int(3600 / sample_period / prof["cycles_per_hour"])
        start = rng.integers(0, max(1, interval // 3))
        while start < n:
            dur = int(rng.uniform(*prof["on_duration"]) * 60 / sample_period)
            end = min(start + dur, n)
            pw = prof["power"] * rng.uniform(0.92, 1.05)
            rise = min(prof["rise_time"], dur // 3)
            for i in range(rise):
                signal[start + i] = pw * (i + 1) / rise
            signal[start + rise:end - rise] = pw
            for i in range(rise):
                if end - rise + i < n:
                    signal[end - rise + i] = pw * (rise - i) / rise
            jitter = rng.integers(interval // 4, interval)
            start = end + jitter

        noise = rng.normal(0, prof["power"] * 0.015, n).astype(np.float32)
        traces[app] = np.clip(signal + noise, 0, None)

    baseline = 80 + rng.normal(0, 5, n).astype(np.float32)
    aggregate = baseline + sum(traces.values())
    return aggregate, traces
