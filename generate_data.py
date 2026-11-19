"""
generate_data.py -- seeded generator for a SYNTHETIC Carolinas charter-trip dataset.

One row per booked charter trip: either an inshore trip (4 hours) or an offshore
trip (8 hours, May-October only). Three simulated seasons. Running this file
twice gives the identical CSV (fixed seed).

THIS DATA IS SYNTHETIC. It shows that the modeling pipeline works. It does not
show that real fish behave this way.

Target: productive = 1 if keepers per angler >= 2 (inshore) or >= 4 (offshore).

Planted effects (see data_spec.md for the full story):
  Both trip types
    + Hot streak: a slowly drifting state; the trailing 3-day catch rate (same
      trip type) is a noisy window onto it
    + Species: small effects
    + Days since last front: small positive effect, capped at 5 days
  Inshore
    + Water temperature sweet spot near 70F (non-linear)
    + Wind: higher actual wind lowers catch
    + Tide: moving tide beats slack tide (strong)
  Offshore
    + Water temperature sweet spot near 74F (non-linear)
    + Sea state: actual wave height above ~3 ft lowers catch (strong)
    + Wind: weak direct effect (it matters mostly through waves)
    + Tide: very weak
Planted NULL effects (the model should NOT find these as real drivers):
  party_size (per-angler rate is unaffected), moon_phase, fcst_air_temp_f
  (air temp tracks water temp, so it may look useful only by proxy), and
  wave height on inshore trips.
Planted LEAKY column:
  bites_logged_during_trip -- only exists after the trip happens.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 355
N_SEASONS = 3
START_YEAR = 2022            # simulated seasons; dates are not real records
OUT = Path(__file__).parent / "data" / "fishing_trips.csv"

TRIP_HOURS = {"inshore": 4, "offshore": 8}
CUTOFF = {"inshore": 2, "offshore": 4}      # keepers per angler for "productive"
OFFSHORE_MONTHS = range(5, 11)              # May through October

INSHORE_SPECIES = ["redfish", "speckled_trout", "flounder"]
INSHORE_P = [0.40, 0.35, 0.25]
SPECIES_EFFECT = {"redfish": 0.15, "speckled_trout": 0.0, "flounder": -0.20,
                  "grouper": 0.0, "snapper": 0.10, "jacks": 0.20, "cobia": -0.30}
TIDES = ["incoming", "outgoing", "slack_high", "slack_low"]
TIDE_P = [0.30, 0.30, 0.20, 0.20]
DEFAULT_RATE = {"inshore": 0.55, "offshore": 0.55}   # keepers per angler-hour


def moon_phase(day):
    """Fraction of the 29.53-day lunar cycle (0 = new moon, 0.5 = full)."""
    days = (pd.Timestamp(day) - pd.Timestamp("2000-01-06")).days
    return (days % 29.53) / 29.53


def trailing_rate(history, day, trip_type):
    """Mean keepers per angler-hour of this trip type over the previous 3 days.
    Uses earlier days only, so it is known the evening before."""
    h = [(d, r) for d, r, t in history if t == trip_type]
    recent = [r for d, r in h
              if day - pd.Timedelta(days=3) <= d <= day - pd.Timedelta(days=1)]
    if recent:
        return float(np.mean(recent))
    if h:
        return float(np.mean([r for _, r in h]))
    return DEFAULT_RATE[trip_type]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    history = []  # (date, keepers per angler-hour, trip_type) for every earlier trip
    n = 0

    for s in range(N_SEASONS):
        year = START_YEAR + s
        days = pd.date_range(f"{year}-04-01", f"{year}-11-30")
        state = 0.0          # slowly drifting "how well are the fish biting" level
        wt_noise = 0.0
        since_front = 3

        for day in days:
            doy = day.dayofyear
            state = 0.9 * state + rng.normal(0, np.sqrt(1 - 0.9 ** 2))
            wt_noise = 0.8 * wt_noise + rng.normal(0, 1.2)
            wt_true = 66 + 17 * np.sin(2 * np.pi * (doy - 110) / 365) + wt_noise
            since_front = 0 if rng.random() < 0.15 else since_front + 1
            wind_actual = rng.gamma(4.0, 2.6)
            wave_actual = 0.18 * wind_actual + rng.gamma(2.0, 0.6)   # feet
            air = wt_true + rng.normal(3, 6)

            # Known the evening before (what a captain would actually have):
            fcst_wind = max(0.0, wind_actual + rng.normal(0, 2.5))
            fcst_wave = max(0.5, wave_actual + rng.normal(0, 0.7))
            fcst_air = air + rng.normal(0, 2)
            water_measured = wt_true + rng.normal(0, 1)

            trips = ["inshore"] * int(rng.poisson(1.9))
            if day.month in OFFSHORE_MONTHS:
                trips += ["offshore"] * int(rng.poisson(0.8))

            todays = []
            for trip_type in trips:
                n += 1
                off = trip_type == "offshore"
                tide = str(rng.choice(TIDES, p=TIDE_P))
                moving = tide in ("incoming", "outgoing")

                if off:
                    if day.month <= 6:      # spring cobia run
                        species = str(rng.choice(["grouper", "snapper", "jacks", "cobia"],
                                                 p=[0.30, 0.30, 0.15, 0.25]))
                    else:
                        species = str(rng.choice(["grouper", "snapper", "jacks"],
                                                 p=[0.40, 0.40, 0.20]))
                    party = int(rng.choice([2, 3, 4, 5, 6], p=[0.15, 0.20, 0.30, 0.20, 0.15]))
                    z = (np.log(4.4) + 0.15
                         - 0.35 * ((wt_true - 74) / 10) ** 2
                         - 0.02 * (wind_actual - 10)
                         - 0.30 * max(0.0, wave_actual - 3.0)
                         + (0.05 if moving else -0.05)
                         + 0.30 * state
                         + SPECIES_EFFECT[species]
                         + 0.03 * min(since_front, 5)
                         + rng.normal(0, 0.30))
                else:
                    species = str(rng.choice(INSHORE_SPECIES, p=INSHORE_P))
                    party = int(rng.choice([2, 3, 4, 5, 6], p=[0.25, 0.25, 0.25, 0.15, 0.10]))
                    z = (np.log(2.3) + 0.25
                         - 0.5 * ((wt_true - 70) / 10) ** 2
                         - 0.05 * (wind_actual - 10)
                         + (0.25 if moving else -0.25)
                         + 0.40 * state
                         + SPECIES_EFFECT[species]
                         + 0.03 * min(since_front, 5)
                         + rng.normal(0, 0.30))

                hours = TRIP_HOURS[trip_type]
                keepers = rng.poisson(np.exp(z) * party)
                productive = int(keepers >= CUTOFF[trip_type] * party)
                bites = int(keepers * 1.8 + rng.poisson(2))   # LEAKY: post-trip only

                rows.append({
                    "trip_id": f"T{n:04d}",
                    "trip_date": day.date().isoformat(),
                    "season": s + 1,
                    "month": day.month,
                    "trip_type": trip_type,
                    "trip_hours": hours,
                    "species_targeted": species,
                    "party_size": party,
                    "tide_stage_start": tide,
                    "moon_phase": round(moon_phase(day), 3),
                    "days_since_front": int(since_front),
                    "fcst_wind_mph": round(fcst_wind, 1),
                    "fcst_wave_height_ft": round(fcst_wave, 1),
                    "fcst_air_temp_f": round(fcst_air, 1),
                    "water_temp_f": round(water_measured, 1),
                    "trailing_3day_catch_rate": round(trailing_rate(history, day, trip_type), 3),
                    "bites_logged_during_trip": bites,
                    "productive": productive,
                })
                todays.append((day, keepers / (party * hours), trip_type))

            history.extend(todays)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Date range: {df.trip_date.min()} to {df.trip_date.max()}")
    print(f"Base rate (productive = 1): {df.productive.mean():.3f}")
    print(df.groupby("season").productive.agg(["count", "mean"]).round(3))
    print(df.groupby("trip_type").productive.agg(["count", "mean"]).round(3))
