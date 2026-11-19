# Model card

- **Decision and user:** A Carolinas charter captain decides the evening before whether to run a booked trip or offer a reschedule.
- **Unit of prediction:** One booked charter trip.
- **Target:** `productive = 1` when the party reaches the trip-type keeper-per-angler cutoff.
- **Data:** Synthetic seasons 1-3; seasons 1-2 have 1274 training rows and season 3 has 622 test rows.
- **Features:** `trip_type` (known at prediction time? yes), `trip_hours` (known at prediction time? yes), `month` (known at prediction time? yes), `species_targeted` (known at prediction time? yes), `party_size` (known at prediction time? yes), `tide_stage_start` (known at prediction time? yes), `moon_phase` (known at prediction time? yes), `days_since_front` (known at prediction time? yes), `fcst_wind_mph` (known at prediction time? yes), `fcst_wave_height_ft` (known at prediction time? yes), `fcst_air_temp_f` (known at prediction time? yes), `water_temp_f` (known at prediction time? yes), `trailing_3day_catch_rate` (known at prediction time? yes).
- **Excluded on purpose:** `bites_logged_during_trip` (post-trip leakage), `trip_id` (identifier), and `trip_date`/`season` (split context).
- **Split:** Time split: seasons 1-2 for fitting, season 3 for one final evaluation.
- **Baseline and score:** Majority baseline test AUC 0.500, cost $98.39 per trip; month rule AUC 0.470, cost $153.86 per trip.
- **Model and test score:** logistic_regression test AUC 0.729, accuracy 0.706, cost $100.80 per trip.
- **Threshold and cost:** 0.333, derived from $300 false-positive cost and $600 false-negative cost.
- **Where it is weakest:** Month 8 has AUC 0.579.
- **Do not use it for:** Guaranteeing a catch, promising revenue, or generalizing from synthetic data to real fish.
- **What would change the recommendation:** Revisit the recommendation if the false-positive or false-negative cost estimates change, because the threshold is recomputed from those costs.

Synthetic data: this shows the pipeline works, not that real fish behave this way.
