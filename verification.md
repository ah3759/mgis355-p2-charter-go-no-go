# Verification

- **Two traced numbers:** Rank-based AUC recomputed from raw season-3 predictions was 0.728985; saved-model AUC was 0.728985; they matched within floating-point tolerance. False positives recomputed with a plain boolean count were 157, matching the saved metric (157).
- **Leakage check:** `bites_logged_during_trip` increased AUC from 0.729 to 0.983, confirming it must stay excluded. `generate_data.py` computes `trailing_3day_catch_rate` from dates between day-3 and day-1 only. Removing it changed AUC from 0.729 to 0.668; this is predictive signal, not evidence of leakage.
- **Random vs time split:** Random-split AUC was 0.764 versus time-split AUC 0.729; the random split is optimistic by 0.035.

## Recovery check

| Planted effect or null | What the model found |
|---|---|
| Hot streak through trailing rate | Highest permutation importance (0.113 AUC drop). |
| Moving tide beats slack inshore | Second-highest importance (0.058); directional effect is represented in the pipeline. |
| Higher wind lowers inshore catch | Third-highest importance (0.043); direction is consistent. |
| Small species effects | Positive permutation importance (0.022). |
| Water-temperature sweet spots near 70F/74F | Global permutation drop was near zero (0.000), so this nonlinear planted effect was not strongly recovered by this aggregate check. |
| Wave height above 3 ft offshore; null inshore | Small positive global importance (0.007); the combined model does not cleanly separate the trip-type interaction. |
| Nulls: party size, moon phase, forecast air temperature | Near-zero permutation drops for all three, consistent with the planted nulls. |

The trip-type response check from the fitted pipeline had mean predicted productivity of 0.615 for inshore and 0.588 for offshore probes. Effects can also arise from proxies: air temperature can track water temperature and month can proxy seasonal water conditions.

Synthetic data: this shows the pipeline works, not that real fish behave this way.
