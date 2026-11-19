# gta6-rarity-forecast-c

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-c/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-c/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Refresh-aware empirical forecast of GTA VI trophy rarities on Exophase. Exophase updates in jumps, not continuously: between refreshes the tracked-player count is frozen and every rate sits still; at a refresh the population jumps and each rate is dragged toward the rate among the newly tracked players, who are on average much earlier in the game. This entry simulates that process directly — a sequence of dilution steps interleaved with smooth cohort progression — measures the jump sizes, spacing and implied new-player rates from the case history itself when it contains jumps, and reads five quantiles off five scenarios spanning the refresh-count, new-player-rate and progression uncertainty.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

`main.py` is Python 3.13, standard library only, no network, no model calls at run time. It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `/output/<id>.out` (`$MAKEFUN_OUTPUT_DIR` else `/output/`). Output is deterministic: two runs on the same case are byte-identical.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
