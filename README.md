# Renewable Energy Orchestrator

An agentic energy dispatcher for a utility with 5 solar farms, 3 wind farms, 2 batteries and a grid connection. Every 15 minutes it observes the situation, decides whether to charge, discharge, buy, sell or hold a reserve, applies the decision, and explains it. Built for the ET x Accenture AI Hackathon (Agentic Edition), Problem 4.

**Design rule:** a numeric policy makes every decision. The LLM only turns operator text into structured inputs and decisions into plain English.

## Results (30 days, synthetic data)

| | Baseline | Smart agent |
|---|---|---|
| Net cost | 390,217 | 39,227 |
| Carbon (tonnes) | 7,780 | 13,016 |
| Curtailed MWh | 0 | 0 |

- The smart agent cuts net cost by about 90% and raises carbon by about 67%. This is a cost vs carbon trade-off, tuned in `tune.py`.
- The gain is buy-cheap, sell-high arbitrage in a simplified market with no fees and an unlimited grid. Treat it as an upper bound.
- Forecasting: LightGBM beats "same as yesterday" by 21 to 28% MAE on the last 6 days.
- Forecast-driven threshold on those 6 days: net cost -3,100 vs -2,394 for the fixed threshold (baseline 62,407).
- Storm reserve: stored energy during the storm stays at or above 129.5 MWh, against 0 without the alert.

## Project files

| File | Purpose |
|---|---|
| `make_data.py` | Generates `data.csv` (30 days, 15-minute steps) |
| `battery.py` | `Battery` class: capacity, max rate, efficiency, state of charge, availability |
| `simulator.py` | Baseline and smart controllers, `run_simulation`, energy balance check, outage and storm events |
| `forecast.py` | LightGBM forecasts (p10, p50, p90) for demand, price, solar, wind, with a naive baseline |
| `eda.py` | Data checks and noise measurement |
| `tune.py` | Search over the "cheap price" threshold and carbon weight |
| `test_forecast.py`, `test_storm.py` | Terminal experiments for the forecast-driven agent and the storm reserve |
| `llm.py` | Alert parser, storm parser, decision explainer (Gemini, then Ollama, then fallback) |
| `app.py` | Streamlit dashboard |

## Setup

```
pip install pandas numpy matplotlib lightgbm streamlit requests
```

Python 3.12 or newer is recommended.

### Optional: LLM keys

The app works without any key. It then uses keyword parsing and a labelled template (`[template]`) for explanations.

- **Gemini:** set an environment variable, never put the key in a file.
  - PowerShell: `$env:GEMINI_API_KEY="your_key"`
  - Check the model name in `llm.py` (`GEMINI_MODEL`) against Google AI Studio, as names change.
- **Ollama (local):** install Ollama and run `ollama pull llama3.2:3b`.

## Run order

Scripts depend on each other's output files, so run them in this order:

| Step | Command | Creates |
|---|---|---|
| 1 | `python make_data.py` | `data.csv` |
| 2 | `python simulator.py` | `results_baseline.csv`, `results_smart.csv`, `results_smart_outage.csv` |
| 3 | `python forecast.py` | `forecast_metrics.csv`, `forecast_demand.csv`, `forecast_price.csv`, `forecast_solar.csv`, `forecast_wind.csv` |
| 4 | `python eda.py` | `eda_plots.png` (needs step 3) |
| 5 | `python tune.py` | `tuning_results.csv`, `tuning_best.csv` |
| 6 | `streamlit run app.py` | the dashboard (reads the files above) |

`tune.py` changes the price threshold in memory only. The default in `simulator.py` is `LOW_PRICE = df["price"].quantile(0.40)`, the value the tuning picked.

## Dashboard

- **KPI cards:** net cost, carbon and curtailment, smart vs baseline.
- **Charts:** cumulative net cost and battery state of charge.
- **Operator alert box:** for example `Battery B1 maintenance on Jan 12 for 2 days`.
- **Storm alert box:** `Storm on Jan 12 from 18:00 to 22:00`. A second box takes updates: `stop at 19:30` or `extended until 01:00`.
- **Decision log and explainer:** pick a 15-minute step and click Explain Decision.
- **Forecast vs actual, tuning table, data exploration.**

Type times as `HH:MM` (for example `18:00` or `9:00 PM`). Times written like `6 pm` are not recognised.

## Output columns (`results_*.csv`)

`time, generation, demand, price, charged, discharge, curtailed, bought, B1_soc, B2_soc, reason, grid_cost, grid_revenue, sold, carbon_kg`

- `charged`, `discharge`, `bought`, `sold` are what actually happened, not what the controller requested.
- The `reason` text records the final outcome, for example `batteries full, surplus sold`.

## How the agent decides

- **Baseline:** surplus charges the batteries, deficit discharges them, leftover surplus is sold. It ignores price.
- **Smart agent:**
  - Rule A: on a deficit at a cheap price, buy from the grid and keep the battery charge.
  - Rule B: at a cheap price with no surplus and room in the batteries, charge from the grid.
  - Storm alert: before the storm, stop draining batteries and top them up. During the storm, keep a 40% reserve floor and buy the deficit instead.
- **Safety check:** every step, generation + discharge + grid buy = demand + charge + grid sell + curtailment. A warning prints if it fails. None appears in any run.

## Limitations

- Synthetic data. Net cost can be negative because the market has no fees and an unlimited grid.
- Carbon is counted on grid purchases only.
- The policy is rule-based and tuned by search. There is no MPC and no demand response.
- The live dashboard uses a fixed price threshold. The forecast-driven threshold is a separate experiment on the last 6 days.
- Alert times are supplied before the run. There is no live mid-run replanning.
- Forecast p10 to p90 bands are overconfident (coverage 57 to 85% against an 80% target).
- The LLM explains decisions after the fact. It does not make them.

## Future work

MPC with probabilistic forecasts, a live 15-minute step mode, grid export and line-limit events, calibrated forecast bands, real weather and market data, demand response and load shifting.