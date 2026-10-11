import numpy as np, pandas as pd

np.random.seed(42)

# 30 days, one row every 15 minutes
time = pd.date_range("2026-01-01", periods=30*96, freq="15min")
hour = time.hour + time.minute / 60
n = len(time)

df = pd.DataFrame({"time": time})

# 5 solar farms: bell shape during the day, reduced by random clouds
for i in range(1, 6):
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    clouds = np.random.uniform(0.6, 1.0, n)
    df[f"solar{i}_mw"] = 50 * sun * clouds

# 3 wind farms: random but smooth wind
for i in range(1, 4):
    wind = pd.Series(np.random.normal(0, 1, n)).rolling(60, min_periods=1).mean()
    wind = (wind - wind.min()) / (wind.max() - wind.min())
    df[f"wind{i}_mw"] = 60 * wind

# Demand: high in the day, low at night
df["demand_mw"] = 150 + 60 * np.sin((hour - 8) / 24 * 2 * np.pi) + np.random.normal(0, 5, n)

# Price: morning and evening peaks, plus random spikes
price = 40 + 25 * np.sin((hour - 6) / 24 * 4 * np.pi)
spikes = np.random.rand(n) < 0.01
df["price"] = price + spikes * 100 + np.random.normal(0, 3, n)

df.to_csv("data.csv", index=False)
print(df.head())