import numpy as np
import pandas as pd
import lightgbm as lgb

steps_per_day = 96       # 96 * 15min = 24h
test_days = 6           # last 6 days are the test set (first 24 days = for training purpose)

signals = ['demand', 'price', 'solar', 'wind']

def load_signals(path = 'data.csv'):
    
    """Read data.csv and build 4 clean signals: demand, price, total solar, total wind."""

    raw = pd.read_csv(path)
    out = pd.DataFrame()
    out['time'] = pd.to_datetime(raw['time'])
    out['demand'] = raw['demand_mw']
    out['price'] = raw['price']
    out['solar'] = sum(raw[f'solar{i}_mw'] for i in range(1,6))
    out['wind'] = sum(raw[f'wind{i}_mw'] for i in range(1, 4))
    return out

def build_features(data, signal):

    """Feature for one signal. Every lag is at least 96 steps(24h) old."""

    df = pd.DataFrame(index=data.index)
    min_of_day = data['time'].dt.hour * 60 + data['time'].dt.minute

    df['sin_t'] = np.sin(2 * np.pi * min_of_day / 1440)
    df["cos_t"] = np.cos(2 * np.pi * min_of_day / 1440)

    df["hour"] = data["time"].dt.hour
    
    df["lag_24h"] = data[signal].shift(steps_per_day)
    df["lag_48h"] = data[signal].shift(2 * steps_per_day)
    df["lag_24h_smooth"] = data[signal].shift(steps_per_day).rolling(4).mean()
    return df

def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def forecast_signal(data, signal):
    """Train on the first days, predict the last TEST_DAYS days. Returns a DataFrame."""
    
    X = build_features(data, signal)
    y = data[signal]

    valid = X.dropna().index                       # first 2 days have no lag_48h
    split = len(data) - test_days * steps_per_day
    train_idx = [i for i in valid if i < split]
    test_idx = [i for i in valid if i >= split]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                  min_child_samples=20, verbose=-1)

    mean_model = lgb.LGBMRegressor(objective="regression", **params).fit(X_train, y_train)
    p10_model = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **params).fit(X_train, y_train)
    p90_model = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **params).fit(X_train, y_train)

    result = pd.DataFrame({
        "time": data.loc[test_idx, "time"].values,
        "actual": y_test.values,
        "naive": X_test["lag_24h"].values,         # "same as yesterday"
        "lightgbm": mean_model.predict(X_test),
        "p10": p10_model.predict(X_test),
        "p90": p90_model.predict(X_test),
    })
    # solar and wind can't be negative
    if signal in ("solar", "wind"):
        for c in ["lightgbm", "p10", "p90"]:
            result[c] = result[c].clip(lower=0)
    result["p10"] = np.minimum(result["p10"], result["lightgbm"])   # keep band ordered
    result["p90"] = np.maximum(result["p90"], result["lightgbm"])
    return result

def run_all(path="data.csv"):

    """Forecast every signal. Returns (dict of forecast DataFrames, metrics DataFrame)."""
    
    data = load_signals(path)
    forecasts, rows = {}, []
    for s in signals:
        r = forecast_signal(data, s)
        forecasts[s] = r
        coverage = float(((r["actual"] >= r["p10"]) & (r["actual"] <= r["p90"])).mean())
        rows.append({
            "signal": s,
            "naive_MAE": mae(r["actual"], r["naive"]),
            "lgbm_MAE": mae(r["actual"], r["lightgbm"]),
            "naive_RMSE": rmse(r["actual"], r["naive"]),
            "lgbm_RMSE": rmse(r["actual"], r["lightgbm"]),
            "improvement_%": 100 * (1 - mae(r["actual"], r["lightgbm"]) / mae(r["actual"], r["naive"])),
            "p10_p90_coverage_%": 100 * coverage,     # should be near 80
        })
    return forecasts, pd.DataFrame(rows).round(2)


if __name__ == "__main__":
    forecasts, metrics = run_all()
    print(metrics.to_string(index=False))
    metrics.to_csv("forecast_metrics.csv", index=False)
    for s, r in forecasts.items():
        r.to_csv(f"forecast_{s}.csv", index=False)
    print("\nSaved forecast_metrics.csv and forecast_<signal>.csv")