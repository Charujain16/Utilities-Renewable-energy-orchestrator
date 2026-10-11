import streamlit as st
import pandas as pd
import simulator
from llm import parse_alert, explain, parse_storm

st.set_page_config(page_title="Renewable Energy Orchestrator", layout="wide")
st.title("Renewable Energy Orchestrator")
st.caption("Rule-based baseline vs price-aware smart agent, same data, same batteries")


@st.cache_data
def cached_parse(text):
    return parse_alert(text)


@st.cache_data
def run_both(sell_min, o_start, o_end, storm):
    outage = None if o_start is None else {"start": o_start, "end": o_end}
    simulator.STORM = storm          # set INSIDE the cached function
    base = simulator.run_simulation(simulator.baseline_controller, SELL_PRICE_MIN=sell_min, outage=outage)
    smart = simulator.run_simulation(simulator.smart_controller, SELL_PRICE_MIN=sell_min, outage=outage)
    return base, smart


# ---------- sidebar: scenario controls ----------
st.sidebar.header("Scenario")
sell_min = st.sidebar.slider("Minimum sell price", 0, 80, 0)

o_start, o_end = None, None

# manual outage
if st.sidebar.checkbox("Battery B1 outage"):
    day = st.sidebar.slider("Outage start day (Jan)", 1, 30, 10)
    length = st.sidebar.slider("Outage length (days)", 1, 10, 3)
    start = pd.Timestamp(f"2026-01-{day:02d}")
    o_start = start.strftime("%Y-%m-%d %H:%M:%S")
    o_end = (start + pd.Timedelta(days=length)).strftime("%Y-%m-%d %H:%M:%S")

# text alert (overrides the manual outage)
alert = st.sidebar.text_input("Operator alert",
                              placeholder="e.g. Battery B1 maintenance on Jan 10 for 2 days")
if alert.strip():
    parsed = cached_parse(alert)
    if parsed:
        st.sidebar.success("Outage detected")
        st.sidebar.json(parsed)
        o_start, o_end = parsed["start"], parsed["end"]
    else:
        st.sidebar.warning("No outage detected")

storm = None
storm_text = st.sidebar.text_input("Storm alert",
                                   placeholder="e.g. Storm on Jan 12 from 18:00 to 22:00")
if storm_text.strip():
    storm = parse_storm(storm_text)
    if storm:
        update_text = st.sidebar.text_input("Storm update",
                                            placeholder="e.g. stop at 19:30  /  extended until 01:00")
        if update_text.strip():
            storm = parse_storm(update_text, current=storm) or storm
            if "cancelled_at" in storm:
                st.sidebar.warning(f"Storm cancelled at {storm['cancelled_at']}")
            else:
                st.sidebar.success("Storm active")
        st.sidebar.json(storm)
    else:
        st.sidebar.warning("No storm detected")



# ---------- run both controllers on the same scenario ----------
base, smart = run_both(sell_min, o_start, o_end, storm)
base["time"] = pd.to_datetime(base["time"])
smart["time"] = pd.to_datetime(smart["time"])


def kpis(r):
    return {
        "net_cost": r["grid_cost"].sum() - r["grid_revenue"].sum(),
        "carbon": r["carbon_kg"].sum() / 1000,
        "curtailed": r["curtailed"].sum() * 0.25,
    }


def pct(new, old):
    return f"{(new - old) / old * 100:.1f}% vs baseline"


kb, ks = kpis(base), kpis(smart)

# ---------- KPI cards ----------
c1, c2, c3 = st.columns(3)
c1.metric("Net cost (smart)", f"{ks['net_cost']:,.0f}",
          pct(ks["net_cost"], kb["net_cost"]), delta_color="inverse")
c2.metric("Carbon tonnes (smart)", f"{ks['carbon']:,.0f}",
          pct(ks["carbon"], kb["carbon"]), delta_color="inverse")
c3.metric("Curtailed MWh (smart)", f"{ks['curtailed']:,.0f}",
          f"baseline: {kb['curtailed']:,.0f}", delta_color="off")

st.info("Trade-off: the smart agent cuts cost by buying cheap grid power to charge batteries, "
        "which raises carbon. Cost and carbon compete, and the agent picks cost.")

# ---------- charts ----------
left, right = st.columns(2)
with left:
    st.subheader("Cumulative net cost")
    cost = pd.DataFrame({
        "baseline": (base["grid_cost"] - base["grid_revenue"]).cumsum().values,
        "smart": (smart["grid_cost"] - smart["grid_revenue"]).cumsum().values,
    }, index=smart["time"])
    st.line_chart(cost)
with right:
    st.subheader("Battery state of charge (smart)")
    st.line_chart(smart.set_index("time")[["B1_soc", "B2_soc"]])

# ---------- decision log ----------
st.subheader("Agent decision log")
days = sorted(smart["time"].dt.date.unique())
chosen = st.selectbox("Pick a day", days, index=min(9, len(days) - 1))
day_df = smart[smart["time"].dt.date == chosen]
st.dataframe(day_df[["time", "price", "generation", "demand",
                     "charged", "discharge", "bought", "sold", "reason"]],
             use_container_width=True, height=350)

# ---------- explain a decision ----------
st.subheader("Explain a Dispatch Decision")
row_index = st.selectbox("Pick a decision", day_df.index,
                         format_func=lambda i: f"{smart.loc[i, 'time']} - {smart.loc[i, 'reason']}")
if st.button("Explain Decision"):
    st.write(explain(smart.loc[row_index]))

# ---------- forecast, tuning, EDA ----------
st.subheader("Forecast vs actual (last 6 days)")
sig = st.selectbox("Signal", ["demand", "price", "solar", "wind"])
f = pd.read_csv(f"forecast_{sig}.csv", parse_dates=["time"]).set_index("time")
st.line_chart(f[["actual", "lightgbm", "p10", "p90"]])
st.dataframe(pd.read_csv("forecast_metrics.csv"))

st.subheader("Policy tuning: cost vs carbon")
st.dataframe(pd.read_csv("tuning_best.csv"))

st.subheader("Data exploration")
st.image("eda_plots.png")


# ---------- forecast-driven dispatch comparison ----------
@st.cache_data
def forecast_experiment():
    rows = []
    try:
        simulator.STORM = None                 # this experiment has no storm
        simulator.load_forecast()
        for name, ctrl, use in [("Baseline", simulator.baseline_controller, False),
                                ("Smart, fixed threshold", simulator.smart_controller, False),
                                ("Smart, forecast-driven", simulator.smart_controller, True)]:
            simulator.USE_FORECAST = use
            r = simulator.run_simulation(ctrl, start=2400)
            rows.append({"controller": name,
                         "net_cost": round(r["grid_cost"].sum() - r["grid_revenue"].sum()),
                         "carbon_tonnes": round(r["carbon_kg"].sum() / 1000)})
    finally:
        simulator.USE_FORECAST = False         # always switch it off again
    return pd.DataFrame(rows)


st.subheader("Forecast-driven agent vs fixed-rule agent (test days Jan 25-30)")
exp = forecast_experiment()
st.bar_chart(exp.set_index("controller")["net_cost"])
st.dataframe(exp, hide_index=True)
st.caption("A replay of 6 days the forecasting model did not see during training. "
           "Lower net cost is better: more money earned from sales than spent on buying power.")