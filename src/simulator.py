from battery import Battery
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

LOW_PRICE = df["price"].quantile(0.40)   # "cheap" price for the smart controller
print(LOW_PRICE)

USE_FORECAST = False
FORECAST = np.full(len(df), np.nan)

STORM = None
RESERVE_FRACTION = 0.4

def load_forecast(path="forecast_price.csv"):
    global FORECAST
    f = pd.read_csv(path)
    m = dict(zip(f["time"].astype(str), f["lightgbm"]))
    FORECAST = df["time"].map(m).to_numpy(dtype=float)

def baseline_controller(row, batteries):
    """
    A simple controller that decides how to charge/discharge batteries based on the energy surplus or deficit, meaning 
    the difference between generation and demand. If there is a surplus, it will try to charge the batteries, 
    and if there is a deficit, it will try to discharge the batteries.
    """

    solar_total = sum(getattr(row, f'solar{i}_mw') for i in range(1, 6))    # solar_total is the total energy produced by the solar farms, measured in MW
    wind_total = sum(getattr(row, f'wind{i}_mw') for i in range(1, 4))    # wind_total is the total energy produced by the wind farms, measured in MW
        
    generation = solar_total + wind_total  # generation is the amount of energy that is produced by the solar and wind farms
    # print('generation:', generation)
    
    demand_mw = row.demand_mw   # demand_mw is the amount of energy that is needed to meet the demand of the grid
    # print('demand:', demand_mw)
    
    surplus = generation - demand_mw     # surplus means the amount of energy that can be stored in batteries or sold to the grid
    # print('surplus:', surplus)

    # controller decision 
    decision = {
        "generation": generation,
        "battery_charge_mw": 0,
        "battery_discharge_mw": 0,
        "grid_charge_mw": 0,
        "grid_buy_mw": 0,
        "grid_sell_mw": 0,
        "curtail_mw": 0,
        "reason": "do nothing",
    }

    # Controller decide what to do
    if surplus > 0:          # if there is a surplus of energy, we will try to charge the batteries with the surplus energy.
            decision["battery_charge_mw"] = surplus
            decision["reason"] = f"surplus, charge battery"
    elif surplus < 0:       # if there is a deficit of energy, we will try to discharge the batteries to cover the deficit.
            decision["battery_discharge_mw"] = -surplus
            decision["reason"] = f"deficit, discharge battery"
    else:
            decision["reason"] = f"balanced, do nothing"

    return decision


def smart_controller(row, batteries):

    """
    A smarter controller that takes into account the price of electricity and the state of charge of the batteries.
    It will try to buy electricity from the grid when the price is low, and sell electricity to the grid when the price is high.
    It will also try to keep the batteries at a reasonable state of charge, so that they can be used to store energy when the price is low, 
    and discharge energy when the price is high. Basically it try to decide what to do.
    """

    # start from the baseline decision, then change it using price and battery level
    decision = baseline_controller(row, batteries)
    surplus = decision["generation"] - row.demand_mw
    room = sum(b.capacity - b.soc for b in batteries if b.available)

    low = LOW_PRICE
    window = FORECAST[row.Index+1 : row.Index+97]      # next 24h of forecast prices
    use_fc = USE_FORECAST and len(window) >= 48 and not np.isnan(window).any()
    if use_fc:
        low = np.percentile(window, 40)

    # Rule A: deficit but price is cheap -> buy from grid, keep battery charge for later
    if surplus < 0 and row.price <= low:
        decision["grid_buy_mw"] = -surplus
        decision["battery_discharge_mw"] = 0
        decision["reason"] = "deficit, price cheap, buy grid and save battery"

    # Rule B: price is cheap and batteries have room -> top up from grid
    if row.price <= low and room > 1 and surplus <= 0 and (not use_fc or window.max() >= row.price * 1.12):
        decision["grid_charge_mw"] = sum(b.max_rate for b in batteries if b.available)
        decision["reason"] += " | cheap price, charge from grid"
    
    if STORM:
        t = pd.Timestamp(row.time)
        stop = pd.Timestamp(STORM.get("cancelled_at", "2100-01-01"))
        
        if STORM and t < stop:
            tym = pd.Timestamp(row.time)  # represent current simulation time
            start_time = pd.Timestamp(STORM['start'])     # storm start time
            end_time = pd.Timestamp(STORM.get('cancelled_at', STORM['end']))     # storm end time

            # before the strom: fill the batteries from the grid, whatever the price
            if start_time - pd.Timedelta(hours=12) <= tym < start_time and room > 1 and surplus <= 0:
                decision["grid_buy_mw"] = -surplus
                decision["battery_discharge_mw"] = 0
                decision['grid_charge_mw'] = sum(b.max_rate for b in batteries if b.available)
                decision['reason'] = "storm prep: buy deficit, keep battery"

            # during the storm: keep the reserve, buy instead of discharging below it
            if start_time <= tym < end_time:
                stored = sum(b.soc for b in batteries)
                reserve = RESERVE_FRACTION * sum(b.capacity for b in batteries)
                need = -surplus * 0.25
                if surplus < 0 and stored - need <= reserve:
                    decision['grid_buy_mw'] -= surplus
                    decision['battery_discharge_mw'] = 0
                    decision['reason'] = 'storm: reserve protected, buy from grid'

    return decision

def run_simulation(controller, SELL_PRICE_MIN=00, outage = None, start=0):

    """
    this function runs the simulation by iterating over each row of the input data, applying the controller, 
    and updating the state of charge of the batteries accordingly. So that we will keep track of the energy balance 
    and any curtailment or grid purchases that occur.
    """

    b1 = Battery("B1")
    b2 = Battery("B2", capacity_mwh=200, max_rate_mw=50, efficiency=0.9)
    batteries = [b1, b2]
    records = []
    

    # one pass per 15-minute step
    for row in df.iloc[start:].itertuples():

        if outage is not None:
            b1.available = not (outage["start"] <= row.time < outage["end"])

        decision = controller(row, batteries)

        remaining_charge = decision["battery_charge_mw"]  # remaining energy to be charged
        total_charged = 0
        for battery in batteries:
            actual = battery.charge(remaining_charge)
            total_charged += actual
            remaining_charge -= actual

        remaining_discharge = decision['battery_discharge_mw']   # simulator ask batteries to provide charge or energy(MW)
        total_discharged = 0
        for battery in batteries:
            actual = battery.discharge(remaining_discharge)
            total_discharged += actual
            remaining_discharge -= actual

        # charge from grid (separate loop, so it never mixes with the surplus leftover)
        remaining_grid_charge = decision["grid_charge_mw"]
        total_grid_charged = 0
        for battery in batteries:
            actual = battery.charge(remaining_grid_charge)
            total_grid_charged += actual
            remaining_grid_charge -= actual
        total_charged += total_grid_charged

        # --- leftovers: set EVERY row, so they always exist ---
        curtailed = remaining_charge       # surplus the batteries couldn't take
        bought = remaining_discharge + decision["grid_buy_mw"] + total_grid_charged       # shortage the batteries couldn't cover, remaining amount is purchased from the grid.
        sold = 0                           # no selling rule yet

        if remaining_charge > 0:
            if row.price >= SELL_PRICE_MIN:
                sold = remaining_charge
                curtailed = 0

        # --- energy balance check ---
        supply = decision["generation"] + total_discharged + bought
        use = row.demand_mw + total_charged + sold + curtailed
        if abs(supply - use) > 0.01:
            print("BALANCE WARNING", row.time, round(supply - use, 4))

        grid_cost = bought * row.price * 0.25
        grid_revenue = sold * row.price * 0.25

        if decision["battery_charge_mw"] > 0 and total_charged < 0.01:
            if sold > 0:
                decision["reason"] += " | batteries full, surplus sold"
            elif curtailed > 0:
                decision["reason"] += " | batteries full, surplus curtailed"
                
        if decision["battery_discharge_mw"] > 0 and total_discharged < 0.01 and bought > 0:
            decision["reason"] += " | batteries empty, bought from grid"

        # for every 15 min interval, we stored:
        dic = {
            'time': row.time,
            'generation': decision['generation'],
            'demand': row.demand_mw,
            'price': row.price,
            'charged': total_charged,
            'discharge': total_discharged,
            'curtailed': curtailed,
            'bought': bought,
            'B1_soc': b1.soc,
            'B2_soc': b2.soc,
            'reason': decision['reason'],
            'grid_cost': grid_cost,
            'grid_revenue': grid_revenue,
            'sold': sold,
            'carbon_kg': bought * 0.25 * 700,   # NEW: 700 kg CO2 per MWh of grid power
        }
        records.append(dic)

    return pd.DataFrame(records)


def print_totals(name, df_results):
    total_cost = df_results["grid_cost"].sum()
    total_revenue = df_results["grid_revenue"].sum()
    print(name)
    print("  Total Cost:", round(total_cost))
    print("  Total Revenue:", round(total_revenue))
    print("  Net Cost:", round(total_cost - total_revenue))
    print("  Curtailed MWh:", round(df_results["curtailed"].sum() * 0.25))
    print("  Carbon tonnes:", round(df_results["carbon_kg"].sum() / 1000))


if __name__ == "__main__":
    base = run_simulation(baseline_controller)
    smart = run_simulation(smart_controller)
    base.to_csv("results_baseline.csv", index=False)
    smart.to_csv("results_smart.csv", index=False)

    print_totals("BASELINE", base)
    print_totals("SMART", smart)

    outage = {"start": "2026-01-10 00:00:00", "end": "2026-01-11 00:00:00"}
    smart_out = run_simulation(smart_controller, outage=outage)
    # smart_out.to_csv("results_smart_outage.csv", index=False)
    print_totals("SMART + B1 OUTAGE", smart_out)