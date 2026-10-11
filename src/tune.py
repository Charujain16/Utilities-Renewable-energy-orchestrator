import simulator
import pandas as pd


results_list = []
quantiles = [0.1, 0.25, 0.4, 0.6]
carbon_prices = [0, 0.005, 0.01, 0.02, 0.05, 0.1]


for i in quantiles:
    
    price_value = simulator.df['price'].quantile(i)
    simulator.LOW_PRICE = price_value
    result = simulator.run_simulation(simulator.smart_controller)
    # print('result: ',result)
    net_cost = result['grid_cost'].sum() - result['grid_revenue'].sum()
    # print('net_cost: ', net_cost)
    carbon = result['carbon_kg'].sum()
    # print(f'carbon: {carbon}')

    for cp in carbon_prices:
        score = net_cost + carbon * cp
        dic = {
            'quantile' : i,
            'low_price':  simulator.LOW_PRICE,
            'carbon_price': cp,
            'net_cost': net_cost,
            'carbon': carbon,
            'score': score
        }
        results_list.append(dic)


table = pd.DataFrame(results_list)

best = table.loc[table.groupby("carbon_price")["score"].idxmin()]
print(best[["carbon_price", "quantile", "net_cost", "carbon", "score"]])

table.to_csv('tuning_results.csv', index=False)
best.to_csv('tuning_best.csv', index=False)