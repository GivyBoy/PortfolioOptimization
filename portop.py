import pandas as pd
from pandas_datareader import data as web
import numpy as np
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

try:
    data = input('Enter the stocks here: ').upper()
except:
    print('Enter the tickers(for example, AAPL for Apple) for the stocks.')

for datum in data:
    if datum == ',':
        data = data.replace(datum, '')
stocks = data.split()

stock_data = pd.DataFrame()

start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
for stock in stocks:
    try:
        stock_data[stock] = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
    except:
        print('Enter the tickers(for example, AAPL for Apple) for the stocks.')
        break
try:
    investment = int(input("Enter the amount you'd like to invest, here: "))
except:
    print('Enter a non-decimal value.')
weights = np.array([1 / len(stock_data.columns) for i in range(1, len(stock_data.columns) + 1)])

dsr = stock_data.pct_change()

acv = dsr.cov() * 252

port_var = np.dot(weights.T, np.dot(acv, weights))
print(port_var)

port_vol = np.sqrt(port_var)
print(port_vol)

apr = np.sum(dsr.mean() * weights) * 252

er = expected_returns.mean_historical_return(stock_data)
rm = risk_models.sample_cov(stock_data)

sr = EfficientFrontier(er, rm)

weights = sr.max_sharpe()

cleanWeights = sr.clean_weights()

latestPrices = get_latest_prices(stock_data)
weight = cleanWeights
da = DiscreteAllocation(weight, latestPrices, total_portfolio_value=investment)

allocation, leftover = da.lp_portfolio()

print(sr.portfolio_performance(verbose=True))
print(allocation)
print(round(leftover, 2))
