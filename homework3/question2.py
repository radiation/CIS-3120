import numpy as np

'''
I didn't bother with any functions or classes because the code is short and simple.
There's no need to even define a main function because the code is executed sequentially.
'''

# Headers & datatypes for the stock_prices.csv file: date, open, high, low, close, volume
dtype = [('date', 'U10'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'i8')]

# Load the data using the defined dtype; we need this so date is a string and not a float
stock_prices = np.genfromtxt('stock_prices.csv', delimiter=',', skip_header=1, dtype=dtype)

# Get the highest price ever
max_close = np.max(stock_prices['high'])
print(f"1. The highest closing price is: {max_close}")

# Find the date with the highest closing price
max_index_high = np.argmax(stock_prices['close'])
date_of_high = stock_prices['date'][max_index_high]
print(f"2. The date with the highest closing price is: {date_of_high}")

# Find the average daily trading volume
average_volume = np.mean(stock_prices['volume'])
print(f"3. The average volume is: {average_volume}")

# Find the standard deviation of the daily trading volume
std_dev_volume = np.std(stock_prices['volume'])
print(f"4. The standard deviation of the volume is: {std_dev_volume}")

# Find the number of days where the closing price > the opening price
num_days = np.sum(stock_prices['close'] > stock_prices['open'])
print(f"5. The number of days where the closing price is greater than the opening price is: {num_days}")

# Find the percentage of days where the closing price > the opening price
percentage_days = num_days / len(stock_prices) * 100
print(f"6. The percentage of days where the closing price > the opening price is: {percentage_days:.2f}%")