import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('WMT.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create additional columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Daily Range'] = df['High'] - df['Low']

# 1. Line Plot of Closing Prices Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'])
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.savefig('line_closing_prices.png')

# 2. Scatter Plot of Opening vs. Closing Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Open', y='Close', data=df)
plt.title('Opening vs. Closing Prices')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.savefig('scatter_open_vs_close.png')

# 3. Histogram of Daily Trading Volume
plt.figure(figsize=(8, 6))
sns.histplot(df['Volume'], bins=30, kde=True)
plt.title('Distribution of Daily Trading Volume')
plt.xlabel('Volume')
plt.savefig('histogram_volume.png')

# 4. Box Plot of Closing Prices by Year
plt.figure(figsize=(12, 6))
sns.boxplot(x='Year', y='Close', data=df)
plt.title('Closing Prices by Year')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.savefig('boxplot_yearly_close.png')

# 5. Scatter Plot of Daily High vs. Daily Low Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Low', y='High', data=df)
plt.title('Daily High vs. Low Prices')
plt.xlabel('Low Price')
plt.ylabel('High Price')
plt.savefig('scatter_high_vs_low.png')

# 6. Line Plot of Adjusted Closing Prices Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Adj Close'])
plt.title('Adjusted Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.savefig('line_adj_closing_prices.png')

# 7. Histogram of Daily Price Range (High - Low)
plt.figure(figsize=(8, 6))
sns.histplot(df['Daily Range'], bins=30, kde=True)
plt.title('Distribution of Daily Price Range')
plt.xlabel('Price Range (High - Low)')
plt.savefig('histogram_price_range.png')

# 8. Box Plot of Opening Prices by Month
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Open', data=df)
plt.title('Opening Prices by Month')
plt.xlabel('Month')
plt.ylabel('Opening Price')
plt.savefig('boxplot_monthly_open.png')

# 9. Scatter Plot of Volume vs. Closing Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Volume', y='Close', data=df)
plt.title('Volume vs. Closing Prices')
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.savefig('scatter_volume_vs_close.png')

# 10. Line Plot of Monthly Average Closing Prices Over Time
monthly_avg_close = df.resample('M', on='Date')['Close'].mean()
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg_close)
plt.title('Monthly Average Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Monthly Avg Closing Price')
plt.savefig('line_monthly_avg_close.png')

# 11. Box Plot of Closing Prices by Quarter
plt.figure(figsize=(12, 6))
sns.boxplot(x='Quarter', y='Close', data=df)
plt.title('Closing Prices by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Closing Price')
plt.savefig('boxplot_quarterly_close.png')

# 12. Line Plot of Yearly Average Trading Volume Over Time
yearly_avg_volume = df.resample('Y', on='Date')['Volume'].mean()
plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_volume)
plt.title('Yearly Average Trading Volume Over Time')
plt.xlabel('Year')
plt.ylabel('Yearly Avg Volume')
plt.savefig('line_yearly_avg_volume.png')