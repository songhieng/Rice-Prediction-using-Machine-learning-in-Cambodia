import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('global_food_prices.csv')

# Ensure 'Date' column is in the right format
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))

# Filter for 'Rice' in 'Phnom Penh'
rice_data = df[(df['Product_Name'] == 'Rice (mixed, low quality) - Wholesale')
               & (df['CITY'] == 'Phnom Penh')]

# Ensure data is sorted by date
rice_data = rice_data.sort_values(by='Date')

# Plot Price_USD over time
plt.figure(figsize=(12, 6))
plt.plot(rice_data['Date'], rice_data['Price_KHR'], marker='o')
plt.xlabel('Date')
plt.ylabel('Price KHMR')
plt.title('Price of Rice in Phnom Penh Over Time')
plt.show()

# Calculate correlation matrix
corr_matrix = df.corr()

# Create a heatmap
fig = px.imshow(corr_matrix)

# Show the plot
fig.show()
