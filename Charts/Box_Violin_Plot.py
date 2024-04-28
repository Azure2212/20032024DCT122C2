import numpy as np
import pandas as pd

# Generate random temperature data for each month
np.random.seed(0)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperature_data = {
    'Month': np.repeat(months, 10),  # Repeat each month 10 times
    'Temperature': np.random.normal(loc=20, scale=2, size=120)  # Random normal distribution for temperature
}

# Create a DataFrame from the generated data
temperature_df = pd.DataFrame(temperature_data)

# Display the first few rows of the DataFrame
print(temperature_df)


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the seaborn plots
sns.set(style="whitegrid")

# Create subplots with the specified size
plt.figure(figsize=(14, 6))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(x='Month', y='Temperature', data=temperature_df, palette='pastel')
plt.title('Box Plot of Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')

# Violin plot
plt.subplot(1, 2, 2)
sns.violinplot(x='Month', y='Temperature', data=temperature_df, palette='pastel')
plt.title('Violin Plot of Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')

plt.tight_layout()
plt.show()
