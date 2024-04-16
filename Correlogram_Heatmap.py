import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
n_samples = 100


np.random.seed(0)  
age = np.random.normal(loc=40, scale=10, size=n_samples)
education_level = np.random.randint(1, 5, size=n_samples)  #  1=High School, 2=Bachelor's, 3=Master's, 4=PhD
income = np.random.normal(loc=50000, scale=15000, size=n_samples)
spending = np.random.normal(loc=30000, scale=10000, size=n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Education Level': education_level,
    'Income': income,
    'Spending': spending
})
print(data)

# Plot the correlogram
sns.set(style='ticks')
sns.pairplot(data)
plt.suptitle('Correlogram', fontweight='bold', y=1.02)
plt.show()



#Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Matrix', fontweight='bold')
plt.show()
