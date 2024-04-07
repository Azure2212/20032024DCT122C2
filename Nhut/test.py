import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)
collectn_1 = np.random.normal(100, 10, 200)
collectn_2 = np.random.normal(80, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(70, 25, 200)

# Combine these different collections into a list
data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]

# Create a figure instance and specify the size
fig, ax = plt.subplots(figsize=(10, 6))

# Create the violin plot
ax.violinplot(data_to_plot)

# Add labels and title
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Data 1', 'Data 2', 'Data 3', 'Data 4'])
ax.set_xlabel('Dataset')
ax.set_ylabel('Values')
ax.set_title('Violin Plot Example')

plt.show()
