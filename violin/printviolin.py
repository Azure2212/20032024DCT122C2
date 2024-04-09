import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'(current_dir)/../../input/emotion/rafdb_basic.hdf5')

df = pd.read_hdf(root_dir)
#type_df=df[df['type']=='train']

#print(df.columns)


# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Emotion
sns.violinplot(x='type', y='emotion', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Violin Plot of Emotion')
axes[0, 0].set_xlabel('Type')
axes[0, 0].set_ylabel('Emotion')

# Gender
sns.violinplot(x='type', y='gender', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot of Gender')
axes[0, 1].set_xlabel('Type')
axes[0, 1].set_ylabel('Gender')

# Race
sns.violinplot(x='type', y='race', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot of Race')
axes[1, 0].set_xlabel('Type')
axes[1, 0].set_ylabel('Race')

# Age
sns.violinplot(x='type', y='age', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot of Age')
axes[1, 1].set_xlabel('Type')
axes[1, 1].set_ylabel('Age')

plt.tight_layout()
plt.show()