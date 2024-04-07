import matplotlib.pyplot as plt
import os
import pandas as pd

current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'{current_dir}/../../../../input/emotion/rafdb_basic.hdf5')

data = pd.read_hdf(root_dir)
data_type = 'train'

# Age
data_age = [0, 1, 2, 3, 4]
count_age = []

for age in data_age:
    count_age.append(len(data[(data['age'] == age) & (data['type'] == data_type)]))

plt.bar(data_age, count_age, color='red')

for x, y in zip(data_age, count_age):
    plt.text(x, y, str(y), ha='center', va='bottom')

plt.xlabel('Age Dataset')
plt.ylabel('Number of Samples')
plt.title('Distribution of Samples From 0 to 4 in ' + data_type)
plt.show()

# Gender
data_gender = [0, 1, 2]
count_gender = []

for gender in data_gender:
    count_gender.append(len(data[(data['gender'] == gender) & (data['type'] == data_type)]))

plt.bar(data_gender, count_gender, color='blue')

for x, y in zip(data_gender, count_gender):
    plt.text(x, y, str(y), ha='center', va='bottom')

plt.xlabel('Gender Dataset')
plt.ylabel('Number of Samples')
plt.title('Distribution of Samples in ' + data_type)
plt.show()

# Emotion
data_emo = [1, 2, 3, 4, 5, 6, 7]
count_emo = []

for emo in data_emo:
    count_emo.append(len(data[(data['emotion'] == emo) & (data['type'] == data_type)]))

plt.bar(data_emo, count_emo, color='orange')

for x, y in zip(data_emo, count_emo):
    plt.text(x, y, str(y), ha='center', va='bottom')

plt.xlabel('Emotion Dataset')
plt.ylabel('Number of Samples')
plt.title('Distribution of Samples in ' + data_type)
plt.show()

# Race
data_race = [0, 1, 2]
count_race = []

for race in data_race:
    count_race.append(len(data[(data['race'] == race) & (data['type'] == data_type)]))

plt.bar(data_race, count_race, color='green')

for x, y in zip(data_race, count_race):
    plt.text(x, y, str(y), ha='center', va='bottom')

plt.xlabel('Race Dataset')
plt.ylabel('Number of Samples')
plt.title('Distribution of Samples in ' + data_type)
plt.show()
