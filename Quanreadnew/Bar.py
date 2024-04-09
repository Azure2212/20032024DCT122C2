import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.abspath(globals().get("__file__","./"))
root_dir = os.path.abspath(f'(current_dir)/../../input/emotion/rafdb_basic.hdf5')

df = pd.read_hdf(root_dir)
train_df = df[df['type']=='train']
#print(train_df)
#-----------------------------------emotion----------------------------#
emotion_counts = train_df['emotion'].value_counts()
emotion_descriptions ={
    1:'Surprise', 2: 'Fear', 3: 'Disgust', 4:'Happiness', 5:'Sadness', 6: 'Anger', 7: 'Neutral'
}
plt.figure(figsize = (8 , 9))
bars = plt.bar(emotion_counts.index,emotion_counts.values)

plt.xlabel('Những loại cảm xúc')
plt.ylabel('Số lượng')
plt.title('Thống kê trong Train')

for i,bar in enumerate(bars):#duyệt qua từng cột trong biểu đồ
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height,
             f'{height}\n{emotion_descriptions[emotion_counts.index[i]]}',
             ha='center', va='bottom')

plt.show()
#-----------------------------------emotion----------------------------#
#bar.get_x() + bar.get_width() / 2: vị trí trung tâm của cột
#height: Chiều cao của cột, là số lượng tương ứng với loại cảm xúc.
#f'{height}\n{emotion_descriptions[emotion_counts.index[i]]}': Output
#   {height}: Hiển thị số lượng (chiều cao của cột).
#   \n:dấu xuống dòng