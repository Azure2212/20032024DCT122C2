import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc tệp HDF5 vào DataFrame
df = pd.read_hdf("/kaggle/input/dataset/rafdb_basic.hdf5")

df.loc[df['type']== 'train','type']=0
df.loc[df['type']== 'test','type']=1
# Chọn các cột quan trọng
selected_columns = ['gender', 'age', 'emotion', 'race', "type"]

# Kiểm tra tính tồn tại của các cột trong DataFrame
if all(col in df.columns for col in selected_columns):
    # Lựa chọn các cột
    df_train = df[selected_columns]

    # Thiết lập kích thước hình vẽ và kiểu dáng seaborn
    sns.set(style="whitegrid")# nền chart
    plt.figure(figsize=(8, 10)) #vẽ mới với kích thước là 20x10 inches

    # Tạo pairplot với phân loại màu theo từng cột
    # hue='gender' - biểu đồ scatterplot với phân loại theo gender
    # palette='Set1' - nhóm màu
    pairplot = sns.pairplot(df_train, hue='gender', palette='Set1',kind='scatter', diag_kind='kde',plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

    # Thêm tiêu đề và điều chỉnh bố cục
    plt.suptitle('Correlogram Chart', y=1)
    plt.tight_layout() # ham dieu chinh tu dong title,chú thích, khoảng cách

    # Hiển thị biểu đồ
    plt.show()
else:
    print("One or more selected columns are not present in the DataFrame.")
