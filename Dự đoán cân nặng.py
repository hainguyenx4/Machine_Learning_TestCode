
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("weight-height.csv")
dataset.head()

#Lấy và phân loại dữ liệu chiều cao và cân nặng trong dataset 
height = dataset.Height
weight = dataset.Weight

print("Cân nặng trung bình: ", height.mean())#tính và in ra cân nặng trung bình
print("Chiều cao trung bình: ", weight.mean())#tính và in ra chiều cao trung bình

#Chuyển đổi giới tính từ dạng chuỗi sang số
dataset['Gender'].replace('Female',0, inplace=True)#chuyển đổi nữ(female) thành 0
dataset['Gender'].replace('Male',1, inplace=True)#chuyền đổi nam(male) thành 1

#Chia data thành 2 phần: training data và test data
X = dataset.iloc[:, :-1].values#X là mảng giá trị của tất cả các hàng và cột trừ cột cuối(-1) 
y = dataset.iloc[:, 2].values#y là mảng giá trị của cột cuối (weight)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Sử dụng LinearRegression để đào tạo, thuật toán LenearRegression thường đưa ra dự đoán có đầu ra là giá trị thực
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Dự đoán cân nặng dựa vào chiều cao
chieu_cao= input("Nhập chiều cao tính theo inch:")
gioi_tinh= input("Nhập giới tính:")
if "Nam" == gioi_tinh: 
	your_weight_pred = lin_reg.predict([[0,float(chieu_cao)]])
	print('Kết quả dự đoán cân nặng của bạn = ', your_weight_pred,'pound')
if "Nữ" == gioi_tinh:
	your_weight_pred = lin_reg.predict([[1,float(chieu_cao)]])
	print('Kết quả dự đoán cân nặng của bạn = ',your_weight_pred, 'pound')









