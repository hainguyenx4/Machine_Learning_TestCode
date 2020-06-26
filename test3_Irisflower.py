import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()# Lấy dữ liệu dataset IrisFlower
iris_X = iris.data# lấy dữ liệu hoa
iris_y = iris.target#lấy số nguyên in 0,1,2 đại diện cho mỗi phân lớp của Iris,một cách khác lấy tên hoa là iris.target_names
print ('Number of classes: %d' %len(np.unique(iris_y)))#số phân lớp
print ('Number of data points: %d' %len(iris_y))#số điểm dữ liệu

#Lấy một vài dữ liệu mẫu của lớp Setosar
X0 = iris_X[iris_y == 0,:]
print ('\nSamples from class 0(Setosar):\n', X0[:5,:])
#Lấy một vài dữ liệu mẫu của lớp Versicolor
X1 = iris_X[iris_y == 1,:]
print ('\nSamples from class 1(Versicolor):\n', X1[:5,:])
#Lấy một vài dữ liệu mẫu của lớp Verginica
X2 = iris_X[iris_y == 2,:]
print ('\nSamples from class 2(Verginica):\n', X2[:5,:])
#Lựa chọn 50 điểm dữ liệu ngẫu nhiên cho tập dữ liệu test, 100 điểm còn lại cho tập dữ liệu trainning.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print ("Training size: %d" %len(y_train))#In ra số lượng điểm dùng để training
print ("Test size    : %d" %len(y_test))#In ra số lượng điểm dùng để test

#Lấy k=1,với mỗi điểm test data, xét 1 điểm training data gần nhất và lấy label của điểm đó để dự đoán cho điểm test này.
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)# X_train là giá trị huấn luyện, y_train là giá trị đích
y_pred = clf.predict(X_test)#Dự đoán các nhẫn lớp mà dữ liệu cung cấp

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])

#hàm accuracy_score dùng để đánh giá độ chính xác của thuật toán KNN
from sklearn.metrics import accuracy_score
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))#lấy số điểm test_data dự đoán đúng chia cho tổng số lượng điểm trong tập test_data

#Tăng k lên thành 10 để tăng độ chính xác, kỹ thuật major voting
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)# X_train là giá trị huấn luyện, y_train là giá trị đích
y_pred = clf.predict(X_test)#Dự đoán các nhẫn lớp mà dữ liệu cung cấp

print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#Kỹ thuật đánh trọng số điểm lân cận
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')#parameter weights mang ý nghĩa đánh trọng số cao hơn vào những điểm gần với điểm cần xét=> độ tin tưởng cao hơn.
clf.fit(X_train, y_train)# X_train là giá trị huấn luyện, y_train là giá trị đích
y_pred = clf.predict(X_test)#Dự đoán các nhẫn lớp mà dữ liệu cung cấp

print ("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))


