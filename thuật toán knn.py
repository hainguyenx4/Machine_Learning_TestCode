import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lấy ngẫu nhiên 20 phần tử từ 0 tới 100 và gán nó kiểu số thực mảng hai chiều
trainData = np.random.randint(0, 100, (20, 2)).astype(np.float32)

# Lấy ngẫu nhiên 20 phần từ 0 và 1
ketqua = np.random.randint(0, 2, (20, 1)).astype(np.float32)
print('Out put:')
print (trainData)
print (ketqua)

# Lấy những phần tử có ketqua = 1 từ traindata
red = trainData[ketqua.ravel() == 1]

# Lấy những phần tử có ketqua = 0 từ traindata
blue = trainData[ketqua.ravel() == 0]
	
#Lấy ngẫu nhiên một một phần tử trong khoảng từ 0 và 100, đây là phần tử mới thêm ngẫu nhiên vào trong đồ thị
newMember = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# Ta In kết quả ra như sau
print ('======Red ============')
print (red)

print ('======blue ============')
print (blue)

print ('======New member ============')
print (newMember)

plt.scatter(red[:, 0], red[:, 1], 100, 'r', 's')#lệnh vẽ các điểm màu đỏ
plt.scatter(blue[:, 0], blue[:, 1], 100, 'b', '^')# lệnh vẽ các điểm màu xanh
plt.scatter(newMember[:, 0], newMember[:, 1], 100, 'y', '*')# lệnh vẽ điểm mới
knn = cv2.ml.KNearest_create()# tạo một biến knn sử dụng thư viện ml trong cv2 áp dụng thuật toán K- nearest

# Huấn luyện với mảng khởi tạo, biến 0 và nhãn đã biết
knn.train(trainData, 0, ketqua)

# In kêt quả resut = 1 => vuông, 0 => tam giác
temp, result, nearest, distance = knn.findNearest(newMember, 3)

print("Result: {}\n".format(result))
print("Nearest: {}\n".format(nearest))
print("Distance: {}\n".format(distance))


plt.show()