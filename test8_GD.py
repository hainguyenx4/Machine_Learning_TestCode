
import math
import numpy as np 
import matplotlib.pyplot as plt

#Hàm được sử dụng để tính đạo hàm J(θ)=θ^2+5Sin(θ)
def derivative(x):
    return 2*x+ 5*np.cos(x)

#Hàm được sử dụng để tính giá trị của hàm số J(θ)
def cost(x):
    return x**2 + 5*np.sin(x)

#Hàm dùng để thực hiện thuật toán Gardient Descent 
def myGD1(alpha, x0):#đầu vào của hàm số là learning rate và điểm bắt đầu
    x = [x0] #Gán x vào điểm bắt đầu
    for it in range(100):
        x_new = x[-1] - alpha*derivative(x[-1])#cập nhật điểm mới
        if abs(derivative(x_new)) < 1e-3:#Dừng thuật toán khi hàm có độ lớn đủ nhỏ
            break
        x.append(x_new)# cập nhập điểm tối ưu
    return (x, it)

(x1, it1) = myGD1(.1, -5)# Thử nghiệm với điểm khởi tạo x0=-5
(x2, it2) = myGD1(.1, 5)# Thử nghiệm với điểm khởi tạo x0=5
 

print ('Out put:')
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))#x[-1]=điểm cuối
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
