from numpy import genfromtxt
from sklearn import linear_model

dataPath = r"C:\Users\Admin\Desktop\python\人工智能\多元线性回归\Delivery.csv"
deliveryData = genfromtxt(dataPath,delimiter=',')

print ("data")
print (deliveryData)

x= deliveryData[:,:-1]
y = deliveryData[:,-1]

print (x)
print (y)

lr = linear_model.LinearRegression()
lr.fit(x, y)
print (lr)

print("coefficients:")
print (lr.coef_)           #b1,b2,b3...bn
print("intercept:")
print (lr.intercept_)      #b0

xPredict = [[102,6]]
yPredict = lr.predict(xPredict)
print("predict:")
print (yPredict)
