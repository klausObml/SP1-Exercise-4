import numpy as np
import matplotlib.pyplot as plt

x = np.load("./x.npy")
y = np.load("./y.npy")

#print(x.shape)

# formulate LS problem

y_hat = np.log(y)
A = np.zeros([x.shape[0],3])
#print(A.shape)

# fill A with the values of the powers of x
for i in range(len(x)):
    A[i][0] = x[i][0]**2
    A[i][1] = x[i][0]
    A[i][2] = 1

# c = pseudoinverse of A times x
c = np.linalg.inv(A.T @ A) @ A.T @ y_hat
c_numpy = np.linalg.pinv(A) @ y_hat

#print(c)

# revert to gaussian pulse
mu = -0.5*c[1]/c[0]
#print("mu: " + str(mu))
sigma_squared = -1/(2*c[0])
#print("sigma^2: " + str(sigma_squared))
beta = np.exp(c[2]+(mu**2/(2*sigma_squared)))

# plot the results
x_hat = np.linspace(np.min(x),np.max(x),100)
y_of_x = beta * np.exp(-1/(2*sigma_squared) * (x_hat-mu)**2)

fig = plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y, label='Data')
plt.plot(x_hat,y_of_x, label='Interpolation',c='orange')
plt.grid()
plt.legend()
plt.savefig("./ex4_1_1.png")
plt.show()