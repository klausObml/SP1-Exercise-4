import numpy as np
import matplotlib.pyplot as plt

x = np.load("./x.npy")
y = np.load("./y.npy")

print(x.shape)

# formulate LS problem

y_hat = np.log(y)
A = np.zeros([x.shape[0],3])
print(A.shape)

# fill A with the values of the powers of x
for i in range(len(x)):
    A[i][0] = x[i]**2
    A[i][1] = x[i]
    A[i][2] = 1

# c = pseudoinverse of A times x
c = np.linalg.inv(A.T @ A) @ A.T @ y_hat
c_numpy = np.linalg.pinv(A) @ y_hat


# revert to gaussian pulse
mu = c[1]/2
sigma_squared = -2*c[0]
beta = np.exp(c[2]+(mu**2/(2*sigma_squared)))

# plot the results
x = np.linspace(1,8,100)
y = beta * np.exp(-1/(2*sigma_squared) * (x-mu)**2)

fig = plt.figure()
plt.plot(x,y)
plt.grid()
plt.savefig("./ex4_1_1.png")
plt.show()