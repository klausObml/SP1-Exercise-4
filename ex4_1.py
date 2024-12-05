import numpy as np

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

print(c[-1])
print(c_numpy[-1])

