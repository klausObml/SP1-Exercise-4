import numpy as np
import matplotlib.pyplot as plt

# define plotting domain
x = np.linspace(-1,1,100)

# define basis functions
q0 = np.poly1d([0,0,0,1/np.sqrt(2)])
q1 = np.poly1d([0,0,np.sqrt(3/2),0])
q2 = np.poly1d([0,np.sqrt(45/8),0,np.sqrt(45/8)*(-1/3)])
q3 = np.poly1d([np.sqrt(175/8),0,np.sqrt(175/8)*(-3/5),0])

# define original function and approximation coefficients
f1 = np.poly1d([5,3,-4,1])
c1 = np.array([2*np.sqrt(2),-np.sqrt(2/3),2*np.sqrt(2/5),2*np.sqrt(2/7)])

f2 = np.poly1d([1,0,0,0,0,0])
c2 = np.array([0,np.sqrt(6)/7,0,4/9*np.sqrt(2/7)])

f3 = np.sin(np.pi*x)
# np.sqrt(14)*((np.pi)**2 -15)/np.pi**3
# np.sqrt(6)/np.pi
c3 = np.array([0,0.77970,0,-0.61911])

# define approximation functions
f1_hat = np.poly1d(c1[0]*q0) + np.poly1d(c1[1]*q1) + np.poly1d(c1[2]*q2) + np.poly1d(c1[3]*q3)
f2_hat = np.poly1d(c2[0]*q0) + np.poly1d(c2[1]*q1) + np.poly1d(c2[2]*q2) + np.poly1d(c2[3]*q3)
f3_hat = np.poly1d(c3[0]*q0) + np.poly1d(c3[1]*q1) + np.poly1d(c3[2]*q2) + np.poly1d(c3[3]*q3)

# plot f1
plt.figure()
plt.plot(x, f1(x), linewidth=5, label=r'$f_1(x)$')
plt.plot(x, f1_hat(x), label=r"$\hat{f}_1(x)$")
plt.grid()
plt.legend()
plt.savefig("ex4_2_f1.png")

# plot f2
plt.figure()
plt.plot(x, f2(x), label=r'$f_2(x)$')
plt.plot(x, f2_hat(x), label=r"$\hat{f}_2(x)$")
plt.grid()
plt.legend()
plt.savefig("ex4_2_f2.png")

# plot f3
plt.figure()
plt.plot(x, f3, linewidth = 5, label=r'$f_3(x)$')
plt.plot(x, f3_hat(x), label=r"$\hat{f}_3(x)$")
plt.grid()
plt.legend()
plt.savefig("ex4_2_f3.png")


plt.show()
