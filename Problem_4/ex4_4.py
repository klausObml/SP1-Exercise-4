import numpy as np
import matplotlib.pyplot as plt

# 4.4.2 implement haar1D and ihaar1D

def haar1D (x : np.array) -> np.array:
    N = len(x)
    w = np.zeros(N)
    # lower half is computed as average
    for n in range (0, N//2):
        w[n] = (x[2*n] + x[2*n+1])/np.sqrt(2)
    # upper half is computed as difference
    for n in range (0, N//2):
        w[n+N//2] = (x[2*n] - x[2*n+1])/np.sqrt(2)
    return w

def ihaar1D (w : np.array) -> np.array:
    N = len(w)
    x = np.zeros(N)
    # lower half is computed as average
    for n in range (0, N//2):
        x[2*n] = (w[n] + w[n+N//2])/np.sqrt(2)
    # upper half is computed as difference
    for n in range (0, N//2):
        x[2*n+1] = (w[n] - w[n+N//2])/np.sqrt(2)
    return x

# test it on a random vector x
"""
x = np.random.rand(8)
w = haar1D(x)
x_hat = ihaar1D(w)

print("x = ", x)
print("w = ", w)
print("x_hat = ", x_hat)
"""
# checks out good =)

#######################################################

# 4.4.3 implement haar2D and ihaar2D


def haar2D(image: np.array) -> np.array:
    image = image.astype(float)  # Ensure image is in float format
    N, M = image.shape
    transformed = np.copy(image)

    # Apply Haar transform to rows
    for i in range(N):
        transformed[i, :] = haar1D(transformed[i, :])

    # Apply Haar transform to columns
    for j in range(M):
        transformed[:, j] = haar1D(transformed[:, j])

    return transformed


# 2D Inverse Haar Transform
def ihaar2D(image: np.array) -> np.array:
    N, M = image.shape
    reconstructed = np.copy(image)

    # Apply inverse Haar transform to columns
    for j in range(M):
        reconstructed[:, j] = ihaar1D(reconstructed[:, j])

    # Apply inverse Haar transform to rows
    for i in range(N):
        reconstructed[i, :] = ihaar1D(reconstructed[i, :])

    return reconstructed

image = plt.imread('tiger.png')
# test it on the image
image_haar = haar2D(image)
image_hat = ihaar2D(image_haar)


plt.subplot(1,3,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,2)
plt.title('Haar transformed image')
plt.axis('off')
plt.imshow(image_haar, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,3)
plt.title('inverse Haar transformed image')
plt.axis('off')
plt.imshow(image_hat, cmap='gray', interpolation='lanczos')
plt.show()



#######################################################

# 4.4.4 implement haar2D_k and ihaar2D_k

# increasing order only applies on approximation part of previous level

def haar2D_k(image: np.array, k: int) -> np.array:
    image = image.astype(float)  # Ensure image is in float format
    N, M = image.shape
    transformed = np.copy(image)

    transformed = haar2D(transformed)

    for order in range (1,k+1):
        transformed_approx = haar2D(transformed[:N//(2*(order)), :M//(2*(order))])
        # replace the approximation part of the image with the transformed approximation
        transformed[:N//(2*(order)), :M//(2*(order))] = transformed_approx
    
    return transformed

def ihaar2D_k(image: np.array, k: int) -> np.array:
    N, M = image.shape
    reconstructed = np.copy(image)

    for order in range(k, 0, -1):
        reconstructed_approx = ihaar2D(reconstructed[:N//(2*((order))), :M//(2*(order))])
        # replace the approximation part of the image with the transformed approximation
        reconstructed[:N//(2*(order)), :M//(2*(order))] = reconstructed_approx
    
    reconstructed = ihaar2D(reconstructed)
    return reconstructed

image = plt.imread('tiger.png')
# test it on the image
image_haar = haar2D_k(image, 3)
image_hat = ihaar2D_k(image_haar, 3)

# plot all three images
plt.subplot(1,3,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,2)
plt.title('third order Haar transformed image')
plt.axis('off')
plt.imshow(image_haar, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,3)
plt.title('inverse Haar transformed image')
plt.axis('off')
plt.imshow(image_hat, cmap='gray', interpolation='lanczos')
plt.show()

