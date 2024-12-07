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

plt.figure()
plt.tight_layout()
plt.subplot(1,3,1)
plt.title('Original')
plt.axis('off')
plt.imshow(image, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,2)
plt.title('Haar tf')
plt.axis('off')
plt.imshow(image_haar, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,3)
plt.title('inverse Haar tf')
plt.axis('off')
plt.imshow(image_hat, cmap='gray', interpolation='lanczos')
plt.savefig('ex4_4_3.png')


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
plt.figure()
plt.subplot(1,3,1)
plt.title('Original')
plt.axis('off')
plt.imshow(image, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,2)
plt.title(r'$3^{rd}$ order Haar tf')
plt.axis('off')
plt.imshow(image_haar, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,3)
plt.title('inverse Haar tf')
plt.axis('off')
plt.imshow(image_hat, cmap='gray', interpolation='lanczos')
plt.savefig('ex4_4_4.png')

#######################################################

# 4.4.5 fuse two clock images together with a 3rd order haar transform

def fuse_images(image1: np.array, image2: np.array, k: int) -> np.array:
    # Ensure the images are the same size
    assert image1.shape == image2.shape, "Images must have the same dimensions for fusion."
    
    # Ensure the images are floats
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    
    N, M = image1.shape
    fused_image = np.zeros((N, M), dtype=float)

    for n in range (0, N//(2**k)):
        for m in range (0, M//(2**k)):
            fused_image[n, m] = (image1[n, m] + image2[n, m])/2.0

    for n in range (N//(2**k), N):
        for m in range (M//(2**k), M):
            # take the higher magnitude pixel
            if abs(image1[n, m]) > abs(image2[n, m]):
                fused_image[n, m] = image1[n, m]
            else:
                fused_image[n, m] = image2[n, m]

    return fused_image

image1 = plt.imread('clockA.png')
image2 = plt.imread('clockB.png')

# Ensure grayscale
if image1.ndim == 3:
    image1 = np.mean(image1, axis=2)
if image2.ndim == 3:
    image2 = np.mean(image2, axis=2)

image1_haar = haar2D_k(image1, 3)
image2_haar = haar2D_k(image2, 3)

fused_image = fuse_images(image1_haar, image2_haar, 3)

#plt.figure()
#plt.imshow(fused_image, cmap='gray', interpolation='lanczos')
#plt.axis('off')


fused_image_hat = ihaar2D_k(fused_image, 3)

# plot all three images
plt.figure()
plt.subplot(1,3,1)
plt.title('Original image 1')
plt.axis('off')
plt.imshow(image1, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,2)
plt.title('Original image 2')
plt.axis('off')
plt.imshow(image2, cmap='gray', interpolation='lanczos')
plt.subplot(1,3,3)
plt.title('Fused image')
plt.axis('off')
plt.imshow(fused_image_hat, cmap='gray', interpolation='lanczos')
plt.tight_layout()
plt.savefig('ex4_4_5.png')
plt.show()


