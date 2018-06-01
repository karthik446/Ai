import scipy.signal
import numpy as np

# 7X7 image
image = np.array([[1, 2, 3, 4, 5, 6, 7],
                  [8, 9, 10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19, 20, 21],
                  [22, 23, 24, 25, 26, 27, 28],
                  [29, 30, 31, 32, 33, 34, 35],
                  [36, 37, 38, 39, 40, 41, 42],
                  [43, 44, 45, 46, 47, 48, 49]])

# Define Kernel
filter_kernel = np.array([[-1, 1, -1], [-2, 3, 1], [2, -4, 0]])

c = scipy.signal.convolve2d(image, filter_kernel, mode="same", boundary="fill", fillvalue=0)
# print(c)


def add_padding(size, row_size, col_size):
    image1 = np.zeros((size, size))
    for i in range(row_size):
        for j in range(col_size):
            image1[i + 1, j + 1] = image[i, j]
    return image1


# Convolution from scratch
row, col = 7, 7
# rotate the filter kernel twice by 90 degrees - i.e. 180 degrees
filter_kernel_flipped = np.rot90(filter_kernel, 2)
print(filter_kernel)
print(filter_kernel_flipped)
after_padding = add_padding(9, row, col)

# output image
image_out = np.zeros((row, col))

for i in range(1, 1+row):
    for j in range(1, 1+col):
        arr = np.zeros((3, 3))
        for k, k1 in zip(range(i-1, i+2), range(3)):
            for l, l1 in zip(range(j-1, j+2), range(3)):
                arr[k1, l1] = after_padding[k, l]

        image_out[i-1, j-1] = np.sum(np.multiply(arr, filter_kernel_flipped))

print(image_out)