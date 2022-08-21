# -*- coding:utf-8 -*-
import math
import time

import cv2
from numba import cuda


@cuda.jit
def process_gpu(img, channels):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for c in range(channels):
        color = img[tx, ty][c] * 2.0 + 30
        if color > 255:
            img[tx, ty][c] = 255
        elif color < 0:
            img[tx, ty][c] = 0
        else:
            img[tx, ty][c] = color


def process_cpu(ima, dst):
    rows, cols, channels = img.shape
    for row in range(rows):
        for col in range(cols):
            for c in range(3):
                color = img[row, col][c] * 2.0 + 30
                if color > 255:
                    dst[row, col][c] = 255
                elif color < 0:
                    dst[row, col][c] = 0
                else:
                    dst[row, col][c] = color


if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    rows, cols, channels = img.shape
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    ## CPU function
    start_cpu = time.time()
    process_cpu(img, dst_cpu)
    end_cpu = time.time()
    time_cpu = end_cpu - start_cpu
    print(" CPU Process time: " + str(time_cpu))

    ##GPU function
    dImg = cuda.to_device(img)
    threadsperblock = (4, 4)
    blockspergrid_x = int(math.ceil(rows / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid, threadsperblock](dImg, channels)
    cuda.synchronize()
    end_gpu = time.time()
    dst_gpu = dImg.copy_to_host()
    time_gpu = start_gpu - end_gpu
    print(" GPU Process time: " + str(time_gpu))

    cv2.imwrite('result_cpu.jpg', dst_cpu)
    cv2.imwrite('result_gpu.jpg', dst_gpu)
    print('Done.')
