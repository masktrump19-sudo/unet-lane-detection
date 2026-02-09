import numpy as np
import cv2
from numba import njit
import time

# 普通 Python 函数
def sum_array_python(arr):
    s = 0
    for i in range(arr.size):
        s += arr[i]
    return s

# Numba 优化函数
@njit
def sum_array_numba(arr):
    s = 0
    for i in range(arr.size):
        s += arr[i]
    return s

# 测试
arr = np.random.rand(1_000_000)

# 预热 Numba（首次调用需要编译）
sum_array_numba(arr)

# 性能测试
start = time.time()
sum_array_python(arr)
print(f"Python 耗时: {time.time() - start:.6f} 秒")

start = time.time()
sum_array_numba(arr)
print(f"Numba 耗时: {time.time() - start:.6f} 秒")