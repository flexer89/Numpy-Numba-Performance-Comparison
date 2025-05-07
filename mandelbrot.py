import numpy as np
import math
import numba

def mandelbrot_numpy(width, height, x_min, x_max, y_min, y_max, max_iter):
  x = np.linspace(x_min, x_max, width)
  y = np.linspace(y_min, y_max, height)
  C = x[:, np.newaxis] + 1j * y[np.newaxis, :]

  Z = np.zeros_like(C, dtype=np.complex128)
  iterations = np.zeros(C.shape, dtype=int)
  mask = np.full(C.shape, True, dtype=bool)

  for i in range(max_iter):
    Z[mask] = Z[mask]**2 + C[mask]
    diverged = np.abs(Z) > 2
    iterations[mask & diverged] = i
    mask &= ~diverged
    if not mask.any():
      break

  iterations[mask] = max_iter
  return iterations

@numba.njit
def mandel_point_cpu(real, imag, max_iters):
  c = complex(real, imag)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4.0:
      return i
  return max_iters

@numba.njit
def mandelbrot_numba_cpu(width, height, x_min, x_max, y_min, y_max, max_iter):
  image = np.zeros((width, height), dtype=np.int32)
  pixel_size_x = (x_max - x_min) / width
  pixel_size_y = (y_max - y_min) / height

  for x in range(width):
    real = x_min + x * pixel_size_x
    for y in range(height):
      imag = y_min + y * pixel_size_y
      color = mandel_point_cpu(real, imag, max_iter)
      image[x, y] = color
  return image


@numba.cuda.jit(device=True)
def mandel_point_gpu(real, imag, max_iters):
  c = complex(real, imag)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4.0:
      return i
  return max_iters

@numba.cuda.jit
def mandelbrot_cuda_kernel(min_x, max_x, min_y, max_y, image, iters):
  width = image.shape[0]
  height = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = numba.cuda.grid(2)
  gridX = numba.cuda.gridDim.x * numba.cuda.blockDim.x
  gridY = numba.cuda.gridDim.y * numba.cuda.blockDim.y

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y
      if x < width and y < height:
           image[x, y] = mandel_point_gpu(real, imag, iters)

def mandelbrot_numba_gpu(width, height, x_min, x_max, y_min, y_max, max_iter):
    image = np.zeros((width, height), dtype=np.int32)
    d_image = numba.cuda.to_device(image)

    threadsperblock = (32, 16)
    blockspergrid_x = math.ceil(image.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mandelbrot_cuda_kernel[blockspergrid, threadsperblock](
        x_min, x_max, y_min, y_max, d_image, max_iter)

    image = d_image.copy_to_host()
    return image