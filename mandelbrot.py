import numpy as np

def mandelbrot_numpy(width, height, x_min, x_max, y_min, y_max, max_iter):
  """Generowanie zbioru Mandelbrota przy użyciu NumPy."""
  x = np.linspace(x_min, x_max, width)
  y = np.linspace(y_min, y_max, height)
  C = x[:, np.newaxis] + 1j * y[np.newaxis, :] # Siatka liczb zespolonych c

  Z = np.zeros_like(C, dtype=np.complex128)
  # Tablica przechowująca liczbę iteracji (czas ucieczki)
  iterations = np.zeros(C.shape, dtype=int)
  # Maska boolowska dla punktów, które jeszcze nie uciekły
  mask = np.full(C.shape, True, dtype=bool)

  for i in range(max_iter):
    # Iteruj tylko dla punktów wewnątrz maski
    Z[mask] = Z[mask]**2 + C[mask]
    # Znajdź punkty, które właśnie uciekły
    diverged = np.abs(Z) > 2
    # Zapisz liczbę iteracji dla punktów, które uciekły w tym kroku
    iterations[mask & diverged] = i
    # Zaktualizuj maskę, usuwając punkty, które uciekły
    mask &= ~diverged
    # Jeśli wszystkie punkty uciekły, przerwij
    if not mask.any():
      break

  # Punkty, które nie uciekły po max_iter, należą do zbioru (ustaw na max_iter)
  iterations[mask] = max_iter
  return iterations # Zwraca tablicę z liczbą iteracji dla każdego piksela

import numpy as np
import numba

@numba.njit
def mandel_point_cpu(real, imag, max_iters):
  """Oblicza czas ucieczki dla pojedynczego punktu c = real + imag*1j."""
  c = complex(real, imag)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    # Sprawdzenie warunku ucieczki (użycie kwadratu modułu jest szybsze)
    if (z.real*z.real + z.imag*z.imag) >= 4.0:
      return i # Zwraca liczbę iteracji
  return max_iters # Punkt należy do zbioru (lub nie uciekł w max_iters)

@numba.njit
def mandelbrot_numba_cpu(width, height, x_min, x_max, y_min, y_max, max_iter):
  """Generowanie zbioru Mandelbrota przy użyciu Numba JIT na CPU."""
  image = np.zeros((width, height), dtype=np.int32)
  pixel_size_x = (x_max - x_min) / width
  pixel_size_y = (y_max - y_min) / height

  for x in range(width):
    real = x_min + x * pixel_size_x
    for y in range(height):
      imag = y_min + y * pixel_size_y
      color = mandel_point_cpu(real, imag, max_iter)
      # Uwaga: indeksowanie obrazu może być (y, x) lub (x, y)
      # zależnie od konwencji, tu przyjmujemy (width, height) -> (x, y)
      image[x, y] = color
  return image

import numba.cuda
import numpy as np
import math

@numba.cuda.jit(device=True) # Funkcja pomocnicza wykonywana na GPU
def mandel_point_gpu(real, imag, max_iters):
  """Oblicza czas ucieczki dla pojedynczego punktu c = real + imag*1j (wersja GPU)."""
  c = complex(real, imag)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4.0:
      return i
  return max_iters

@numba.cuda.jit
def mandelbrot_cuda_kernel(min_x, max_x, min_y, max_y, image, iters):
  """Jądro CUDA do generowania zbioru Mandelbrota."""
  height = image.shape[1] # Zakładając image[width, height]
  width = image.shape

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  # Pobranie globalnych współrzędnych wątku (x, y)
  startX, startY = numba.cuda.grid(2)
  # Obliczenie kroku dla pętli grid-stride (na wypadek gdyby siatka była mniejsza niż obraz)
  gridX = numba.cuda.gridDim.x * numba.cuda.blockDim.x
  gridY = numba.cuda.gridDim.y * numba.cuda.blockDim.y

  # Pętla grid-stride (każdy wątek może obliczyć więcej niż jeden piksel)
  for x in range(startX, width, gridX):
      real = min_x + x * pixel_size_x
      for y in range(startY, height, gridY):
          imag = min_y + y * pixel_size_y
          # Sprawdzenie granic przed obliczeniem i zapisem
          if x < width and y < height:
              image[x, y] = mandel_point_gpu(real, imag, iters)

# Funkcja hosta do uruchomienia jądra
def mandelbrot_numba_gpu(width, height, x_min, x_max, y_min, y_max, max_iter):
    """Generowanie zbioru Mandelbrota przy użyciu Numba na GPU."""
    image = np.zeros((width, height), dtype=np.int32)
    # Przeniesienie tablicy wynikowej na GPU
    d_image = numba.cuda.to_device(image)

    # Konfiguracja siatki i bloków
    threadsperblock = (16, 16) # Typowy rozmiar bloku 2D
    blockspergrid_x = math.ceil(image.shape / threadsperblock)
    blockspergrid_y = math.ceil(image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Uruchomienie jądra
    mandelbrot_cuda_kernel[blockspergrid, threadsperblock](
        x_min, x_max, y_min, y_max, d_image, max_iter)

    # Synchronizacja (opcjonalna, jeśli kopiujemy od razu)
    # numba.cuda.synchronize()

    # Skopiowanie wyniku z powrotem na hosta
    image = d_image.copy_to_host()
    return image