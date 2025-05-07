import matplotlib.pyplot as plt
import numpy as np

resolutions_small = ['5x5', '10x10', '25x25', '50x50', '100x100', '200x200']
numpy_cpu_small = [0.9567, 0.9638, 1.4608, 2.1986, 4.8517, 14.9533]
numba_cpu_small = [0.0022, 0.0076, 0.0378, 0.1565, 0.6215, 2.4693]
numba_gpu_small = [0.7995, 0.7015, 0.5505, 0.6086, 0.6991, 0.9056]

resolutions_large = ['500x500', '1000x1000', '2000x2000', '4000x4000', '6000x6000', '8000x8000']
numpy_cpu_large = [92.3840, 425.9445, 1918.0471, 8512.7585, 19937.6965, 36007.5226]
numba_cpu_large = [14.9744, 58.9605, 237.4027, 957.0664, 2148.7826, 3741.8975]
numba_gpu_large = [2.2465, 4.7986, 26.7021, 65.0070, 153.7686, 254.3510]

resolutions = resolutions_small + resolutions_large
numpy_cpu_times = numpy_cpu_small + numpy_cpu_large
numba_cpu_times = numba_cpu_small + numba_cpu_large
numba_gpu_times = numba_gpu_small + numba_gpu_large

init_overhead_ms = 550

numba_gpu_plus_overhead = [t + init_overhead_ms for t in numba_gpu_times]

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(resolutions, numpy_cpu_times, marker='o', linestyle='-', label='NumPy CPU')
ax.plot(resolutions, numba_cpu_times, marker='s', linestyle='--', label='Numba CPU')
ax.plot(resolutions, numba_gpu_times, marker='^', linestyle=':', label='Numba GPU (bez narzutu init.)')
ax.plot(resolutions, numba_gpu_plus_overhead, marker='x', linestyle='-.', label=f'Numba GPU + Narzut Init. ({init_overhead_ms} ms)')

ax.set_yscale('log')

ax.set_xlabel('Rozdzielczość Obrazu')
ax.set_ylabel('Średni Czas Wykonania (ms, skala logarytmiczna)')
ax.set_title('Porównanie Wydajności Obliczeń Zbioru Mandelbrota')

plt.xticks(rotation=45, ha='right')


ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.legend()

plt.tight_layout()

plt.savefig('wykres_wydajnosci_log.png')
print("Wykres został zapisany do pliku 'wykres_wydajnosci_liniowy.png'")