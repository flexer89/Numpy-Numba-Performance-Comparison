import argparse
import time
import numpy as np
from numba import cuda
import math
import os
import sys
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
try:
    from mandelbrot import (
        mandelbrot_numpy,
        mandelbrot_numba_cpu,
        mandelbrot_cuda_kernel
    )
except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że plik mandelbrot.py znajduje się w tym samym katalogu.")
    sys.exit(1)


def time_func(func, *args, reps=5, warmup=1):
    for _ in range(warmup):
        func(*args)

    start_time = time.perf_counter()
    for _ in range(reps):
        func(*args)
    end_time = time.perf_counter()
    return (end_time - start_time) / reps * 1000

def time_gpu_kernel(kernel_func, blockspergrid, threadsperblock, args_tuple, reps=5, warmup=1):

    for _ in range(warmup):
        kernel_func[blockspergrid, threadsperblock](*args_tuple)
        cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(reps):
        kernel_func[blockspergrid, threadsperblock](*args_tuple)
        cuda.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / reps * 1000


def benchmark_mandelbrot(sizes, max_iter=100, reps=5, warmup=1):
    results = {}
    print("\n--- Benchmark Zbioru Mandelbrota ---")
    print(f"(max_iter = {max_iter})")
    print(f"{'Rozdzielczość':<15} | {'NumPy CPU (ms)':<18} | {'Numba CPU (ms)':<18} | {'Numba GPU Kernel (ms)':<22} | {'Numba GPU Total (ms)':<21}")
    print("-" * 100)

    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    gpu_available = cuda.is_available()

    for width in sizes:
        height = int(width * (y_max - y_min) / (x_max - x_min))
        res_str = f"{width}x{height}"
        print(f"{res_str:<15} | ", end="", flush=True)
        results[res_str] = {}

        image_host_ref = np.zeros((height, width), dtype=np.int32)

        # 1. NumPy CPU
        try:
            t_numpy = time_func(mandelbrot_numpy, width, height, x_min, x_max, y_min, y_max, max_iter, reps=reps, warmup=warmup)
            results[res_str]['numpy_cpu'] = t_numpy
            print(f"{t_numpy:<18.4f} | ", end="", flush=True)

        except Exception as e:
            print(f"{'Błąd':<18} | ", end="", flush=True)
            results[res_str]['numpy_cpu'] = float('inf')
            print(f"  (Błąd NumPy: {e})")

        # 2. Numba CPU
        try:
            t_numba_cpu = time_func(mandelbrot_numba_cpu, width, height, x_min, x_max, y_min, y_max, max_iter, reps=reps, warmup=warmup)
            results[res_str]['numba_cpu'] = t_numba_cpu
            print(f"{t_numba_cpu:<18.4f} | ", end="", flush=True)
            # Sprawdzenie poprawności (opcjonalne, tylko raz)
            # img_numba_cpu = mandelbrot_numba_cpu(width, height, x_min, x_max, y_min, y_max, max_iter)
            # if 'img_np' in locals():
            #      assert np.array_equal(img_np, img_numba_cpu)
        except Exception as e:
            print(f"{'Błąd':<18} | ", end="", flush=True)
            results[res_str]['numba_cpu'] = float('inf')
            print(f"  (Błąd Numba CPU: {e})")

        # 3. Numba GPU
        if gpu_available:
            try:
                threadsperblock_gpu = (16, 16)
                blockspergrid_x = math.ceil(image_host_ref.shape[1] / threadsperblock_gpu[1])
                blockspergrid_y = math.ceil(image_host_ref.shape[0] / threadsperblock_gpu[0])
                blockspergrid_gpu = (blockspergrid_x, blockspergrid_y)

                start_total = time.perf_counter()
                image_device = cuda.to_device(image_host_ref)
                cuda.synchronize()
                transfer_done = time.perf_counter()

                kernel_args = (x_min, x_max, y_min, y_max, image_device, max_iter)

                mandelbrot_cuda_kernel[blockspergrid_gpu, threadsperblock_gpu](*kernel_args)
                cuda.synchronize()
                kernel_done = time.perf_counter()

                image_res_gpu = image_device.copy_to_host()
                cuda.synchronize()
                copy_back_done = time.perf_counter()
                end_total = copy_back_done

                t_gpu_total = (end_total - start_total) * 1000
                t_gpu_kernel = time_gpu_kernel(mandelbrot_cuda_kernel,
                                               blockspergrid_gpu, threadsperblock_gpu,
                                               kernel_args,
                                               reps=reps, warmup=warmup)

                results[res_str]['gpu_kernel'] = t_gpu_kernel
                results[res_str]['gpu_total'] = t_gpu_total
                print(f"{t_gpu_kernel:<22.4f} | ", end="", flush=True)
                print(f"{t_gpu_total:<21.4f}")

                del image_device
            except Exception as e:
                print(f"{'Błąd GPU':<22} | {'Błąd GPU':<21}")
                results[res_str]['gpu_kernel'] = float('inf')
                results[res_str]['gpu_total'] = float('inf')
                print(f"  (Błąd Numba GPU: {e})")
        else:
             print(f"{'Brak GPU':<22} | {'Brak GPU':<21}")
             results[res_str]['gpu_kernel'] = float('inf')
             results[res_str]['gpu_total'] = float('inf')

    print("-" * 100)
    return results


def print_speedups(results, baseline_key='numpy_cpu'):
    print("\n--- Współczynniki Przyspieszenia ---")
    print(f"(Baseline: {baseline_key})")

    if not results:
        print("Brak wyników do analizy przyspieszenia.")
        return

    try:
        first_size_key = next(iter(results))
        methods = [k for k in results[first_size_key].keys() if k != baseline_key]
        if baseline_key not in results[first_size_key]:
             print(f"Błąd: Klucz bazowy '{baseline_key}' nie znaleziony w wynikach.")
             potential_baselines = ['numba_cpu', 'numpy_cpu']
             old_baseline = baseline_key
             for pb in potential_baselines:
                  if pb in results[first_size_key]:
                       baseline_key = pb
                       print(f"Używam '{baseline_key}' jako nowego baseline.")
                       methods = [k for k in results[first_size_key].keys() if k != baseline_key]
                       break
             if old_baseline == baseline_key:
                  print("Nie można kontynuować bez poprawnego baseline.")
                  return

    except StopIteration:
        print("Brak wyników do analizy przyspieszenia.")
        return

    column_width = 15
    header = f"{'Rozmiar':<15} | " + " | ".join([f"{m.replace('_', ' ').upper():^{column_width}}" for m in methods])
    print(header)
    print("-" * len(header))

    for size, timings in results.items():
        baseline_time = timings.get(baseline_key)
        if baseline_time is None or baseline_time <= 0 or baseline_time == float('inf'):
            print(f"{size:<15} | {'Baseline N/A':^{len(header)-18}}")
            continue

        print(f"{size:<15} | ", end="")
        speedup_values = []
        for method in methods:
            method_time = timings.get(method)
            if method_time is None or method_time <= 0 or method_time == float('inf'):
                speedup_str = "N/A"
            else:
                speedup = baseline_time / method_time
                speedup_str = f"{speedup:.2f}x"
            speedup_values.append(f"{speedup_str:^{column_width}}")
        print(" | ".join(speedup_values))

    print("-" * len(header))

def plot_results(all_results, output_dir='plots'):
    if not MATPLOTLIB_AVAILABLE:
        print("\nBiblioteka matplotlib nie jest zainstalowana. Wykresy nie zostaną wygenerowane.")
        return
    if not all_results:
        print("\nBrak wyników do wygenerowania wykresów.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerowanie wykresów w katalogu: {output_dir}")

    for algorithm_name, results_data in all_results.items():
        if not results_data:
            print(f"Brak wyników dla algorytmu: {algorithm_name}, pomijanie wykresów.")
            continue

        sizes = list(results_data.keys())
        try:
            def get_width(size_str):
                try:
                    return int(str(size_str).split('x')[0])
                except:
                    return float('inf')
            sizes.sort(key=get_width)
        except ValueError:
            print(f"Ostrzeżenie: Nie można posortować rozmiarów dla {algorithm_name}, używam kolejności ze słownika.")


        first_size_key = sizes[0]
        all_methods = list(results_data[first_size_key].keys())

        valid_methods = []
        for method in all_methods:
            is_valid = False
            for size in sizes:
                time_val = results_data[size].get(method)
                if time_val is not None and time_val != float('inf'):
                    is_valid = True
                    break
            if is_valid:
                valid_methods.append(method)

        if not valid_methods:
             print(f"Brak poprawnych danych czasowych dla {algorithm_name}, pomijanie wykresów.")
             continue

        timings = {method: [] for method in valid_methods}
        for size in sizes:
            for method in valid_methods:
                time_val = results_data[size].get(method)
                timings[method].append(time_val if (time_val is not None and time_val != float('inf')) else 0)

        # --- Wykres 1: Czas wykonania (słupkowy) ---
        plt.figure(figsize=(12, 7))
        num_methods = len(valid_methods)
        bar_width = 0.8 / num_methods
        index = np.arange(len(sizes))

        for i, method in enumerate(valid_methods):
            bar_positions = index + (i - num_methods / 2 + 0.5) * bar_width
            plt.bar(bar_positions, timings[method], bar_width, label=method.replace('_', ' ').upper())

        plt.xlabel("Rozmiar Problemu")
        plt.ylabel("Średni Czas Wykonania (ms)")
        plt.title(f"Porównanie Wydajności - {algorithm_name.capitalize()}")
        plt.xticks(index, sizes, rotation=45, ha="right")
        plt.legend(title="Metoda")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        plot_filename_time = os.path.join(output_dir, f"{algorithm_name}_performance_comparison.png")
        try:
            plt.savefig(plot_filename_time)
            print(f"Zapisano wykres czasu: {plot_filename_time}")
        except Exception as e:
            print(f"Błąd podczas zapisywania wykresu czasu dla {algorithm_name}: {e}")
        plt.close()

        # --- Wykres 2: Przyspieszenie (słupkowy) ---
        baseline_key = 'numpy_cpu'
        if baseline_key not in valid_methods:
             print(f"Baseline '{baseline_key}' nie ma poprawnych danych dla {algorithm_name}, pomijanie wykresu przyspieszenia.")
             continue

        plt.figure(figsize=(12, 7))
        baseline_times = np.array(timings[baseline_key])
        baseline_times[baseline_times <= 0] = np.nan

        methods_for_speedup = [m for m in valid_methods if m != baseline_key]
        num_methods_speedup = len(methods_for_speedup)
        if num_methods_speedup == 0:
             print(f"Brak innych metod do porównania z baseline '{baseline_key}' dla {algorithm_name}.")
             plt.close()
             continue

        bar_width_speedup = 0.8 / num_methods_speedup
        index_speedup = np.arange(len(sizes))

        for i, method in enumerate(methods_for_speedup):
            method_times = np.array(timings[method])
            method_times[method_times <= 0] = np.nan
            speedup = baseline_times / method_times
            speedup = np.nan_to_num(speedup, nan=0.0)

            bar_positions = index_speedup + (i - num_methods_speedup / 2 + 0.5) * bar_width_speedup
            plt.bar(bar_positions, speedup, bar_width_speedup, label=method.replace('_', ' ').upper())

        plt.xlabel("Rozmiar Problemu")
        plt.ylabel(f"Współczynnik Przyspieszenia (względem {baseline_key.upper()})")
        plt.title(f"Przyspieszenie Obliczeń - {algorithm_name.capitalize()}")
        plt.xticks(index_speedup, sizes, rotation=45, ha="right")
        plt.axhline(1, color='grey', linestyle='--', linewidth=0.8)
        plt.legend(title="Metoda")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        plot_filename_speedup = os.path.join(output_dir, f"{algorithm_name}_speedup_comparison.png")
        try:
            plt.savefig(plot_filename_speedup)
            print(f"Zapisano wykres przyspieszenia: {plot_filename_speedup}")
        except Exception as e:
             print(f"Błąd podczas zapisywania wykresu przyspieszenia dla {algorithm_name}: {e}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Numba CPU/GPU vs NumPy.")
    parser.add_argument('--algorithm', type=str, default='mandelbrot',
                        choices=['mandelbrot'],
                        help="Algorytm do testowania (mandelbrot)")
    parser.add_argument('--sizes_mandelbrot', nargs='+', type=int,
                        default=[500, 1000, 2000],
                        help="Lista szerokości obrazu dla zbioru Mandelbrota.")
    parser.add_argument('--mandel_iter', type=int, default=100,
                        help="Maksymalna liczba iteracji dla zbioru Mandelbrota.")
    parser.add_argument('--reps', type=int, default=5,
                        help="Liczba powtórzeń pomiaru czasu.")
    parser.add_argument('--warmup', type=int, default=1,
                        help="Liczba powtórzeń rozgrzewkowych.")
    parser.add_argument('--plot', action='store_true',
                        help="Generuj wykresy wyników (wymaga matplotlib).")
    parser.add_argument('--plot_dir', type=str, default='plots',
                        help="Katalog do zapisania wygenerowanych wykresów.")

    args = parser.parse_args()

    gpu_available = False
    try:
        devices = cuda.list_devices()
        if devices:
            print("Dostępne urządzenia CUDA:")
            for i, device in enumerate(devices):
                 print(f"  {i}: {device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name}")
            gpu_available = True
        else:
             print("Ostrzeżenie: Brak dostępnych urządzeń CUDA.")
    except cuda.cudadrv.error.CudaSupportError as e:
        print(f"Błąd inicjalizacji CUDA: {e}")
        print("Ostrzeżenie: Brak wsparcia dla CUDA lub problem ze sterownikami. Testy GPU zostaną pominięte.")
    except Exception as e:
         print(f"Nieoczekiwany błąd podczas sprawdzania urządzeń CUDA: {e}")
         print("Ostrzeżenie: Testy GPU zostaną pominięte.")

    if not gpu_available:
         print("Testy GPU zostaną pominięte.")


    all_results = {}

    if args.algorithm in ['mandelbrot', 'all']:
        results_mandel = benchmark_mandelbrot(args.sizes_mandelbrot, args.mandel_iter, args.reps, args.warmup)
        all_results['mandelbrot'] = results_mandel
        print_speedups(results_mandel)

    print("\nZakończono benchmark.")

    if args.plot:
        plot_results(all_results, output_dir=args.plot_dir)