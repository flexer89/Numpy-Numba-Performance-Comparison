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
        mandelbrot_cuda_kernel,
        mandelbrot_numba_gpu
    )
except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że plik mandelbrot.py znajduje się w tym samym katalogu.")
    sys.exit(1)
    
import warnings

from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

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


def benchmark_mandelbrot(sizes, max_iter=100, reps=5, warmup=1, gpu_warmup=True):

    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    gpu_available = cuda.is_available()

    results = {}
    print("\n--- Benchmark Zbioru Mandelbrota ---")
    print(f"(max_iter = {max_iter})")
    print(f"{'Rozdzielczość':<15} | {'NumPy CPU (ms)':<18} | {'Numba CPU (ms)':<18} | {'Numba GPU (ms)':<21}")
    print("-" * 100)

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
            print(f"{t_numba_cpu:<18.4f}", end="", flush=True)
        except Exception as e:
            print(f"{'Błąd':<0} | ", end="", flush=True)
            results[res_str]['numba_cpu'] = float('inf')
            print(f"  (Błąd Numba CPU: {e})")

        # 3. Numba GPU
        if gpu_available:
            try:
                t_numba_gpu_total = time_func(mandelbrot_numba_gpu,
                                                width, height,
                                                x_min, x_max,
                                                y_min, y_max,
                                                max_iter,
                                                reps=reps, warmup=warmup) 

                results[res_str]['gpu_total'] = t_numba_gpu_total
                print(f" | {t_numba_gpu_total:<21.4f}") 

            except Exception as e:

                print(f"{'Błąd GPU':<22} | {'Błąd GPU':<21}")
                results[res_str]['gpu_total'] = float('inf')
                print(f"   (Błąd Numba GPU: {e})")
        else:
            print(f"{'Brak GPU':<22} | {'Brak GPU':<21}")
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
        methods = [k for k in results[first_size_key].keys() if k != baseline_key and k != 'gpu_kernel']
        if baseline_key not in results[first_size_key]:
             print(f"Błąd: Klucz bazowy '{baseline_key}' nie znaleziony w wynikach.")
             potential_baselines = ['numba_cpu', 'numpy_cpu']
             old_baseline = baseline_key
             for pb in potential_baselines:
                  if pb in results[first_size_key]:
                       baseline_key = pb
                       print(f"Używam '{baseline_key}' jako nowego baseline.")
                       methods = [k for k in results[first_size_key].keys() if k != baseline_key and k != 'gpu_kernel']
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


def print_and_plot_detailed_results(results, output_dir='plots', algorithm_name='benchmark', plot_enabled=True):

    def get_width(size_str):
        try:
             return int(str(size_str).split('x')[0])
        except (ValueError, IndexError, AttributeError):
             print(f"Ostrzeżenie: Nie można sparsować szerokości z klucza '{size_str}'. Używanie wartości 0 do sortowania.")
             return 0
    try:
        valid_size_keys = [k for k in results.keys() if isinstance(k, str) and 'x' in k]
        if not valid_size_keys:
            raise ValueError("Brak kluczy w formacie 'WxH' w wynikach.")
        sizes = sorted(valid_size_keys, key=get_width)
    except ValueError as e:
         print(f"Krytyczny błąd: Nie można posortować rozmiarów. Sprawdź format kluczy w 'results'. Błąd: {e}")
         raise
    except Exception as e:
         print(f"Krytyczny błąd podczas sortowania rozmiarów: {e}")
         raise

    if not sizes:
        print("Krytyczny błąd: Brak poprawnych rozmiarów w wynikach po filtrowaniu.")
        raise ValueError("Brak rozmiarów w wynikach.")

    first_size_key = sizes[0]
    if first_size_key not in results or not isinstance(results[first_size_key], dict):
         print(f"Krytyczny błąd: Dane dla rozmiaru '{first_size_key}' są nieprawidłowe lub brak ich.")
         raise KeyError(f"Nieprawidłowe lub brakujące dane dla klucza rozmiaru '{first_size_key}'.")

    all_available_methods_in_first = list(results[first_size_key].keys())
    if not all_available_methods_in_first:
        print(f"Krytyczny błąd: Brak metod zarejestrowanych dla rozmiaru '{first_size_key}'.")
        raise ValueError("Brak metod w wynikach.")

    methods_to_process = [m for m in all_available_methods_in_first if m != 'gpu_kernel']

    if not methods_to_process:
        print("Krytyczny błąd: Brak metod do przetworzenia po odfiltrowaniu 'gpu_kernel'.")
        raise ValueError("Brak metod (innych niż gpu_kernel) do analizy.")

    print(f"\nINFO: Metody uwzględnione w analizie i na wykresach (gpu_kernel pominięty): {', '.join(methods_to_process)}")

    comparisons_to_make = [
        ('numba_cpu', 'numpy_cpu', 'NumbaCPU/NumPy'),
        ('gpu_total', 'numpy_cpu', 'GPU Total/NumPy'),
        ('gpu_total', 'numba_cpu', 'GPU Total/NumbaCPU'),
    ]

    valid_comparisons = [
        c for c in comparisons_to_make
        if c[0] in methods_to_process and c[1] in methods_to_process
    ]

    if valid_comparisons:
        column_width = 18
        header_parts = [f"{'Rozmiar':<15}"] + [f"{c[2]:^{column_width}}" for c in valid_comparisons]
        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))
        for size in sizes:
            if size not in results:
                print(f"Ostrzeżenie: Brak danych dla rozmiaru {size} w tabeli.")
                continue
            timings_row = results[size]
            if not isinstance(timings_row, dict):
                 print(f"Ostrzeżenie: Nieprawidłowe dane dla rozmiaru {size} (oczekiwano słownika), pomijanie wiersza.")
                 continue

            print(f"{size:<15} | ", end="")
            speedup_values = []
            for faster_key, slower_key, _ in valid_comparisons:
                time_faster = timings_row.get(faster_key)
                time_slower = timings_row.get(slower_key)
                speedup_str = "N/A"

                is_f_valid = isinstance(time_faster, (int, float)) and time_faster > 0 and not math.isinf(time_faster) and not math.isnan(time_faster)
                is_s_valid = isinstance(time_slower, (int, float)) and time_slower > 0 and not math.isinf(time_slower) and not math.isnan(time_slower)

                if is_f_valid and is_s_valid:
                    speedup = time_slower / time_faster
                    speedup_str = f"{speedup:.2f}x"
                elif not is_f_valid and not is_s_valid: speedup_str = "Oba N/A"
                elif not is_f_valid: speedup_str = f"{faster_key}-N/A"
                elif not is_s_valid: speedup_str = f"{slower_key}-N/A"
                speedup_values.append(f"{speedup_str:^{column_width}}")
            print(" | ".join(speedup_values))
        print("-" * len(header))
    else:
        print("Brak wystarczających danych dla zdefiniowanych porównań (po odfiltrowaniu gpu_kernel). Tabela przyspieszenia nie zostanie wygenerowana.")

    if not plot_enabled:
        print("\nGenerowanie wykresów jest wyłączone.")
        return

    print("\n--- Generowanie Wykresów (bez gpu_kernel) ---")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Wykresy zostaną zapisane w katalogu: {output_dir}")

    plot_timings = {method: [] for method in methods_to_process}
    plot_sizes = []

    for size in sizes:
        if size not in results: continue
        timings_row = results[size]
        if not isinstance(timings_row, dict): continue

        has_any_valid_data_no_kernel = any(
            isinstance(timings_row.get(m), (int, float)) and
            not math.isinf(timings_row.get(m)) and
            not math.isnan(timings_row.get(m))
            for m in methods_to_process
        )

        if not has_any_valid_data_no_kernel:
             print(f"Ostrzeżenie: Pomijanie rozmiaru '{size}' w danych do wykresów - brak poprawnych czasów dla metod (bez kernela).")
             continue

        plot_sizes.append(size)

        for method in methods_to_process:
             time_val = timings_row.get(method)
             if isinstance(time_val, (int, float)) and not math.isinf(time_val) and not math.isnan(time_val):
                 plot_timings[method].append(time_val)
             else:
                 plot_timings[method].append(np.nan)

    if not plot_sizes:
         print("Krytyczny błąd: Brak rozmiarów z poprawnymi danymi do wygenerowania jakichkolwiek wykresów (po odfiltrowaniu gpu_kernel).")
         return

    index = np.arange(len(plot_sizes))

    print("\n--- Generowanie Wykresu Czasu (Bez gpu_kernel, Skala Log) ---")
    plt.figure(figsize=(12, 7))
    methods_plotted_log = 0
    for method in methods_to_process:
        if not np.all(np.isnan(plot_timings[method])):
            plt.plot(index, plot_timings[method], marker='o', linestyle='-', linewidth=1.5, markersize=5, label=method.replace('_', ' ').upper())
            methods_plotted_log += 1
        else:
             print(f"Ostrzeżenie: Brak poprawnych danych dla metody '{method}' na wykresie czasu (log).")

    if methods_plotted_log > 0:
        plt.xlabel("Rozmiar Problemu (Szerokość Obrazu)")
        plt.ylabel("Średni Czas Wykonania [ms] (Skala Logarytmiczna)")
        plt.title(f"Porównanie Wydajności - {algorithm_name.capitalize()}")
        plt.xticks(index, plot_sizes, rotation=45, ha="right")
        plt.yscale('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: plt.legend(title="Metoda")

        plt.grid(True, axis='x', which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plot_filename_time_log = os.path.join(output_dir, f"{algorithm_name}_performance_no_kernel_log.png")
        try:
            plt.savefig(plot_filename_time_log)
            print(f"Zapisano wykres czasu (bez gpu_kernel, log): {plot_filename_time_log}")
        except Exception as e:
            print(f"Błąd podczas zapisywania wykresu {plot_filename_time_log}: {e}")
    else:
        print("Pominięto generowanie wykresu czasu (log) - brak danych do narysowania.")
    plt.close()

    print("\n--- Generowanie Wykresu Przyspieszenia (Bez gpu_kernel, Słupkowy z Wartościami) ---")
    baseline_key = 'numpy_cpu'
    if baseline_key not in methods_to_process:
        print(f"Baseline '{baseline_key}' nie znajduje się wśród przetwarzanych metod (po odfiltrowaniu gpu_kernel). Pomijanie wykresu przyspieszenia.")
    elif baseline_key not in plot_timings or np.all(np.isnan(plot_timings[baseline_key])):
            print(f"Baseline '{baseline_key}' zawiera tylko NaN lub brak danych. Pomijanie wykresu przyspieszenia.")
    else:
        methods_for_speedup = [m for m in methods_to_process if m != baseline_key]

        if methods_for_speedup:
            plt.figure(figsize=(12, 7))
            baseline_times = np.array(plot_timings[baseline_key], dtype=float)
            baseline_times[np.isnan(baseline_times) | (baseline_times <= 0)] = np.inf

            num_methods_speedup = len(methods_for_speedup)

            total_bar_width_fraction = 0.8
            bar_width_speedup = total_bar_width_fraction / num_methods_speedup

            label_fontsize = 7 if bar_width_speedup > 0.15 else 6

            index_speedup = np.arange(len(plot_sizes))
            max_speedup_value = 0

            for i, method in enumerate(methods_for_speedup):
                method_times = np.array(plot_timings.get(method, [np.nan]*len(plot_sizes)), dtype=float)
                method_times[np.isnan(method_times) | (method_times <= 0)] = np.inf

                with np.errstate(divide='ignore', invalid='ignore'):
                    speedup = baseline_times / method_times
                speedup[~np.isfinite(speedup)] = 0.0

                if np.any(speedup > 0):
                    bar_positions = index_speedup + (i - num_methods_speedup / 2 + 0.5) * bar_width_speedup
                    bars = plt.bar(bar_positions, speedup, bar_width_speedup, label=method.replace('_', ' ').upper())

                    for j, bar in enumerate(bars):
                        s_val = speedup[j]
                        if s_val > 0:
                            yval = bar.get_height()
                            max_speedup_value = max(max_speedup_value, yval)
                            plt.text(bar.get_x() + bar.get_width()/2.0,
                                     yval,
                                     f"{s_val:.2f}x",
                                     ha='center',
                                     va='bottom',
                                     fontsize=label_fontsize,
                                     rotation=0)
                else:
                     print(f"Ostrzeżenie: Brak poprawnych danych przyspieszenia dla metody '{method}' (wzgl. '{baseline_key}').")

            plt.xlabel("Rozmiar Problemu (Szerokość Obrazu)")
            plt.ylabel(f"Współczynnik Przyspieszenia (względem {baseline_key.upper()})")
            plt.title(f"Przyspieszenie Obliczeń - {algorithm_name.capitalize()}")
            plt.xticks(index_speedup, plot_sizes, rotation=45, ha="right")
            plt.axhline(1, color='grey', linestyle='--', linewidth=0.8, label=f'Baseline {baseline_key.upper()} (1x)')


            if max_speedup_value > 0:

                plt.ylim(top=max_speedup_value * 1.20)

            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), title="Metoda", fontsize=8)

            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            plot_filename_speedup = os.path.join(output_dir, f"{algorithm_name}_speedup_no_kernel.png")
            try:
                plt.savefig(plot_filename_speedup)
                print(f"Zapisano wykres przyspieszenia (bez gpu_kernel): {plot_filename_speedup}")
            except Exception as e:
                print(f"Błąd podczas zapisywania wykresu {plot_filename_speedup}: {e}")
        else:
             print(f"Brak innych metod (poza baseline '{baseline_key}') do porównania w wykresie przyspieszenia.")
        plt.close()


    print("\n--- Generowanie Wykresu Czasu (Bez gpu_kernel, Skala Liniowa) ---")
    plt.figure(figsize=(12, 7))
    methods_plotted_linear = 0
    
    for method in methods_to_process:
        if not np.all(np.isnan(plot_timings[method])):
            plt.plot(index, plot_timings[method], marker='o', linestyle='-', linewidth=1.5, markersize=5, label=method.replace('_', ' ').upper())
            methods_plotted_linear += 1
        else:
            print(f"Ostrzeżenie: Brak poprawnych danych dla metody '{method}' na wykresie czasu (lin).")

    if methods_plotted_linear > 0:
        plt.xlabel("Rozmiar Problemu (Szerokość Obrazu)")
        plt.ylabel("Średni Czas Wykonania [ms] (Skala Liniowa)")

        plt.title(f"Porównanie Wydajności - {algorithm_name.capitalize()}")
        plt.xticks(index, plot_sizes, rotation=45, ha="right")
        plt.yscale('linear')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: plt.legend(title="Metoda")
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plot_filename_time_lin = os.path.join(output_dir, f"{algorithm_name}_performance_no_kernel_linear.png")
        try:
            plt.savefig(plot_filename_time_lin)
            print(f"Zapisano wykres czasu (bez gpu_kernel, lin): {plot_filename_time_lin}")
        except Exception as e:
            print(f"Błąd podczas zapisywania wykresu {plot_filename_time_lin}: {e}")
    else:
        print("Pominięto generowanie wykresu czasu (lin) - brak jakichkolwiek danych do narysowania.")
    plt.close()

    print("\nGenerowanie wykresów zakończone.")

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
                        default=[500, 1000, 2000, 4000, 6000, 8000],
                        help="Lista szerokości obrazu dla zbioru Mandelbrota.")
    parser.add_argument('--mandel_iter', type=int, default=100,
                        help="Maksymalna liczba iteracji dla zbioru Mandelbrota.")
    parser.add_argument('--reps', type=int, default=10,
                        help="Liczba powtórzeń pomiaru czasu.")
    parser.add_argument('--warmup', type=int, default=1,
                        help="Liczba powtórzeń rozgrzewkowych.")
    parser.add_argument('--plot', action='store_true', default=True,
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

    if args.algorithm in ['mandelbrot']:
        print(f"\nUruchamianie benchmarku dla: {args.algorithm}")
        results_mandel = benchmark_mandelbrot(
            args.sizes_mandelbrot,
            args.mandel_iter,
            args.reps,
            args.warmup,
            gpu_warmup=True
        )
        all_results['mandelbrot'] = results_mandel

        print_and_plot_detailed_results(results_mandel)

    print("\nZakończono benchmark.")
