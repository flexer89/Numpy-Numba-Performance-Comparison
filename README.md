# Porównanie Wydajności NumPy vs Numba (CPU/GPU)

Projekt ten przeprowadza benchmark i porównuje wydajność różnych metod implementacji dla popularnego algorytu numerycznyego w Pythonie: generowania zbioru Mandelbrota. Porównywane są implementacje wykorzystujące standardowy NumPy, kompilację Numba dla CPU oraz kompilację Numba dla GPU (CUDA).

## Cel Projektu

Głównym celem jest zrozumienie i zwizualizowanie różnic w wydajności pomiędzy:

* Standardowymi, często już zoptymalizowanymi operacjami NumPy.
* Przenośnym przyspieszeniem kodu Python za pomocą Numba na CPU.
* Wykorzystaniem mocy obliczeniowej GPU za pomocą Numba CUDA, z uwzględnieniem zarówno czasu samego jądra obliczeniowego, jak i całkowitego czasu wraz z transferem danych.

## Funkcjonalności

* **Benchmark algorytmu:**
    * Generowanie zbioru Mandelbrota (WxH)
* **Porównanie 4 metod:**
    * `NumPy CPU`: Standardowa implementacja NumPy.
    * `Numba CPU`: Implementacja z pętlami skompilowana przez Numba dla CPU.
    * `Numba GPU Kernel`: Czas wykonania samego jądra CUDA na GPU.
    * `Numba GPU Total`: Całkowity czas wykonania na GPU (transfer danych H2D -> jądro -> transfer danych D2H).
* **Konfiguracja przez linię poleceń:** Możliwość wyboru algorytmu, rozmiarów problemu, liczby powtórzeń, iteracji Mandelbrota itp.
* **Czytelne wyniki:** Prezentacja wyników w formie tabel w konsoli, w tym obliczone współczynniki przyspieszenia (speedup).
* **Generowanie wykresów:** Opcjonalne tworzenie wykresów słupkowych (za pomocą Matplotlib) porównujących czasy wykonania i przyspieszenie, zapisywanych do plików PNG.

## Wymagania

* Python (np. 3.8 lub nowszy)
* NumPy
* Numba
* Matplotlib (opcjonalnie, tylko jeśli chcesz generować wykresy - flaga `--plot`)
* **Dla testów GPU:**
    * Kompatybilna karta graficzna NVIDIA GPU
    * Zainstalowany [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (wersja kompatybilna z Twoją wersją Numba i sterownikami)
    * Zainstalowane sterowniki NVIDIA wspierające zainstalowaną wersję CUDA

## Instalacja

1.  **Sklonuj repozytorium:**
    ```bash
    git clone <URL_TWOJEGO_REPOZYTORIUM>
    cd <NAZWA_KATALOGU_REPOZYTORIUM>
    ```
    Zastąp `<URL_TWOJEGO_REPOZYTORIUM>` i `<NAZWA_KATALOGU_REPOZYTORIUM>` odpowiednimi wartościami.

2.  **Zainstaluj zależności:**
    Zaleca się utworzenie wirtualnego środowiska (np. `venv` lub `conda`).
    ```bash
    pip install numpy numba matplotlib
    ```
    *Uwaga: Instalacja Numba może pociągnąć za sobą odpowiednią wersję `llvmlite`.*

3.  **Konfiguracja CUDA (dla GPU):** Upewnij się, że masz poprawnie zainstalowany i skonfigurowany CUDA Toolkit oraz sterowniki NVIDIA. Numba powinna automatycznie wykryć Twoje urządzenie GPU, jeśli wszystko jest poprawnie skonfigurowane. Skrypt sprawdzi dostępność GPU przy uruchomieniu.

## Użycie

Główny skrypt to `main.py`. Możesz go uruchomić z różnymi argumentami linii poleceń.

### Podstawowe Użycie

Uruchomienie wszystkich benchmarków z domyślnymi rozmiarami, bez generowania wykresów:
```bash
python main.py
```

### Opcje Uruchomieniowe (Flagi)

Poniższa tabela opisuje dostępne argumenty linii poleceń:

| Flaga                          | Opis                                                                                                | Typ                      | Domyślnie          | Dostępne Opcje              |
| :----------------------------- | :-------------------------------------------------------------------------------------------------- | :----------------------- | :----------------- | :------------------------- |
| `--algorithm <nazwa>`          | Określa, który algorytm ma być testowany.                                                           | `tekst`                  | `mandelbrot`              | `mandelbrot`, |
| `--sizes_mandelbrot <W1> ...`| Lista szerokości `W` obrazu Mandelbrota (oddzielone spacjami). Wysokość obliczana automatycznie.         | `lista liczb całk.`      | `500 1000 2000 3000 5000`    | -                          |
| `--mandel_iter <liczba>`       | Maksymalna liczba iteracji dla zbioru Mandelbrota.                                                  | `liczba całkowita`       | `100`              | -                          |
| `--reps <liczba>`              | Liczba powtórzeń każdego pomiaru czasu (dla uśrednienia).                                           | `liczba całkowita`       | `5`                | -                          |
| `--warmup <liczba>`            | Liczba powtórzeń "rozgrzewkowych" przed pomiarem.                                                     | `liczba całkowita`       | `1`                | -                          |
| `--plot`                       | Włącza generowanie wykresów wyników (wymaga `matplotlib`).                                            | `flaga`                  | -                  | -                          |
| `--plot_dir <ścieżka>`         | Katalog zapisu wygenerowanych wykresów (jeśli użyto `--plot`). Tworzony, jeśli nie istnieje.            | `tekst`                  | `plots`            | -                          |

### Pomoc

Aby wyświetlić w terminalu pełną listę opcji wraz z ich opisami (bezpośrednio z `argparse`):
```bash
python main.py --help