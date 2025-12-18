"""
Исследование алгоритмов, имеющих несколько способов реализации
с целью выбора наиболее эффективного по времени

Замеряет время очень долго. Для ускорения запускал файл benchmarking.py (там то же самое)
через pypy3 из корневой директории

pypy3 benchmarking.py
Но его необходимо скачать и в нем установить numpy и matplotlib
"""

from implementation_choice import *
import matplotlib.pyplot as plt
import numpy as np
import time
import random

repeats_small = 2
repeats_huge = 1
repeats_pow = 2


def research_matmul():
    """
    Сравнение производительности 3 алгоритмов умножения матриц:
    - naive_mul - умножение по определению    ->  (Самый оптимальный по результатам)
    - block_mul - блочное умножение
    - Strassen_mul - умножение по алгоритму Штрассена
    """

    small_sizes = list(range(2, 16 + 1))
    huge_sizes = [20, 25, 30, 35, 40, 45, 50]

    sizes = small_sizes + huge_sizes    # объединено в одно, но прошлые графики сохранены

    # Исследование на небольших размерах (от 2 до 16 вкл)
    # и больших тоже, но с ограничениями для тех, что долго вычисляются (метод Штрассена)
    time_results_naive = []
    time_results_block = []
    time_results_Strassen = []

    for size in sizes:
        time_total_naive = 0
        time_total_block = 0
        time_total_Strassen = 0
        for _ in range(repeats_small):
            A = create_test_matrix(size)
            B = create_test_matrix(size)

            st_naive = time.perf_counter()
            res_naive = A.naive_mul(B)
            end_naive = time.perf_counter()
            time_total_naive += end_naive - st_naive

            st_block = time.perf_counter()
            res_block = A.blocked_mul(B)
            end_block = time.perf_counter()
            time_total_block += end_block - st_block

            # Ограничение на метод Штрассена
            if size <= 35:
                st_Strassen = time.perf_counter()
                res_Strassen = A.Strassen_mul(B)
                end_Strassen = time.perf_counter()
                time_total_Strassen += end_Strassen - st_Strassen
            else:
                time_total_Strassen += 0

        time_results_naive.append(time_total_naive / repeats_small)
        time_results_block.append(time_total_block / repeats_small)
        if size <= 35:
            time_results_Strassen.append(time_total_Strassen / repeats_small)
        else:
            time_results_Strassen.append(None)

    # Построение графиков
    # 1 Для n <= 16
    plt.figure(figsize=(10, 6))
    plt.plot(small_sizes, time_results_naive[:len(small_sizes)], 'go-', label="По определению", markersize=6)
    plt.plot(small_sizes, time_results_block[:len(small_sizes)], 'yo-', label="Блочное", markersize=6)
    plt.plot(small_sizes, time_results_Strassen[:len(small_sizes)], 'ro-', label="Штрассен", markersize=6)

    plt.xlabel("Размер матрицы (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Исследование умножения матриц")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("mul_small.png", dpi=300, bbox_inches='tight')
    plt.show()


    # 2 График с маленькими и большими матрицами
    Strassen_valid = [t for t in time_results_Strassen if t is not None]
    sizes_Str = [s for s, t in zip(sizes, time_results_Strassen) if t is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_results_naive, 'go-', label="По определению", markersize=6)
    plt.plot(sizes, time_results_block, 'yo-', label="Блочное", markersize=6)
    plt.plot(sizes_Str, Strassen_valid, 'ro-', label="Штрассен", markersize=6)

    plt.xlabel("Размер матрицы (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Исследование умножения матриц")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("mul_total.png", dpi=300, bbox_inches='tight')
    plt.show()


def research_det():
    """
    Сравнение производительности 3 алгоритмов поиска определителя матрицы:
    - det_definition - по определению (через перестановки)
    - det_Laplace - разложение по строке/столбцу, где побольше нулей
    - det_Gauss - через приведение к ступенчатому виду   -> (Самый оптимальный по результатам)

    * для n <= 3 выписаны формулы, поэтому длительность работы зависит только от операций над Q
    """

    # На случайных матрицах
    sizes = list(range(2, 16 + 1))

    # 1) Исследование на небольших размерах (от 2 до 16 вкл), но с ограничениями там, где долго считает
    time_results_def = []
    time_results_Laplace = []
    time_results_Gauss = []

    for size in sizes:
        time_total_def = 0
        time_total_Laplace = 0
        time_total_Gauss = 0

        # Для каждого размера выполняем несколько повторов
        for _ in range(repeats_small):
            A = create_test_matrix(size)

            # Определение через перестановки только для небольших размеров
            if size <= 6: # было 8. плохой график
                try:
                    st_def = time.perf_counter()
                    res_def = A.det_definition()
                    end_def = time.perf_counter()
                    time_total_def += end_def - st_def
                except Exception as e:
                    time_total_def += 0  # Или можно пропустить

            # Метод Лапласа только для размеров до 8  (было до 10 -> очень плохо, приложены график det_random_bad, det_random_bad2 с неудачными замерами)
            if size <= 8:
                try:
                    st_Laplace = time.perf_counter()
                    res_Laplace = A.det_Laplace()
                    end_Laplace = time.perf_counter()
                    time_total_Laplace += end_Laplace - st_Laplace
                except Exception as e:
                    time_total_Laplace += 0

            # Метод Гаусса для всех размеров
            try:
                st_Gauss = time.perf_counter()
                res_Gauss = A.det_Gauss()
                end_Gauss = time.perf_counter()
                time_total_Gauss += end_Gauss - st_Gauss
            except Exception as e:
                time_total_Gauss += 0

        # Сохраняем среднее время
        if size <= 6:
            time_results_def.append(time_total_def / max(1, repeats_small))
        else:
            time_results_def.append(None)

        if size <= 8:
            time_results_Laplace.append(time_total_Laplace / max(1, repeats_small))
        else:
            time_results_Laplace.append(None)

        time_results_Gauss.append(time_total_Gauss / repeats_small)

    # Построение графиков
    plt.figure(figsize=(10, 6))

    # Фильтруем значения None
    sizes_def = [s for s, t in zip(sizes[:6], time_results_def[:6]) if t is not None]
    times_def = [t for t in time_results_def[:6] if t is not None]
    if sizes_def:
        plt.plot(sizes_def, times_def, 'go-', label="По определению", markersize=6)

    sizes_laplace = [s for s, t in zip(sizes[:8], time_results_Laplace[:8]) if t is not None]
    times_laplace = [t for t in time_results_Laplace[:8] if t is not None]
    if sizes_laplace:
        plt.plot(sizes_laplace, times_laplace, 'yo-', label="Разложение по строке/столбцу", markersize=6)

    sizes_gauss = [s for s, t in zip(sizes, time_results_Gauss) if t is not None]
    times_gauss = [t for t in time_results_Gauss if t is not None]
    if sizes_gauss:
        plt.plot(sizes_gauss, times_gauss, 'ro-', label="Гаусс", markersize=6)

    plt.xlabel("Размер матрицы (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Исследование поиска определителя случайных матриц")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("det_random.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2) Конкретные матрицы, где много нулей: сравнение методов Лапласа и Гаусса

    # Создаем матрицы с разным процентом нулей
    zero_percentages = [0.1, 0.3, 0.5, 0.7]
    size_for_test = 8

    time_results_laplace_sparse = []
    time_results_gauss_sparse = []

    for zero_pct in zero_percentages:
        time_laplace = 0
        time_gauss = 0

        for _ in range(repeats_small):
            # Создаем разреженную матрицу
            A = create_sparse_matrix(size_for_test, zero_pct)

            try:
                st_laplace = time.perf_counter()
                A.det_Laplace()
                end_laplace = time.perf_counter()
                time_laplace += end_laplace - st_laplace
            except:
                time_laplace += 0

            st_gauss = time.perf_counter()
            A.det_Gauss()
            end_gauss = time.perf_counter()
            time_gauss += end_gauss - st_gauss

        time_results_laplace_sparse.append(time_laplace / repeats_small)
        time_results_gauss_sparse.append(time_gauss / repeats_small)
        print(
            f"Нулей: {zero_pct * 100}% | Лаплас: {time_laplace / repeats_small:.4f}с | Гаусс: {time_gauss / repeats_small:.4f}с")

    # График для разреженных матриц (Диаграмма)
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(zero_percentages))
    width = 0.35

    plt.bar(x_pos - width / 2, time_results_laplace_sparse, width, label='Лаплас', color='orange')
    plt.bar(x_pos + width / 2, time_results_gauss_sparse, width, label='Гаусс', color='red')

    plt.xlabel("Процент нулей в матрице")
    plt.ylabel("Время выполнения (сек)")
    plt.title(f"Сравнение методов для разреженных матриц (n={size_for_test})")
    plt.xticks(x_pos, [f"{int(p * 100)}%" for p in zero_percentages])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig("det_sparse.png", dpi=300, bbox_inches='tight')
    plt.show()



def research_inverse():
    """
    Сравнение производительности 2 алгоритмов поиска обратной матрицы:
    - Gauss_inv - по методу Гаусса ([A | E] ~ [E | A^-1], E - единичная)  -> (Самый оптимальный по результатам)
    - adjugate_inv - метод союзной матрицы
    """

    sizes = list(range(2, 16 + 1))  # Для метода союзной матрицы ограничимся 10

    time_results_adj = []
    time_results_Gauss = []
    successful_sizes_adj = []
    successful_sizes_gauss = []

    for size in sizes:
        time_total_adj = 0
        time_total_Gauss = 0
        count_adj = 0
        count_gauss = 0

        i = 0
        while i < repeats_small:
            A = create_test_matrix(size)

            # Проверяем, обратима ли матрица
            try:
                # Метод союзной матрицы (только до размера 10)
                if size <= 10:
                    st_adj = time.perf_counter()
                    res_adj = A.adjugate_inv()
                    end_adj = time.perf_counter()
                    time_total_adj += end_adj - st_adj
                    count_adj += 1


                # Метод Гаусса
                st_Gauss = time.perf_counter()
                res_Gauss = A.Gauss_inv()
                end_Gauss = time.perf_counter()
                time_total_Gauss += end_Gauss - st_Gauss
                count_gauss += 1

                i += 1

            except ValueError:
                continue  # Пропускаем вырожденные матрицы

        if count_adj > 0:
            time_results_adj.append(time_total_adj / count_adj)
            successful_sizes_adj.append(size)

        if count_gauss > 0:
            time_results_Gauss.append(time_total_Gauss / count_gauss)
            successful_sizes_gauss.append(size)

    # Построение графиков
    plt.figure(figsize=(10, 6))

    if successful_sizes_adj:
        plt.plot(successful_sizes_adj, time_results_adj, 'go-', label="Союзная матрица", markersize=6)

    if successful_sizes_gauss:
        plt.plot(successful_sizes_gauss, time_results_Gauss, 'yo-', label="Гаусс", markersize=6)

    plt.xlabel("Размер матрицы (n)")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Исследование вычисления обратной матрицы")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("inverse.png", dpi=300, bbox_inches='tight')
    plt.show()



def research_pow():
    """
    Сравнение производительности алгоритмов возведения матрицы в степень:
    - naive_pow - перемножить power-1 раз
    - quick_powering - алгоритм быстрого возведения в степень с оптимизациями
    - diag_and_pow_2x2 - диагонализация и возведение даигонализированной для матриц 2x2
    - analytical_2x2_pow - анаитические формулы для матриц 2x2
    - pow_diag - для уже диагональных матриц
    """

    print("\n=== Исследование возведения в степень ===")

    # 1) Сравнение naive_pow и quick_powering для разных размеров
    # quick_powering -> (Самый оптимальный по результатам)

    sizes_pow = [2, 4, 6]
    powers = [10, 50, 75]

    for power in powers:
        print(f"\n--- Степень = {power} ---")       # debug output

        time_naive_list = []
        time_quick_list = []
        size_list = []

        for size in sizes_pow:

            time_naive = 0
            time_quick = 0
            count = 0

            for _ in range(min(repeats_pow, 2)):  # Меньше повторов для возведения в степень
                A = create_test_matrix(size)

                # naive_pow
                if size <= 4 or power <= 10:  # Только для небольших случаев, так как очень долго считает
                    try:
                        st = time.perf_counter()
                        A.naive_pow(power)
                        end = time.perf_counter()
                        time_naive += end - st
                    except:
                        time_naive += 0

                # quick_powering
                try:
                    st = time.perf_counter()
                    A.quick_powering(power)
                    end = time.perf_counter()
                    time_quick += end - st
                    count += 1
                except:
                    time_quick += 0

            if count > 0:
                time_naive_list.append(time_naive / count if time_naive > 0 else None)
                time_quick_list.append(time_quick / count)
                size_list.append(size)

        # График для этой степени
        plt.figure(figsize=(10, 6))

        # Фильтруем None значения
        valid_indices = [i for i, t in enumerate(time_naive_list) if t is not None]
        if valid_indices:
            plt.plot([size_list[i] for i in valid_indices],
                     [time_naive_list[i] for i in valid_indices],
                     'ro-', label="Naive pow", markersize=6)

        plt.plot(size_list, time_quick_list, 'go-', label="Quick powering", markersize=6)

        plt.xlabel("Размер матрицы (n)")
        plt.ylabel("Время выполнения (сек)")
        plt.title(f"Возведение в степень {power}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"pow_size_power{power}.png", dpi=300, bbox_inches='tight')
        plt.show()


    # 2) Сравнение методов для матриц 2x2
    print("\n--- Методы для матриц 2x2 ---")

    powers_2x2 = [10, 50, 100, 200, 500, 1000]
    time_quick = []
    time_diag = []
    time_analytical = []

    for power in powers_2x2:
        t_quick = 0
        t_diag = 0
        t_analytical = 0
        count = 0

        for _ in range(repeats_pow):
            # Создаем матрицу 2x2, которую можно диагонализовать (эксперементально найдены заранее в implementation_choice)
            A = create_diagonizable_2x2_common()

            try:
                # quick_powering
                st = time.perf_counter()
                A.quick_powering(power)
                end = time.perf_counter()
                t_quick += end - st

                # diag_and_pow_2x2
                st = time.perf_counter()
                A.diag_and_pow_2x2(power)
                end = time.perf_counter()
                t_diag += end - st

                # analytical_2x2_pow
                st = time.perf_counter()
                A.analytical_2x2_pow(power)
                end = time.perf_counter()
                t_analytical += end - st

                count += 1
            except:
                continue

        if count > 0:
            time_quick.append(t_quick / count)
            time_diag.append(t_diag / count)
            time_analytical.append(t_analytical / count)
        else:
            # Если не удалось, используем обычную матрицу (перестраховка)
            A = create_test_matrix(2)
            # Только quick_powering гарантированно работает
            st = time.perf_counter()
            A.quick_powering(power)
            end = time.perf_counter()
            time_quick.append((end - st) / repeats_pow)
            time_diag.append(None)
            time_analytical.append(None)

    # График для матриц 2x2
    plt.figure(figsize=(10, 6))
    plt.plot(powers_2x2, time_quick, 'ro-', label="Quick powering", markersize=6)

    # Фильтруем None для diag
    valid_diag = [(p, t) for p, t in zip(powers_2x2, time_diag) if t is not None]
    if valid_diag:
        plt.plot([p for p, _ in valid_diag], [t for _, t in valid_diag],
                 'go-', label="Диагонализация", markersize=6)

    # Фильтруем None для analytical
    valid_analytical = [(p, t) for p, t in zip(powers_2x2, time_analytical) if t is not None]
    if valid_analytical:
        plt.plot([p for p, _ in valid_analytical], [t for _, t in valid_analytical],
                 'bo-', label="Аналитические формулы", markersize=6)

    plt.xlabel("Степень")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Методы возведения в степень для матриц 2x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("pow_2x2_methods_general.png", dpi=300, bbox_inches='tight')
    plt.show()


    # ОСОБЫЙ СЛУЧАЙ: C = 0

    time_quick = []
    time_diag = []
    time_analytical = []

    for power in powers_2x2:
        t_quick = 0
        t_diag = 0
        t_analytical = 0
        count = 0

        for _ in range(repeats_pow):
            # Создаем матрицу 2x2, которую можно диагонализовать
            A = create_diagonizable_2x2_special()

            try:
                # quick_powering
                st = time.perf_counter()
                A.quick_powering(power)
                end = time.perf_counter()
                t_quick += end - st

                # diag_and_pow_2x2
                st = time.perf_counter()
                A.diag_and_pow_2x2(power)
                end = time.perf_counter()
                t_diag += end - st

                # analytical_2x2_pow
                st = time.perf_counter()
                A.analytical_2x2_pow(power)
                end = time.perf_counter()
                t_analytical += end - st

                count += 1
            except:
                continue

        if count > 0:
            time_quick.append(t_quick / count)
            time_diag.append(t_diag / count)
            time_analytical.append(t_analytical / count)
        else:
            # Если не удалось, используем обычную матрицу (перестраховка)
            A = create_test_matrix(2)
            # Только quick_powering гарантированно работает
            st = time.perf_counter()
            A.quick_powering(power)
            end = time.perf_counter()
            time_quick.append((end - st) / repeats_pow)
            time_diag.append(None)
            time_analytical.append(None)

    # График для матриц 2x2
    plt.figure(figsize=(10, 6))
    plt.plot(powers_2x2, time_quick, 'ro-', label="Quick powering", markersize=6)

    # Фильтруем None для diag
    valid_diag = [(p, t) for p, t in zip(powers_2x2, time_diag) if t is not None]
    if valid_diag:
        plt.plot([p for p, _ in valid_diag], [t for _, t in valid_diag],
                 'go-', label="Диагонализация", markersize=6)

    # Фильтруем None для analytical
    valid_analytical = [(p, t) for p, t in zip(powers_2x2, time_analytical) if t is not None]
    if valid_analytical:
        plt.plot([p for p, _ in valid_analytical], [t for _, t in valid_analytical],
                 'bo-', label="Аналитические формулы", markersize=6)

    plt.xlabel("Степень")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Методы возведения в степень для матриц 2x2, особый случай с=0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("pow_2x2_methods_special.png", dpi=300, bbox_inches='tight')
    plt.show()



    # 3) Возведение уже диагональных матриц в степень
    print("\n--- Диагональные матрицы ---")

    sizes_diag = [2, 4, 8, 16, 32, 64, 100]
    power_diag = 100

    time_diag_pow = []
    time_quick_diag = []

    for size in sizes_diag:
        t_diag = 0
        t_quick = 0

        for _ in range(repeats_pow):
            # Создаем диагональную матрицу
            A = create_diagonal_matrix(size)

            # pow_diag
            st = time.perf_counter()
            A.pow_diag(power_diag)
            end = time.perf_counter()
            t_diag += end - st

            # quick_powering для сравнения
            st = time.perf_counter()
            A.quick_powering(power_diag)
            end = time.perf_counter()
            t_quick += end - st

        time_diag_pow.append(t_diag / repeats_pow)
        time_quick_diag.append(t_quick / repeats_pow)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes_diag, time_diag_pow, 'ro-', label="pow_diag", markersize=6)
    plt.plot(sizes_diag, time_quick_diag, 'go-', label="quick_powering", markersize=6)

    plt.xlabel("Размер диагональной матрицы")
    plt.ylabel("Время выполнения (сек)")
    plt.title(f"Возведение диагональных матриц в степень {power_diag}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("pow_diagonal.png", dpi=300, bbox_inches='tight')
    plt.show()


def research_poly():
    """
    Исследование вычисления значения полинома от матрицы в зависимости от полинома
    Рассматриваем разреженные и плотные, и какой из двух алгоритмов где и как себя проявляет
    - call_sequential
    - call_binary_sparse
    """
    sizes = [2, 4, 6]

    time_thick_poly_sequential_method = []
    time_thick_poly_binary_method = []
    deg_thick = 30
    sparsity_thick = 0.1

    # 1 для плотного полинома 2 метода
    for size in sizes:
        t_sequential_method = 0
        t_binary_method = 0
        coeff1 = create_polynomial_coeffs(deg_thick, size, sparsity_thick)   # для плотного полинома степени <= deg_thick

        P_thick = MatrixPolynomial(len(coeff1) - 1, coeff1)

        for _ in range(repeats_huge):
            A = create_test_matrix(size)

            st = time.perf_counter()
            P_thick.call_binary_sparse(A)
            end = time.perf_counter()
            t_binary_method += end - st

            st = time.perf_counter()
            P_thick.call_sequential(A)
            end = time.perf_counter()
            t_sequential_method += end - st

        time_thick_poly_binary_method += [t_binary_method / repeats_huge]
        time_thick_poly_sequential_method += [t_sequential_method / repeats_huge]

    # plot:
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_thick_poly_binary_method, 'ro-', label="binary_method", markersize=6)
    plt.plot(sizes, time_thick_poly_sequential_method, 'go-', label="sequential_method", markersize=6)

    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (сек)")
    plt.title(f"Плотный полином степени <= {deg_thick}, разреженности {sparsity_thick * 100}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("thick_poly.png", dpi=300, bbox_inches='tight')
    plt.show()


    # 2 для разреженного полинома 2 метода
    time_sparse_poly_sequential_method = []
    time_sparse_poly_binary_method = []
    deg_sparse = 50
    sparsity_sparse = 0.8

    for size in sizes:
        t_sequential_method = 0
        t_binary_method = 0
        coeff2 = create_polynomial_coeffs(deg_sparse, size, sparsity_sparse)  # для разреженного полинома степени <= deg_sparse
        P_sparse = MatrixPolynomial(len(coeff2) - 1, coeff2)

        for _ in range(repeats_huge):
            A = create_test_matrix(size)

            st = time.perf_counter()
            P_sparse.call_binary_sparse(A)
            end = time.perf_counter()
            t_binary_method += end - st

            st = time.perf_counter()
            P_sparse.call_sequential(A)
            end = time.perf_counter()
            t_sequential_method += end - st

        time_sparse_poly_binary_method += [t_binary_method / repeats_huge]
        time_sparse_poly_sequential_method += [t_sequential_method / repeats_huge]

    # plot:
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_sparse_poly_binary_method, 'ro-', label="binary_method", markersize=6)
    plt.plot(sizes, time_sparse_poly_sequential_method, 'go-', label="sequential_method", markersize=6)

    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (сек)")
    plt.title(f"Разреженный полином степени <= {deg_sparse}, разреженности {sparsity_sparse*100}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("sparse_poly.png", dpi=300, bbox_inches='tight')
    plt.show()




# Вспомогательные функции для создания тестовых данных
def create_sparse_matrix(n, zero_probability=0.3):
    """Создание разреженной матрицы с заданной вероятностью нулей"""

    arr = []
    for i in range(n):
        row = []
        for j in range(n):
            if random.random() < zero_probability:
                # Нулевой элемент
                row.append(ZERO_RATIONAL)
            else:
                # Ненулевой элемент
                a = random.randint(-10, 10)
                b = random.randint(1, 10)

                sign = 1 if a < 0 else 0
                digits = [int(d) for d in str(abs(a))]
                numerator = Integer(sign, len(digits), digits)

                denom_digits = [int(d) for d in str(b)]
                denominator = Natural(1, denom_digits)

                row.append(Rational(numerator, denominator))
        arr.append(row)

    return Matrix(n, arr)


def create_diagonal_matrix(n):
    """Создание диагональной матрицы"""

    arr = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a = random.randint(1, 10)
        b = random.randint(1, 10)

        sign = 0  # положительное
        digits = [int(d) for d in str(abs(a))]
        numerator = Integer(sign, len(digits), digits)

        denom_digits = [int(d) for d in str(b)]
        denominator = Natural(1, denom_digits)

        arr[i][i] = Rational(numerator, denominator)

    return Matrix(n, arr)


def create_diagonizable_2x2_common():
    """Создание матрицы 2x2, которую можно диагонализовать над Q"""

    # Рандомная даигонализуемая из следующих, подобранных заранее и точно диагонализируемых:

    A2 = Matrix(2, [
        [Rational(Integer(0, 1, [3]), Natural(1, [2])),
         Rational(Integer(0, 1, [1]), Natural(1, [3]))],
        [Rational(Integer(0, 1, [1]), Natural(1, [2])),
         Rational(Integer(0, 1, [1]), Natural(1, [9]))]
    ])

    A3 = Matrix(2, [
        [Rational(Integer(1, 1, [3]), Natural(1, [1])),
         Rational(Integer(0, 1, [1]), Natural(1, [2]))],
        [Rational(Integer(0, 1, [4]), Natural(1, [3])),
         Rational(Integer(1, 1, [2]), Natural(1, [9]))]
    ])


    A4 = Matrix(2, [
        [Rational(Integer(1, 1, [2]), Natural(1, [3])),
         Rational(Integer(1, 1, [1]), Natural(1, [1]))],
        [Rational(Integer(1, 1, [5]), Natural(1, [2])),
         Rational(Integer(0, 1, [3]), Natural(1, [2]))]
    ])


    A5 = Matrix(2, [
        [Rational(Integer(1, 1, [5]), Natural(1, [1])),
         Rational(Integer(0, 1, [0]), Natural(1, [1]))],
        [Rational(Integer(0, 1, [5]), Natural(1, [7])),
         Rational(Integer(1, 1, [1]), Natural(1, [8]))]
    ])


    A6 = Matrix(2, [
        [Rational(Integer(0, 1, [5]), Natural(1, [4])),
         Rational(Integer(0, 1, [0]), Natural(1, [1]))],
        [Rational(Integer(0, 1, [5]), Natural(1, [6])),
         Rational(Integer(0, 1, [5]), Natural(1, [2]))]
    ])


    A7 = Matrix(2, [
        [Rational(Integer(1, 1, [9]), Natural(1, [5])),
         Rational(Integer(0, 1, [0]), Natural(1, [1]))],
        [Rational(Integer(0, 1, [1]), Natural(1, [5])),
         Rational(Integer(0, 1, [1]), Natural(1, [1]))]
    ])

    return random.choice([A2, A3, A4, A5, A6, A7])


def create_diagonizable_2x2_special():
    """
    Рассматриваем матрицы [[a, b], [c, d]], приводимые к диагональному виду над Q, но с C=0
    """

    # Рандомная даигонализуемая из следующих:

    A1 = Matrix(2, [
        [Rational(Integer(1, 1, [5]), Natural(1, [8])),
         Rational(Integer(0, 1, [4]), Natural(1, [1]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(0, 1, [2]), Natural(1, [3]))]
    ])

    A2 = Matrix(2, [
        [Rational(Integer(0, 2, [1, 0]), Natural(1, [7])),
         Rational(Integer(1, 1, [8]), Natural(1, [9]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(0, 1, [3]), Natural(1, [5]))]
    ])

    A3 = Matrix(2, [
        [Rational(Integer(0, 1, [4]), Natural(1, [1])),
         Rational(Integer(0, 1, [8]), Natural(1, [3]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(1, 1, [3]), Natural(1, [2]))]
    ])


    A4 = Matrix(2, [
        [Rational(Integer(1, 1, [9]), Natural(1, [1])),
         Rational(Integer(0, 1, [1]), Natural(1, [1]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(0, 1, [1]), Natural(1, [2]))]
    ])


    A5 = Matrix(2, [
        [Rational(Integer(0, 1, [4]), Natural(1, [5])),
         Rational(Integer(1, 1, [1]), Natural(2, [1, 0]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(1, 1, [3]), Natural(1, [7]))]
    ])


    A6 = Matrix(2, [
        [Rational(Integer(0, 1, [9]), Natural(1, [8])),
         Rational(Integer(0, 1, [3]), Natural(1, [4]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(0, 1, [1]), Natural(1, [1]))]
    ])


    A7 = Matrix(2, [
        [Rational(Integer(0, 1, [1]), Natural(1, [1])),
         Rational(Integer(1, 1, [1]), Natural(1, [5]))],
        [Rational(Integer(0, 1, [0]), Natural(1, [1])),
         Rational(Integer(1, 1, [3]), Natural(2, [1, 0]))]
    ])

    return random.choice([A1, A2, A3, A4, A5, A6, A7])


def create_polynomial_coeffs(degree, matrix_size, zero_probability=0.5):
    """Создание коэффициентов полинома с заданной вероятностью нулей"""

    coeffs = []
    for i in range(degree + 1):
        if random.random() < zero_probability:
            # Нулевая матрица
            coeffs.append(Matrix.zero(matrix_size))
        else:
            # Случайная матрица
            coeffs.append(create_test_matrix(matrix_size, max_num=5))

    return coeffs


if __name__ == "__main__":
    # research_matmul()      # --done
    # research_det()         # --done
    # research_inverse()     # -- done
    # research_pow()         # -- done
    research_poly()        # -- done