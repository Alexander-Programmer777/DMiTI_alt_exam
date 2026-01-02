"""
Комплексное тестирование матричных операций со сравнением с NumPy (эталон)  -- done
"""

import pytest
import numpy as np
from random import randint

from Matrixes.main import ZERO_RATIONAL
from N.Natural import Natural
from Q.Rational import Rational
from Z.Integer import Integer
from Matrixes.implementation_choice import Matrix
from Matrixes.implementation_choice import MatrixPolynomial


# =================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===================

def create_rational_from_int(value: int):
    """Создать Rational из целого числа"""
    if value >= 0:
        sign = 0
    else:
        sign = 1
        value = abs(value)

    digits = [int(d) for d in str(value)]
    integer = Integer(sign, len(digits) - 1, digits)
    natural = Natural(1, [1])
    return Rational(integer, natural)


def create_matrix_from_list(arr):
    """Создать Matrix из списка списков целых чисел"""
    size = len(arr)
    rational_arr = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(create_rational_from_int(arr[i][j]))
        rational_arr.append(row)
    return Matrix(size, rational_arr)


def matrix_to_numpy(matrix):
    """Конвертировать Matrix в numpy array (float)"""
    np_arr = np.zeros((matrix.size, matrix.size), dtype=float)
    for i in range(matrix.size):
        for j in range(matrix.size):
            # Конвертируем Rational в float
            # Предполагаем, что numerator и denominator можно конвертировать
            num = int(str(matrix.arr[i][j].numerator))
            den = int(str(matrix.arr[i][j].denominator))
            np_arr[i][j] = num / den
    return np_arr


def numpy_to_matrix(np_arr):
    """Конвертировать numpy array в Matrix (округляем до целых)"""
    size = np_arr.shape[0]
    arr = []
    for i in range(size):
        row = []
        for j in range(size):
            # Округляем до целых для простоты сравнения
            row.append(create_rational_from_int(int(round(np_arr[i][j]))))
        arr.append(row)
    return Matrix(size, arr)


def matrices_equal(m1, m2, tolerance=1e-10):
    """Сравнение двух матриц с допуском (через NumPy)"""
    if m1.size != m2.size:
        return False

    m1_np = matrix_to_numpy(m1)
    m2_np = matrix_to_numpy(m2)
    return np.allclose(m1_np, m2_np, rtol=tolerance)


def compare_with_numpy(matrix_op, np_op, operation_name=""):
    """Сравнить результат операции с NumPy"""
    matrix_np = matrix_to_numpy(matrix_op)
    diff = np.abs(matrix_np - np_op)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)

    if max_diff > 1e-10:
        print(f"\n{operation_name}: Max diff = {max_diff:.2e}, Avg diff = {avg_diff:.2e}")
        if max_diff > 1e-5:
            print(f"WARNENG: Large difference in {operation_name}")

    return max_diff < 1e-8


# =================== ФИКСТУРЫ ===================

@pytest.fixture
def small_matrix():
    """Фикстура для маленькой матрицы 2x2"""
    return create_matrix_from_list([[1, 2], [3, 4]])


@pytest.fixture
def zero_matrix_3x3():
    """Фикстура для нулевой матрицы 3x3"""
    return Matrix.zero(3)


@pytest.fixture
def identity_matrix_3x3():
    """Фикстура для единичной матрицы 3x3"""
    return Matrix.identity(3)


@pytest.fixture
def diagonal_matrix():
    """Фикстура для диагональной матрицы"""
    return create_matrix_from_list([[2, 0, 0], [0, 3, 0], [0, 0, 4]])


@pytest.fixture
def triangular_matrix():
    """Фикстура для верхнетреугольной матрицы"""
    return create_matrix_from_list([[1, 2, 3], [0, 4, 5], [0, 0, 6]])


@pytest.fixture
def invertible_matrix():
    """Фикстура для обратимой матрицы"""
    return create_matrix_from_list([[4, 7], [2, 6]])


# =================== ФИКСТУРЫ ДЛЯ БОЛЬШИХ МАТРИЦ ===================

@pytest.fixture
def large_matrix_8x8():
    """Фикстура для матрицы 8x8 (степень двойки)"""
    return create_matrix_from_list([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32],
        [33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48],
        [49, 50, 51, 52, 53, 54, 55, 56],
        [57, 58, 59, 60, 61, 62, 63, 64]
    ])


@pytest.fixture
def large_matrix_10x10():
    """Фикстура для матрицы 10x10 (не степень двойки)"""
    arr = [[(i * 10 + j + 1) for j in range(10)] for i in range(10)]
    return create_matrix_from_list(arr)


@pytest.fixture
def random_large_matrix():
    """Фикстура для случайной большой матрицы"""
    return create_random_matrix(6, max_value=10)


def create_random_matrix(size, max_value=10):
    """Создать случайную матрицу заданного размера"""
    arr = []
    for i in range(size):
        row = []
        for j in range(size):
            value = randint(-max_value, max_value)
            row.append(create_rational_from_int(value))
        arr.append(row)
    return Matrix(size, arr)


# =================== ТЕСТЫ БАЗОВЫХ МЕТОДОВ ===================

class TestConstructor:
    """Тесты конструктора Matrix"""

    def test_valid_creation(self):
        """Тест корректного создания матрицы"""
        matrix = create_matrix_from_list([[1, 2], [3, 4]])
        assert matrix.size == 2
        assert matrix.arr[0][0] == create_rational_from_int(1)
        assert matrix.arr[1][1] == create_rational_from_int(4)

    def test_non_square_matrix(self):
        """Тест создания неквадратной матрицы"""
        with pytest.raises(ValueError, match="Матрица должна быть квадратной"):
            arr = [[create_rational_from_int(1), create_rational_from_int(2)],
                   [create_rational_from_int(3), create_rational_from_int(4), create_rational_from_int(5)]]
            Matrix(2, arr)

    def test_size_mismatch(self):
        """Тест несоответствия указанного размера"""
        arr = [[create_rational_from_int(1), create_rational_from_int(2)],
               [create_rational_from_int(3), create_rational_from_int(4)]]
        with pytest.raises(ValueError, match="Указанный размер 3 не соответствует реальному 2"):
            Matrix(3, arr)


class TestBasicOperations:
    """Тесты базовых операций с сравнением с NumPy"""

    def test_add_basic(self, small_matrix):
        """Тест сложения матриц с сравнением с NumPy"""
        m2 = create_matrix_from_list([[5, 6], [7, 8]])

        # Наша реализация
        result = small_matrix + m2

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        m2_np = matrix_to_numpy(m2)
        expected_np = small_np + m2_np

        # Сравнение
        assert compare_with_numpy(result, expected_np, "Addition")

        # Проверка точного значения
        expected = create_matrix_from_list([[6, 8], [10, 12]])
        assert matrices_equal(result, expected)

    def test_add_zero_matrix(self, small_matrix):
        """Тест сложения с нулевой матрицей"""
        zero_2x2 = Matrix.zero(2)
        result = small_matrix + zero_2x2

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        zero_np = np.zeros((2, 2))
        expected_np = small_np + zero_np

        assert compare_with_numpy(result, expected_np, "Add zero")
        assert matrices_equal(result, small_matrix)

    def test_sub_basic(self, small_matrix):
        """Тест вычитания матриц с сравнением с NumPy"""
        m2 = create_matrix_from_list([[5, 6], [7, 8]])

        # Наша реализация
        result = small_matrix - m2

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        m2_np = matrix_to_numpy(m2)
        expected_np = small_np - m2_np

        # Сравнение
        assert compare_with_numpy(result, expected_np, "diff")

        # Проверка точного значения
        expected = create_matrix_from_list([[-4, -4], [-4, -4]])
        assert matrices_equal(result, expected)

    def test_multiply_by_const(self, small_matrix):
        """Тест умножения на константу"""
        const = create_rational_from_int(2)
        result = small_matrix.multiply_by_const(const)

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        expected_np = small_np * 2

        assert compare_with_numpy(result, expected_np, "Multiply by const")

        # Проверка точного значения
        expected = create_matrix_from_list([[2, 4], [6, 8]])
        assert matrices_equal(result, expected)

    def test_trace(self, small_matrix, identity_matrix_3x3):
        """Тест вычисления следа"""
        # Для [[1,2],[3,4]] trace = 1+4=5
        trace = small_matrix.trace()

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        expected_trace = np.trace(small_np)

        # Конвертируем Rational в float для сравнения
        trace_float = float(str(trace.numerator)) / float(str(trace.denominator))
        assert abs(trace_float - expected_trace) < 1e-10

        # Для единичной 3x3 trace = 3
        trace_identity = identity_matrix_3x3.trace()
        assert trace_identity == create_rational_from_int(3)

    def test_transpose(self, small_matrix):
        """Тест транспонирования с сравнением с NumPy"""
        result = small_matrix.transpose()

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        expected_np = small_np.T

        assert compare_with_numpy(result, expected_np, "Transpose")

        # Проверка точного значения
        expected = create_matrix_from_list([[1, 3], [2, 4]])
        assert matrices_equal(result, expected)

        # Двойное транспонирование возвращает исходную матрицу
        assert matrices_equal(result.transpose(), small_matrix)


# =================== ТЕСТЫ УМНОЖЕНИЯ МАТРИЦ ===================

class TestMatrixMultiplication:
    """Тесты умножения матриц с сравнением с NumPy"""

    def test_naive_mul_basic(self):
        """Базовый тест наивного умножения"""
        A = create_matrix_from_list([[1, 2], [3, 4]])
        B = create_matrix_from_list([[2, 0], [1, 2]])

        result = A.naive_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(result, expected_np, "Naive multiplication")

        # Проверка точного значения
        expected = create_matrix_from_list([[4, 4], [10, 8]])
        assert matrices_equal(result, expected)

    def test_naive_mul_with_identity(self, small_matrix):
        """Тест умножения на единичную матрицу"""
        identity_2x2 = Matrix.identity(2)
        result = small_matrix.naive_mul(identity_2x2)

        # NumPy
        small_np = matrix_to_numpy(small_matrix)
        identity_np = np.eye(2)
        expected_np = np.dot(small_np, identity_np)

        assert compare_with_numpy(result, expected_np, "Multiply by identity")
        assert matrices_equal(result, small_matrix)

    def test_block_mul_basic(self):
        """Базовый тест блочного умножения для маленькой матрицы"""
        A = create_matrix_from_list([[1, 2], [3, 4]])
        B = create_matrix_from_list([[2, 0], [1, 2]])

        result = A.blocked_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(result, expected_np, "Block multiplication 2x2")

        # Проверка точного значения
        expected = create_matrix_from_list([[4, 4], [10, 8]])
        assert matrices_equal(result, expected)

    def test_block_mul_with_identity(self):
        """Тест блочного умножения на единичную матрицу"""
        A = create_matrix_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        identity = Matrix.identity(3)

        result = A.blocked_mul(identity)

        # NumPy
        A_np = matrix_to_numpy(A)
        identity_np = np.eye(3)
        expected_np = np.dot(A_np, identity_np)

        assert compare_with_numpy(result, expected_np, "Block multiply by identity")
        assert matrices_equal(result, A)

    def test_block_mul_with_zero(self):
        """Тест блочного умножения на нулевую матрицу"""
        A = create_matrix_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        zero_mat = Matrix.zero(3)

        result = A.blocked_mul(zero_mat)
        expected = Matrix.zero(3)

        assert matrices_equal(result, expected)


    def test_block_mul_large_matrix(self):
        """Тест блочного умножения для большой матрицы"""
        # Создаём большую матрицу 16x16 (кратно блоку 8)
        n = 16
        A = create_random_matrix(n)  # рандомная
        B = create_random_matrix(n)

        # Проверяем все методы умножения
        naive_result = A.naive_mul(B)
        block_result = A.blocked_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(block_result, expected_np, f"Block multiplication {n}x{n}")
        assert matrices_equal(naive_result, block_result)


    def test_strassen_mul_small(self):
        """Тест умножения Штрассена для маленьких матриц"""
        A = create_matrix_from_list([[1, 2], [3, 4]])
        B = create_matrix_from_list([[5, 6], [7, 8]])

        naive_result = A.naive_mul(B)
        strassen_result = A.Strassen_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(strassen_result, expected_np, "Strassen multiplication")

        # Должны давать одинаковый результат
        assert matrices_equal(naive_result, strassen_result)

    def test_strassen_mul_large_power_of_two(self, large_matrix_8x8):
        """Тест умножения Штрассена для больших матриц степени двойки"""
        A = large_matrix_8x8
        B = create_matrix_from_list([
            [64, 63, 62, 61, 60, 59, 58, 57],
            [56, 55, 54, 53, 52, 51, 50, 49],
            [48, 47, 46, 45, 44, 43, 42, 41],
            [40, 39, 38, 37, 36, 35, 34, 33],
            [32, 31, 30, 29, 28, 27, 26, 25],
            [24, 23, 22, 21, 20, 19, 18, 17],
            [16, 15, 14, 13, 12, 11, 10, 9],
            [8, 7, 6, 5, 4, 3, 2, 1]
        ])

        naive_result = A.naive_mul(B)
        strassen_result = A.Strassen_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(strassen_result, expected_np, "Strassen large")
        assert matrices_equal(naive_result, strassen_result)

    def test_strassen_mul_large_non_power_of_two(self, large_matrix_10x10):
        """Тест умножения Штрассена для больших матриц не степени двойки"""
        A = large_matrix_10x10
        B = create_matrix_from_list([[1] * 10 for _ in range(10)])

        naive_result = A.naive_mul(B)
        strassen_result = A.Strassen_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(strassen_result, expected_np, "Strassen non-power of two")
        assert matrices_equal(naive_result, strassen_result)


# =================== ТЕСТЫ ВЫЧИСЛЕНИЯ ОПРЕДЕЛИТЕЛЯ ===================

class TestDeterminant:
    """Тесты методов вычисления определителя с сравнением с NumPy"""

    def test_det_formula_1x1(self):
        """Тест определителя 1x1"""
        matrix = create_matrix_from_list([[5]])
        det = matrix.det_formula()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_det = np.linalg.det(matrix_np)

        det_float = float(str(det.numerator)) / float(str(det.denominator))
        assert abs(det_float - expected_det) < 1e-10

    def test_det_formula_2x2(self):
        """Тест определителя 2x2"""
        matrix = create_matrix_from_list([[1, 2], [3, 4]])
        det = matrix.det_formula()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_det = np.linalg.det(matrix_np)

        det_float = float(str(det.numerator)) / float(str(det.denominator))
        assert abs(det_float - expected_det) < 1e-10

    def test_det_gauss_3x3(self):
        """Тест определителя методом Гаусса"""
        matrix = create_matrix_from_list([[2, 1, 3], [4, 5, 6], [7, 8, 9]])
        det_gauss = matrix.det_Gauss()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_det = np.linalg.det(matrix_np)

        det_float = float(str(det_gauss.numerator)) / float(str(det_gauss.denominator))
        assert abs(det_float - expected_det) < 1e-10

    def test_det_gauss_triangular(self, triangular_matrix):
        """Тест определителя для треугольной матрицы"""
        det_gauss = triangular_matrix.det_Gauss()

        # NumPy
        matrix_np = matrix_to_numpy(triangular_matrix)
        expected_det = np.linalg.det(matrix_np)

        det_float = float(str(det_gauss.numerator)) / float(str(det_gauss.denominator))
        assert abs(det_float - expected_det) < 1e-10

    def test_det_gauss_large(self, random_large_matrix):
        """Тест определителя для больших матриц"""
        matrix = random_large_matrix
        det_gauss = matrix.det_Gauss()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_det = np.linalg.det(matrix_np)

        det_float = float(str(det_gauss.numerator)) / float(str(det_gauss.denominator))

        # Для больших матриц допуск больше из-за ошибок округления
        tolerance = 1e-6
        assert abs(det_float - expected_det) < tolerance, \
            f"Det mismatch: {det_float} vs {expected_det}, diff = {abs(det_float - expected_det)}"

    def test_all_det_methods_consistency(self):
        """Тест согласованности всех методов вычисления определителя"""
        matrix = create_matrix_from_list([[2, 3, 1], [4, 1, 5], [7, 2, 3]])

        # NumPy для сравнения
        matrix_np = matrix_to_numpy(matrix)
        np_det = np.linalg.det(matrix_np)

        # Получаем определители всеми методами
        det_formula = matrix.det_formula()
        det_definition = matrix.det_definition()
        det_laplace = matrix.det_Laplace()
        det_gauss = matrix.det_Gauss()

        # Конвертируем в float для сравнения
        dets_float = []
        for det in [det_formula, det_definition, det_laplace, det_gauss]:
            dets_float.append(float(str(det.numerator)) / float(str(det.denominator)))

        # Все должны быть равны с NumPy
        for i, det_float in enumerate(dets_float):
            assert abs(det_float - np_det) < 1e-10, \
                f"Method {i} mismatch: {det_float} vs {np_det}"


# =================== ТЕСТЫ ОБРАТНЫХ МАТРИЦ ===================

class TestInverseMatrix:
    """Тесты методов нахождения обратной матрицы с сравнением с NumPy"""

    def test_gauss_inv_2x2(self, invertible_matrix):
        """Тест обращения методом Гаусса для 2x2"""
        inv = invertible_matrix.Gauss_inv()

        # NumPy
        matrix_np = matrix_to_numpy(invertible_matrix)
        try:
            expected_inv_np = np.linalg.inv(matrix_np)
            assert compare_with_numpy(inv, expected_inv_np, "Gauss inverse")
        except np.linalg.LinAlgError:
            pytest.skip("Matrix is singular in NumPy")

        # Проверяем, что A * A^(-1) = E
        product = invertible_matrix.naive_mul(inv)

        product_np = matrix_to_numpy(product)
        identity_np = np.eye(2)

        assert np.allclose(product_np, identity_np, rtol=1e-10)

    def test_gauss_inv_singular(self):
        """Тест попытки обращения вырожденной матрицы"""
        singular_matrix = create_matrix_from_list([[1, 2], [2, 4]])  # det = 0

        # NumPy
        matrix_np = matrix_to_numpy(singular_matrix)
        with np.testing.assert_raises(np.linalg.LinAlgError):
            np.linalg.inv(matrix_np)

        # Наша реализация
        with pytest.raises(ValueError, match="Матрица вырождена"):
            singular_matrix.Gauss_inv()

    def test_adjugate_inv_2x2(self, invertible_matrix):
        """Тест обращения методом алгебраических дополнений"""
        inv_adj = invertible_matrix.adjugate_inv()
        inv_gauss = invertible_matrix.Gauss_inv()

        # Оба метода должны давать одинаковый результат
        assert matrices_equal(inv_adj, inv_gauss)

        # NumPy для проверки
        matrix_np = matrix_to_numpy(invertible_matrix)
        expected_inv_np = np.linalg.inv(matrix_np)

        assert compare_with_numpy(inv_adj, expected_inv_np, "Adjugate inverse")

    def test_gauss_inv_large(self):
        """Тест обращения больших диагональных матриц"""
        # Создаем диагональную матрицу 5x5
        size = 5
        arr = [[0] * size for _ in range(size)]
        for i in range(size):
            arr[i][i] = i + 2  # Не нули, чтобы была обратима
        matrix = create_matrix_from_list(arr)

        inv = matrix.Gauss_inv()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_inv_np = np.linalg.inv(matrix_np)

        assert compare_with_numpy(inv, expected_inv_np, "Large diagonal inverse")

        # Проверяем, что A * A^(-1) = E
        product = matrix.naive_mul(inv)
        product_np = matrix_to_numpy(product)
        identity_np = np.eye(size)

        assert np.allclose(product_np, identity_np, rtol=1e-10)


# =================== ТЕСТЫ ВОЗВЕДЕНИЯ В СТЕПЕНЬ ===================

class TestMatrixPower:
    """Тесты возведения матрицы в степень с сравнением с NumPy"""

    def test_naive_pow_positive(self):
        """Тест наивного возведения в положительную степень"""
        matrix = create_matrix_from_list([[1, 2], [3, 4]])
        power = 3

        result = matrix.naive_pow(power)

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_np = np.linalg.matrix_power(matrix_np, power)

        assert compare_with_numpy(result, expected_np, f"Naive power {power}")

    def test_quick_powering_basic(self):
        """Тест быстрого возведения в степень"""
        matrix = create_matrix_from_list([[1, 2], [3, 4]])

        for power in [0, 1, 2, 3, 4, 5]:
            quick_result = matrix.quick_powering(power)

            # NumPy
            matrix_np = matrix_to_numpy(matrix)
            expected_np = np.linalg.matrix_power(matrix_np, power)

            assert compare_with_numpy(quick_result, expected_np, f"Quick power {power}")

    def test_pow_diag_diagonal(self, diagonal_matrix):
        """Тест возведения диагональной матрицы в степень"""
        power = 3
        result = diagonal_matrix.pow_diag(power)

        # NumPy
        matrix_np = matrix_to_numpy(diagonal_matrix)
        expected_np = np.linalg.matrix_power(matrix_np, power)

        assert compare_with_numpy(result, expected_np, "Diagonal power")

    def test_quick_powering_large(self):
        """Тест быстрого возведения в степень для больших матриц"""
        # Создаем простую матрицу 4x4
        matrix = create_matrix_from_list([
            [1, 0, 0, 1],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [1, 0, 0, 1]
        ])

        power = 5
        result = matrix.quick_powering(power)

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_np = np.linalg.matrix_power(matrix_np, power)

        assert compare_with_numpy(result, expected_np, "Large matrix power")

    # Дополнительные тесты для подхода с характеристическим полиномом
    # new approach
    def test_Hamilton_Cayley(self):
        """Тест нового подхода к возведению матриц в степень через использование теоремы Гамильтона-Кэли"""
        powers = [1, 2, 3, 5, 10, 20]
        sizes = [2, 3, 4, 5, 6]

        for power in powers:
            for size in sizes:
                A = create_random_matrix(size, 10)
                # Сравнение с уже протестированным quick_powering
                assert A.Hamilton_Cayley_pow(power) == A.quick_powering(power)


# =================== ТЕСТЫ СПЕЦИАЛЬНЫХ МАТРИЦ ===================

class TestSpecialMatrices:
    """Тесты специальных свойств матриц"""

    def test_is_identity(self, identity_matrix_3x3):
        """Тест проверки на единичную матрицу"""
        assert identity_matrix_3x3.is_identity()
        assert identity_matrix_3x3.is_one_mat

        # NumPy
        identity_np = np.eye(3)
        assert np.allclose(identity_np, np.eye(3))

    def test_is_zero_matrix(self, zero_matrix_3x3):
        """Тест проверки на нулевую матрицу"""
        assert zero_matrix_3x3.is_zero_matrix()
        assert zero_matrix_3x3.is_zero_mat

        # NumPy
        zero_np = np.zeros((3, 3))
        assert np.allclose(zero_np, 0)

    def test_is_diag(self, diagonal_matrix):
        """Тест проверки на диагональность"""
        assert diagonal_matrix.is_diag()

        # NumPy
        matrix_np = matrix_to_numpy(diagonal_matrix)
        # Проверяем, что вне диагонали нули
        assert np.allclose(matrix_np - np.diag(np.diag(matrix_np)), 0)


# =================== ТЕСТЫ СОБСТВЕННЫХ ЗНАЧЕНИЙ И ДИАГОНАЛИЗАЦИИ ===================

class TestEigenvaluesAndDiagonalization:
    """Тесты для методов собственных значений и диагонализации"""

    def test_get_eigen_2x2_diagonal(self):
        """Тест собственных значений для диагональной матрицы 2x2"""
        # Диагональная матрица: собственные значения = диагональные элементы
        matrix = create_matrix_from_list([[3, 0], [0, 5]])
        result = matrix.get_eigen()

        assert result is not None
        eigenvalues, eigenvectors = result

        # Проверяем собственные значения
        lambda1, lambda2 = eigenvalues

        expected_value1 = create_rational_from_int(5)
        expected_value2 = create_rational_from_int(3)

        # Проверяем, что eigenvalues содержат оба значения (порядок может быть любой)
        assert (lambda1 == expected_value1 and lambda2 == expected_value2) or \
               (lambda1 == expected_value2 and lambda2 == expected_value1)

        # Проверяем собственные векторы (должны быть [1,0] и [0,1])
        v1, v2 = eigenvectors
        # Для собственного значения 3: вектор [1, 0]
        assert v1[0] == create_rational_from_int(1) or v2[0] == create_rational_from_int(1)
        assert v1[1] == create_rational_from_int(0) or v2[1] == create_rational_from_int(0)

    def test_get_eigen_2x2_symmetric(self):
        """Тест собственных значений для симметричной матрицы 2x2"""
        # Матрица [[4, 1], [1, 3]] имеет собственные значения (5±√5)/2
        # Но поскольку √5 иррационально, метод должен вернуть None
        matrix = create_matrix_from_list([[4, 1], [1, 3]])
        result = matrix.get_eigen()

        # Иррациональные собственные значения -> должен вернуть None
        assert result is None

    def test_get_eigen_2x2_rational_eigenvalues(self):
        """Тест матрицы 2x2 с рациональными собственными значениями"""
        # Матрица [[5, 2], [2, 2]] имеет собственные значения 6 и 1
        # Характеристическое уравнение: lmbd^2 - 7lmbd + 6 = 0
        # Дискриминант: 49 - 24 = 25 (полный квадрат)
        matrix = create_matrix_from_list([[5, 2], [2, 2]])
        result = matrix.get_eigen()

        assert result is not None
        eigenvalues, eigenvectors = result

        # Собственные значения: 6 и 1
        lambda1, lambda2 = eigenvalues

        expected_value1 = create_rational_from_int(6)
        expected_value2 = create_rational_from_int(1)

        # Проверяем, что eigenvalues содержат оба значения (порядок может быть любой)
        assert (lambda1 == expected_value1 and lambda2 == expected_value2) or \
               (lambda1 == expected_value2 and lambda2 == expected_value1)

        # Проверяем, что A*v = lmbd*v
        for i in range(2):
            lambda_val = eigenvalues[i]
            v = eigenvectors[i]

            # Вычисляем A*v
            Av0 = matrix.arr[0][0] * v[0] + matrix.arr[0][1] * v[1]
            Av1 = matrix.arr[1][0] * v[0] + matrix.arr[1][1] * v[1]

            # Вычисляем lmbd*v
            lambda_v0 = lambda_val * v[0]
            lambda_v1 = lambda_val * v[1]

            # Должны быть равны
            assert Av0 == lambda_v0
            assert Av1 == lambda_v1

    def test_diagonalize_2x2_success(self):
        """Тест успешной диагонализации матрицы 2x2"""
        # Матрица, которая диагонализуема над Q
        matrix = create_matrix_from_list([[4, 1], [2, 3]])
        result = matrix.diagonalize_2x2()

        assert result is not None
        P, D = result

        # Проверяем, что D диагональная
        assert D.is_diag()

        # Проверяем, что P обратима
        P_inv = P.Gauss_inv()

        # Проверяем: A = P * D * P^(-1)
        reconstructed = P.naive_mul(D).naive_mul(P_inv)
        assert matrices_equal(reconstructed, matrix)

        # Проверяем, что столбцы P - собственные векторы
        # (для диагональной матрицы в правильном порядке)

    def test_diagonalize_2x2_failure_irrational(self):
        """Тест неудачной диагонализации из-за иррациональности"""
        # Матрица [[0, 2], [1, 0]] имеет собственные значения ±√2
        matrix = create_matrix_from_list([[0, 2], [1, 0]])
        result = matrix.diagonalize_2x2()

        # Должна вернуть None (иррациональность)
        assert result is None

    def test_diagonalize_2x2_failure_non_diagonalizable(self):
        """Тест недиагонализуемой матрицы (жорданова клетка)"""
        # Матрица [[1, 1], [0, 1]] не диагонализуема
        matrix = create_matrix_from_list([[1, 1], [0, 1]])
        result = matrix.diagonalize_2x2()

        # Должна вернуть None (иррациональность)
        assert result is None

    def test_diag_and_pow_2x2_success(self):
        """Тест возведения в степень через диагонализацию"""
        matrix = create_matrix_from_list([[4, 1], [2, 3]])
        power = 5

        # Через диагонализацию
        diag_pow_result = matrix.diag_and_pow_2x2(power)

        # Через быстрое возведение (для проверки)
        quick_pow_result = matrix.quick_powering(power)

        # Должны давать одинаковый результат
        assert matrices_equal(diag_pow_result, quick_pow_result)

    def test_diag_and_pow_2x2_fallback(self):
        """Тест отката на бинарное возведение при неудачной диагонализации"""
        # Матрица с иррациональными собственными значениями
        matrix = create_matrix_from_list([[0, 2], [1, 0]])
        power = 3

        # Должен использовать fallback на quick_powering
        result = matrix.diag_and_pow_2x2(power)
        expected = matrix.quick_powering(power)

        assert matrices_equal(result, expected)

    def test_analytical_2x2_pow_success(self):
        """Тест возведения в степень по аналитической формуле"""
        # Матрица с рациональными собственными значениями
        matrix = create_matrix_from_list([[4, 1], [2, 3]])

        for power in [0, 1, 2, 3, 4, 5]:
            analytical_result = matrix.analytical_2x2_pow(power)
            quick_result = matrix.quick_powering(power)
            assert matrices_equal(analytical_result, quick_result)

    def test_analytical_2x2_pow_negative(self):
        """Тест отрицательных степеней по аналитической формуле"""
        matrix = create_matrix_from_list([[4, 1], [2, 3]])

        # Проверяем, что A^(-n) = (A^(-1))^n
        n = 3
        analytical_neg = matrix.analytical_2x2_pow(-n)
        inv = matrix.Gauss_inv()
        inv_pow = inv.analytical_2x2_pow(n)

        assert matrices_equal(analytical_neg, inv_pow)

    def test_analytical_2x2_pow_fallback(self):
        """Тест отката аналитической формулы при иррациональных собственных значениях"""
        # Матрица [[0, 2], [1, 0]] имеет ±√2 (иррационально)
        matrix = create_matrix_from_list([[0, 2], [1, 0]])
        power = 4

        # Должен использовать fallback на quick_powering
        result = matrix.analytical_2x2_pow(power)
        expected = matrix.quick_powering(power)

        assert matrices_equal(result, expected)

    def test_get_sqrt_rational(self):
        """Тест извлечения квадратного корня из рационального числа"""
        matrix = create_matrix_from_list([[1, 0], [0, 1]])  # Любая матрица для вызова метода

        # Полные квадраты
        test_cases = [
            (4, 2),  # √4 = 2
            (9, 3),  # √9 = 3
            (16, 4), # √16 = 4
            (25, 5), # √25 = 5
            (1, 1),  # √1 = 1
        ]

        for value, expected_sqrt in test_cases:
            rational = create_rational_from_int(value)
            sqrt_result = matrix._get_sqrt(rational)


            assert sqrt_result is not None
            sqrt_int = int(str(sqrt_result.numerator))
            assert sqrt_int == expected_sqrt

    def test_get_sqrt_irrational(self):
        """Тест извлечения корня из неполного квадрата"""
        matrix = create_matrix_from_list([[1, 0], [0, 1]])

        # Неполные квадраты
        test_cases = [2, 3, 5, 6, 7, 8, 10]

        for value in test_cases:
            rational = create_rational_from_int(value)
            sqrt_result = matrix._get_sqrt(rational)

            # Должен вернуть None для иррациональных корней
            assert sqrt_result is None

    def test_eigenvalue_properties(self):
        """Тест свойств собственных значений"""
        # Для диагонализуемой матрицы 2x2
        matrix = create_matrix_from_list([[4, 1], [2, 3]])
        result = matrix.get_eigen()

        if result is not None:
            eigenvalues, _ = result
            lambda1, lambda2 = eigenvalues

            # След = сумма собственных значений
            trace = matrix.trace()
            assert lambda1 + lambda2 == trace

            # Определитель = произведение собственных значений
            det = matrix.det_Gauss()
            assert lambda1 * lambda2 == det

    def test_diagonalization_power_property(self):
        """Тест свойства: если A = P*D*P^(-1), то A^k = P*D^k*P^(-1)"""
        matrix = create_matrix_from_list([[5, 2], [2, 2]])
        result = matrix.diagonalize_2x2()

        if result is not None:
            P, D = result
            P_inv = P.Gauss_inv()

            for k in [2, 3, 4]:
                # A^k через диагонализацию
                D_pow = D.pow_diag(k)
                A_pow_diag = P.naive_mul(D_pow).naive_mul(P_inv)

                # A^k через быстрое возведение
                A_pow_quick = matrix.quick_powering(k)

                assert matrices_equal(A_pow_diag, A_pow_quick)


# =================== ТЕСТЫ ДЛЯ БОЛЬШИХ МАТРИЦ (СОБСТВЕННЫЕ ЗНАЧЕНИЯ) ===================

class TestLargeMatrixEigen:
    """Тесты собственных значений для больших специальных матриц"""

    def test_diagonal_large_eigenvalues(self):
        """Тест собственных значений для больших диагональных матриц"""
        size = 5
        arr = [[0] * size for _ in range(size)]
        for i in range(size):
            arr[i][i] = i + 2  # Значения 2, 3, 4, 5, 6

        matrix = create_matrix_from_list(arr)

        # Для диагональных матриц собственные значения = диагональные элементы
        # Но метод get_eigen работает только для 2x2

        # Проверяем через свойства:
        # След = сумма собственных значений
        trace = matrix.trace()
        expected_trace = sum(range(2, size + 2))
        assert trace == create_rational_from_int(expected_trace)

        # Определитель = произведение собственных значений
        det = matrix.det_Gauss()
        expected_det = 1
        for i in range(2, size + 2):
            expected_det *= i
        assert det == create_rational_from_int(expected_det)

    def test_scalar_matrix_eigenvalues(self):
        """Тест собственных значений для скалярной матрицы"""
        # Матрица вида c*E, все собственные значения = c
        c = 3
        matrix = create_matrix_from_list([[c, 0, 0], [0, c, 0], [0, 0, c]])

        # Для скалярной матрицы любой вектор - собственный
        trace = matrix.trace()
        assert trace == create_rational_from_int(3 * c)

        det = matrix.det_Gauss()
        assert det == create_rational_from_int(c ** 3)


# =================== ТЕСТЫ СРАВНЕНИЯ МЕТОДОВ ВОЗВЕДЕНИЯ В СТЕПЕНЬ ===================

class TestPowerMethodComparison:
    """Сравнение разных методов возведения в степень"""

    def test_all_power_methods_2x2(self):
        """Сравнение всех методов возведения в степень для матрицы 2x2"""
        matrix = create_matrix_from_list([[4, 1], [2, 3]])
        power = 5

        # Все методы должны давать одинаковый результат
        naive_result = matrix.naive_pow(power)
        quick_result = matrix.quick_powering(power)
        analytical_result = matrix.analytical_2x2_pow(power)
        diag_result = matrix.diag_and_pow_2x2(power)

        # Проверяем попарно
        assert matrices_equal(naive_result, quick_result)
        assert matrices_equal(quick_result, analytical_result)
        assert matrices_equal(analytical_result, diag_result)

    def test_power_methods_negative_power(self):
        """Сравнение методов для отрицательных степеней"""
        matrix = create_matrix_from_list([[4, 1], [2, 3]])
        power = -3

        # Проверяем, что A^(-n) = (A^(-1))^n
        naive_neg = matrix.naive_pow(power)
        quick_neg = matrix.quick_powering(power)

        inv = matrix.Gauss_inv()
        inv_pow_3 = inv.quick_powering(3)

        assert matrices_equal(naive_neg, quick_neg)
        assert matrices_equal(quick_neg, inv_pow_3)


# =================== ТЕСТЫ МАТРИЧНЫХ ПОЛИНОМОВ ===================

class TestMatrixPolynomial:
    """Тесты матричных полиномов"""

    def test_polynomial_creation(self):
        """Тест создания полинома"""
        coeff1 = create_matrix_from_list([[1, 0], [0, 1]])
        coeff2 = create_matrix_from_list([[2, 0], [0, 2]])
        coeff3 = Matrix.zero(2)

        poly = MatrixPolynomial(2, [coeff1, coeff2, coeff3])
        assert poly.degree == 2
        assert poly.size == 2

    def test_polynomial_evaluation(self):
        """Тест вычисления полинома"""
        # P(X) = A2*X^2 + A1*X + A0
        A2 = create_matrix_from_list([[1, 0], [0, 1]])
        A1 = create_matrix_from_list([[2, 0], [0, 2]])
        A0 = create_matrix_from_list([[3, 0], [0, 3]])

        poly = MatrixPolynomial(2, [A2, A1, A0])

        # Вычисляем для X = E
        X = Matrix.identity(2)
        result = poly(X)

        # P(E) = E*E^2 + 2E*E + 3E = E + 2E + 3E = 6E
        expected = create_matrix_from_list([[6, 0], [0, 6]])
        assert matrices_equal(result, expected)


    def test_polynomial_sparse(self):
        """Тест разреженного полинома - должен использовать sequential метод (66.7% нулей < 80%)"""
        # P(X) = A5*X^5 + A0 (разреженный, но недостаточно для бинарного метода)
        A5 = create_matrix_from_list([[1, 0], [0, 1]])
        A0 = create_matrix_from_list([[1, 0], [0, 1]])

        # Коэффициенты от старшего к младшему
        coefficients = [A5] + [Matrix.zero(2) for _ in range(4)] + [A0]
        poly = MatrixPolynomial(5, coefficients)

        # Для степени 5 порог 80%, у нас 66.7% нулей
        assert poly.degree == 5
        assert len(poly.coefficients) == 6
        assert abs(poly.zero_ratio - 4 / 6) < 0.001  # 4 нуля из 6
        assert poly.use_binary == False  # 66.7% < 80%

        X = create_matrix_from_list([[2, 0], [0, 2]])
        result = poly(X)

        # P(X) = X^5 + E
        # X = 2E, поэтому X^5 = 32E
        # P(X) = 32E + E = 33E
        expected = create_matrix_from_list([[33, 0], [0, 33]])
        assert matrices_equal(result, expected)


    def test_polynomial_dense(self):
        """Тест плотного полинома"""
        # Все коэффициенты ненулевые
        coeffs = []
        for i in range(3):
            val = i + 1
            coeffs.append(create_matrix_from_list([[val, 0], [0, val]]))

        poly = MatrixPolynomial(2, coeffs)

        # Должен выбрать последовательный метод (мало нулей)
        assert not poly.use_binary

        X = Matrix.identity(2)
        result = poly(X)

        # P(X) = 3E*X^2 + 2E*X + 1E
        # При X = E: P(E) = 3E + 2E + E = 6E
        expected = create_matrix_from_list([[6, 0], [0, 6]])
        assert matrices_equal(result, expected)


    def test_guaranteed_sparse_polynomial(self):
        """Гарантированно разреженный полином для бинарного метода"""
        # Используем очень высокую степень 1000, чтобы даже при 50% нулей был бинарный
        # Но сделаем 99% нулей для надежности

        size = 2
        degree = 1000

        # Ненулевые коэффициенты только в начале и конце
        first_coeff = Matrix.identity(size)  # при X^1000
        last_coeff = Matrix.identity(size).multiply_by_const(Rational(Integer(0, 1, [2]), Natural(1, [1])))  # 2E

        # 1000 степень -> 1001 коэффициент
        # 999 нулевых коэффициентов из 1001 = 99.8% нулей
        coefficients = [first_coeff] + [Matrix.zero(size) for _ in range(degree - 1)] + [last_coeff]

        poly = MatrixPolynomial(degree, coefficients)

        # Проверяем
        assert poly.use_binary, (
            f"Полином степени {degree} с {poly.zero_ratio:.1%} нулей должен использовать бинарный метод. "
            f"Порог для степени {degree}: 50%"
        )

        # Простое вычисление с единичной матрицей
        X = Matrix.identity(size)
        result = poly(X)

        # P(E) = E^1000 + 2E = E + 2E = 3E
        expected = Matrix.identity(size).multiply_by_const(Rational(Integer(0, 1, [3]), Natural(1, [1])))

        assert result == expected


    def test_polynomial_sparse_binary_simple(self):
        """Простой тест разреженного полинома с бинарным методом"""
        # P(X) = A30*X^30 + A0  (степень 30 => порог 70%)

        # Создаем простые матрицы 2x2 для теста
        A30 = create_matrix_from_list([[1, 0], [0, 1]])  # E
        A0 = create_matrix_from_list([[2, 0], [0, 2]])  # 2E

        # 30 степень -> 31 коэффициент
        # 29 нулевых коэффициентов из 31 = 93.5% нулей > 70%
        coefficients = [A30] + [Matrix.zero(2) for _ in range(29)] + [A0]

        poly = MatrixPolynomial(30, coefficients)

        # Проверяем параметры
        assert poly.degree == 30
        assert len(poly.coefficients) == 31
        assert poly.zero_ratio >= 0.93  # 29/31 ≈ 0.935
        assert poly.use_binary == True  # 93.5% > 70%

        # Простая проверка вычисления
        X = create_matrix_from_list([[1, 0], [0, 1]])  # X = E

        result = poly(X)
        # P(E) = E^30 + 2E = E + 2E = 3E
        expected = create_matrix_from_list([[3, 0], [0, 3]])

        assert matrices_equal(result, expected)



# =================== ТЕСТЫ БОЛЬШИХ МАТРИЦ С NumPy ===================

class TestLargeMatricesWithNumpy:
    """Тесты для больших матриц с сравнением с NumPy"""

    def test_large_addition_numpy(self):
        """Тест сложения больших матриц с NumPy"""
        size = 8
        A = create_random_matrix(size, max_value=10)
        B = create_random_matrix(size, max_value=10)

        result = A + B

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = A_np + B_np

        assert compare_with_numpy(result, expected_np, "Large addition")

    def test_large_multiplication_numpy(self):
        """Тест умножения больших матриц с NumPy"""
        size = 6
        A = create_random_matrix(size, max_value=5)
        B = create_random_matrix(size, max_value=5)

        result = A.naive_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        expected_np = np.dot(A_np, B_np)

        assert compare_with_numpy(result, expected_np, "Large multiplication")

    def test_large_strassen_vs_naive_numpy(self):
        """Сравнение Штрассена, наивного и NumPy"""
        size = 8  # Степень двойки
        A = create_random_matrix(size, max_value=5)
        B = create_random_matrix(size, max_value=5)

        naive_result = A.naive_mul(B)
        strassen_result = A.Strassen_mul(B)

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        numpy_result = np.dot(A_np, B_np)

        # Все три должны быть близки (равны, так как вычисления точные)
        assert compare_with_numpy(naive_result, numpy_result, "Naive vs NumPy")
        assert compare_with_numpy(strassen_result, numpy_result, "Strassen vs NumPy")
        assert matrices_equal(naive_result, strassen_result)

    def test_large_determinant_numpy(self):
        """Тест определителя больших матриц с NumPy"""
        size = 6
        matrix = create_random_matrix(size, max_value=3)

        det = matrix.det_Gauss()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        np_det = np.linalg.det(matrix_np)

        det_float = float(str(det.numerator)) / float(str(det.denominator))

        tolerance = 1e-10
        assert abs(det_float - np_det) < tolerance, \
            f"Large det mismatch: {det_float} vs {np_det}"

    def test_large_inverse_numpy(self):
        """Тест обращения больших матриц с NumPy"""
        # Создаем хорошо обусловленную матрицу (диагонально доминирующую)
        size = 5
        arr = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j:
                    arr[i][j] = size + 1  # Диагональное преобладание
                else:
                    arr[i][j] = 1
        matrix = create_matrix_from_list(arr)

        inv = matrix.Gauss_inv()

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_inv_np = np.linalg.inv(matrix_np)

        assert compare_with_numpy(inv, expected_inv_np, "Large inverse")

        # Проверяем, что A * A^(-1) ≈ E
        product = matrix.naive_mul(inv)
        product_np = matrix_to_numpy(product)
        identity_np = np.eye(size)

        assert np.allclose(product_np, identity_np, rtol=1e-8)

    def test_large_power_numpy(self):
        """Тест возведения в степень больших матриц с NumPy"""
        size = 4
        # Создаем простую матрицу для стабильного возведения в степень
        arr = [[0] * size for _ in range(size)]
        for i in range(size):
            arr[i][i] = 2 if i % 2 == 0 else 1
            if i < size - 1:
                arr[i][i + 1] = 1
        matrix = create_matrix_from_list(arr)

        power = 4
        result = matrix.quick_powering(power)

        # NumPy
        matrix_np = matrix_to_numpy(matrix)
        expected_np = np.linalg.matrix_power(matrix_np, power)

        assert compare_with_numpy(result, expected_np, "Large matrix power")


# =================== ТЕСТЫ СЛОЖНЫХ ОПЕРАЦИЙ С NumPy ===================

class TestComplexOperationsWithNumpy:
    """Тесты сложных операций с сравнением с NumPy"""

    def test_chain_operations_numpy(self):
        """Тест цепочки операций: (A + B) * C^T"""
        size = 4
        A = create_random_matrix(size, max_value=5)
        B = create_random_matrix(size, max_value=5)
        C = create_random_matrix(size, max_value=5)

        # Наша реализация
        result = (A + B).naive_mul(C.transpose())

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        C_np = matrix_to_numpy(C)

        expected_np = np.dot(A_np + B_np, C_np.T)

        assert compare_with_numpy(result, expected_np, "Chain operations")

    def test_determinant_properties_numpy(self):
        """Тест свойств определителя с NumPy"""
        size = 3
        A = create_random_matrix(size, max_value=3)
        B = create_random_matrix(size, max_value=3)

        # det(AB) = det(A)det(B)
        det_A = A.det_Gauss()
        det_B = B.det_Gauss()
        det_AB = A.naive_mul(B).det_Gauss()

        # NumPy
        A_np = matrix_to_numpy(A)
        B_np = matrix_to_numpy(B)
        np_det_A = np.linalg.det(A_np)
        np_det_B = np.linalg.det(B_np)
        np_det_AB = np.linalg.det(np.dot(A_np, B_np))

        # Конвертируем наши определители
        det_A_float = float(str(det_A.numerator)) / float(str(det_A.denominator))
        det_B_float = float(str(det_B.numerator)) / float(str(det_B.denominator))
        det_AB_float = float(str(det_AB.numerator)) / float(str(det_AB.denominator))

        # Проверяем свойство
        assert abs(det_AB_float - (det_A_float * det_B_float)) < 1e-10
        assert abs(np_det_AB - (np_det_A * np_det_B)) < 1e-10
        assert abs(det_AB_float - np_det_AB) < 1e-10

    def test_inverse_properties_numpy(self):
        """Тест свойств обратной матрицы с NumPy"""
        size = 3
        A = create_random_matrix(size, max_value=3)

        # Пропускаем вырожденные
        if A.det_Gauss() == create_rational_from_int(0):
            pytest.skip("Matrix is singular")

        inv = A.Gauss_inv()

        # (A^(-1))^(-1) = A
        inv_of_inv = inv.Gauss_inv()

        # NumPy
        A_np = matrix_to_numpy(A)
        inv_np = np.linalg.inv(A_np)
        inv_of_inv_np = np.linalg.inv(inv_np)

        assert compare_with_numpy(inv_of_inv, inv_of_inv_np, "Enverse of inverse")
        assert matrices_equal(inv_of_inv, A)


# =================== СТРЕСС И ПРОИЗВОДИТЕЛЬНОСТЬ ===================

class TestStressTests:
    """Стресс-тесты для проверки стабильности"""

    @pytest.mark.slow
    def test_multiplication_consistency_large(self):
        """Проверка согласованности умножения для больших матриц"""
        sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        for size in sizes:
            A = create_random_matrix(size, max_value=3)
            B = create_random_matrix(size, max_value=3)

            # Все методы должны давать одинаковый результат
            naive_result = A.naive_mul(B)
            strassen_result = A.Strassen_mul(B)

            # NumPy для проверки
            A_np = matrix_to_numpy(A)
            B_np = matrix_to_numpy(B)
            numpy_result = np.dot(A_np, B_np)

            # Проверяем
            assert compare_with_numpy(naive_result, numpy_result,
                                      f"Naive vs NumPy (size {size})")
            assert compare_with_numpy(strassen_result, numpy_result,
                                      f"Strassen vs NumPy (size {size})")
            assert matrices_equal(naive_result, strassen_result)

    @pytest.mark.slow
    def test_determinant_consistency_large(self):
        """Проверка определителя для случайных больших матриц"""
        sizes = [2, 3, 4, 5, 6]

        for size in sizes:
            matrix = create_random_matrix(size, max_value=3)

            det_gauss = matrix.det_Gauss()

            # NumPy
            matrix_np = matrix_to_numpy(matrix)
            np_det = np.linalg.det(matrix_np)

            det_float = float(str(det_gauss.numerator)) / float(str(det_gauss.denominator))

            tolerance = 1e-6 if size > 4 else 1e-10
            assert abs(det_float - np_det) < tolerance, \
                f"Size {size}: {det_float} vs {np_det}"


# =================== ТЕСТЫ С НЕЦЕЛЫМИ ЗНАЧЕНИЯМИ ===================

class TestFractionalValues:
    """Тесты с дробными значениями"""

    def create_rational_fraction(self, num, den):
        """Создать дробь num/den"""
        num_int = Integer(0 if num >= 0 else 1, len(str(abs(num))) - 1,
                          [int(d) for d in str(abs(num))])
        den_nat = Natural(len(str(den)) - 1, [int(d) for d in str(den)])
        return Rational(num_int, den_nat)

    def test_fractional_matrix(self):
        """Тест матрицы с дробными значениями"""
        # Матрица с дробями 1/2, 1/3 и т.д.
        size = 2
        arr = [
            [self.create_rational_fraction(1, 2), self.create_rational_fraction(1, 3)],
            [self.create_rational_fraction(2, 3), self.create_rational_fraction(3, 4)]
        ]
        matrix = Matrix(size, arr)

        # Проверяем след
        trace = matrix.trace()
        # trace = 1/2 + 3/4 = 5/4
        expected_trace = self.create_rational_fraction(5, 4)
        assert trace == expected_trace

        # Проверяем определитель
        det = matrix.det_Gauss()
        # det = 1/2 * 3/4 - 1/3 * 2/3 = 3/8 - 2/9 = (27 - 16)/72 = 11/72
        expected_det = self.create_rational_fraction(11, 72)
        assert det == expected_det

        # 1. Проверяем СЛОЖЕНИЕ с самой собой
        sum_result = matrix + matrix
        # A + A = 2A
        expected_sum_arr = [
            [self.create_rational_fraction(1, 1), self.create_rational_fraction(2, 3)],  # 1/2+1/2=1, 1/3+1/3=2/3
            [self.create_rational_fraction(4, 3), self.create_rational_fraction(3, 2)]  # 2/3+2/3=4/3, 3/4+3/4=3/2
        ]
        expected_sum = Matrix(size, expected_sum_arr)

        for i in range(size):
            for j in range(size):
                assert sum_result.arr[i][j] == expected_sum_arr[i][j], \
                    f"Сложение: элемент [{i},{j}]: {sum_result.arr[i][j]} != {expected_sum_arr[i][j]}"

        # 2. Проверяем ВЫЧИТАНИЕ
        # Создаем другую дробную матрицу
        arr2 = [
            [self.create_rational_fraction(1, 4), self.create_rational_fraction(1, 6)],
            [self.create_rational_fraction(1, 3), self.create_rational_fraction(1, 2)]
        ]
        matrix2 = Matrix(size, arr2)

        sub_result = matrix - matrix2
        # A - B
        expected_sub_arr = [
            [
                self.create_rational_fraction(1, 2) - self.create_rational_fraction(1, 4),  # 1/2 - 1/4 = 1/4
                self.create_rational_fraction(1, 3) - self.create_rational_fraction(1, 6)  # 1/3 - 1/6 = 1/6
            ],
            [
                self.create_rational_fraction(2, 3) - self.create_rational_fraction(1, 3),  # 2/3 - 1/3 = 1/3
                self.create_rational_fraction(3, 4) - self.create_rational_fraction(1, 2)  # 3/4 - 1/2 = 1/4
            ]
        ]

        for i in range(size):
            for j in range(size):
                assert sub_result.arr[i][j] == expected_sub_arr[i][j], \
                    f"Вычитание: элемент [{i},{j}]: {sub_result.arr[i][j]} != {expected_sub_arr[i][j]}"

        # 3. Проверяем УМНОЖЕНИЕ на константу
        const = self.create_rational_fraction(2, 3)  # константа 2/3
        mul_const_result = matrix.multiply_by_const(const)

        expected_mul_const_arr = [
            [
                self.create_rational_fraction(1, 2) * const,  # 1/2 * 2/3 = 1/3
                self.create_rational_fraction(1, 3) * const  # 1/3 * 2/3 = 2/9
            ],
            [
                self.create_rational_fraction(2, 3) * const,  # 2/3 * 2/3 = 4/9
                self.create_rational_fraction(3, 4) * const  # 3/4 * 2/3 = 1/2
            ]
        ]

        for i in range(size):
            for j in range(size):
                assert mul_const_result.arr[i][j] == expected_mul_const_arr[i][j], \
                    f"Умножение на константу: элемент [{i},{j}]: {mul_const_result.arr[i][j]} != {expected_mul_const_arr[i][j]}"

        # 4. Проверяем ТРАНСПОНИРОВАНИЕ
        transpose_result = matrix.transpose()
        expected_transpose_arr = [
            [self.create_rational_fraction(1, 2), self.create_rational_fraction(2, 3)],
            [self.create_rational_fraction(1, 3), self.create_rational_fraction(3, 4)]
        ]

        for i in range(size):
            for j in range(size):
                assert transpose_result.arr[i][j] == expected_transpose_arr[i][j], \
                    f"Транспонирование: элемент [{i},{j}]: {transpose_result.arr[i][j]} != {expected_transpose_arr[i][j]}"

        # 5. Проверяем ПРОИЗВЕДЕНИЕ A * A
        prod = matrix.naive_mul(matrix)

        # Вычисляем A^2 вручную:
        # [1/2  1/3]   *   [1/2  1/3]   =   [a11  a12]
        # [2/3  3/4]       [2/3  3/4]       [a21  a22]
        #
        # a11 = (1/2)*(1/2) + (1/3)*(2/3) = 1/4 + 2/9 = (9 + 8)/36 = 17/36
        # a12 = (1/2)*(1/3) + (1/3)*(3/4) = 1/6 + 1/4 = (2 + 3)/12 = 5/12
        # a21 = (2/3)*(1/2) + (3/4)*(2/3) = 1/3 + 1/2 = (2 + 3)/6 = 5/6
        # a22 = (2/3)*(1/3) + (3/4)*(3/4) = 2/9 + 9/16 = (32 + 81)/144 = 113/144

        expected_prod_arr = [
            [self.create_rational_fraction(17, 36), self.create_rational_fraction(5, 12)],
            [self.create_rational_fraction(5, 6), self.create_rational_fraction(113, 144)]
        ]

        for i in range(size):
            for j in range(size):
                assert prod.arr[i][j] == expected_prod_arr[i][j], \
                    f"Умножение A*A: элемент [{i},{j}]: {prod.arr[i][j]} != {expected_prod_arr[i][j]}"

        # 6. Проверяем умножение A * B
        prod_ab = matrix.naive_mul(matrix2)

        # Вычисляем A * B вручную:
        # [1/2  1/3]   *   [1/4  1/6]   =   [c11  c12]
        # [2/3  3/4]       [1/3  1/2]       [c21  c22]
        #
        # c11 = (1/2)*(1/4) + (1/3)*(1/3) = 1/8 + 1/9 = (9 + 8)/72 = 17/72
        # c12 = (1/2)*(1/6) + (1/3)*(1/2) = 1/12 + 1/6 = (1 + 2)/12 = 3/12 = 1/4
        # c21 = (2/3)*(1/4) + (3/4)*(1/3) = 1/6 + 1/4 = (2 + 3)/12 = 5/12
        # c22 = (2/3)*(1/6) + (3/4)*(1/2) = 1/9 + 3/8 = (8 + 27)/72 = 35/72

        expected_prod_ab_arr = [
            [self.create_rational_fraction(17, 72), self.create_rational_fraction(1, 4)],
            [self.create_rational_fraction(5, 12), self.create_rational_fraction(35, 72)]
        ]

        for i in range(size):
            for j in range(size):
                assert prod_ab.arr[i][j] == expected_prod_ab_arr[i][j], \
                    f"Умножение A*B: элемент [{i},{j}]: {prod_ab.arr[i][j]} != {expected_prod_ab_arr[i][j]}"

        # 7. Проверяем с NumPy для всех операций
        matrix_np = np.array([[1 / 2, 1 / 3], [2 / 3, 3 / 4]], dtype=float)
        matrix2_np = np.array([[1 / 4, 1 / 6], [1 / 3, 1 / 2]], dtype=float)

        # Сложение
        sum_np = matrix_np + matrix_np
        sum_result_np = matrix_to_numpy(sum_result)
        assert np.allclose(sum_result_np, sum_np, rtol=1e-10), "Ошибка в сложении (NumPy)"

        # Вычитание
        sub_np = matrix_np - matrix2_np
        sub_result_np = matrix_to_numpy(sub_result)
        assert np.allclose(sub_result_np, sub_np, rtol=1e-10), "Ошибка в вычитании (NumPy)"

        # Умножение на константу
        const_float = 2 / 3
        mul_const_np = matrix_np * const_float
        mul_const_result_np = matrix_to_numpy(mul_const_result)
        assert np.allclose(mul_const_result_np, mul_const_np, rtol=1e-10), "Ошибка в умножении на константу (NumPy)"

        # Транспонирование
        transpose_np = matrix_np.T
        transpose_result_np = matrix_to_numpy(transpose_result)
        assert np.allclose(transpose_result_np, transpose_np, rtol=1e-10), "Ошибка в транспонировании (NumPy)"

        # Умножение матриц
        prod_np = np.dot(matrix_np, matrix_np)
        prod_result_np = matrix_to_numpy(prod)
        assert np.allclose(prod_result_np, prod_np, rtol=1e-10), "Ошибка в умножении A*A (NumPy)"

        prod_ab_np = np.dot(matrix_np, matrix2_np)
        prod_ab_result_np = matrix_to_numpy(prod_ab)
        assert np.allclose(prod_ab_result_np, prod_ab_np, rtol=1e-10), "Ошибка в умножении A*B (NumPy)"


if __name__ == "__main__":

    # Запуск всех тестов
    result = pytest.main([__file__, "-v"])