"""
Промежуточная версия класса работы с невырожденными квадратными матрицами над Q
и класса полинома с коэффициентами из этих матриц, содержащая несколько подходов
к одной и той же задаче, с целью их исследования и выбора наиболее оптимального алгоритма.
"""


from N.Natural import Natural
from P.Polynomial import Polynomial
from Q.Rational import Rational
from Z.Integer import Integer


# Вынесем константы во избежание дублирования кода и создания излишнего количества объектов
ONE_RATIONAL = Rational(Integer(0, 1, [1]), Natural(1, [1]))
ZERO_RATIONAL = Rational(Integer(0, 1, [0]), Natural(1, [1]))


class Matrix:
    def __init__(self, size: int, arr: list[list[Rational]]):
        # Проверка, что матрица квадратная
        n = len(arr)
        for i in range(n):
            if len(arr[i]) != n:
                raise ValueError("Матрица должна быть квадратной")

        if size != n:
            raise ValueError(f"Указанный размер {size} не соответствует реальному {n}")

        self.size = size    # размерность квадратной матрицы, int
        self.arr = arr      # массив строк матрицы, строка - массив рациональных чисел

        # При создании вычисляем статистику про матрицу, чтобы использовать ее при
        # умножении, сложении, возведении в степень (дает выигрыш во времени в этих случаях + имеет ранний выход из циклов)
        self.is_zero_mat = self.is_zero_matrix()
        self.is_one_mat = self.is_identity()

    __slots__ = ('size', 'arr', 'is_zero_mat', 'is_one_mat')


    def __add__(self, other):
        """
        Сложение двух матриц (одного размера)
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")

        if self.is_zero_mat:    # Небольшая оптимизация: проверки на нулевые матрицы, O(1) так как кэшировано
            return other
        if other.is_zero_mat:
            return self

        n = self.size
        res = [
            [self.arr[i][j] + other.arr[i][j] for j in range(n)]  # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)


    def __sub__(self, other):
        """
        Вычитание двух матриц (одного размера)
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")

        if other.is_zero_mat:   # A - 0 = A
            return self

        n = self.size
        res = [
            [self.arr[i][j] - other.arr[i][j] for j in range(n)]  # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)


    # future __mul__
    def multiply_by_const(self, const: Rational):
        """
        Умножение матрицы на константу
        """
        n = self.size
        if not isinstance(const, Rational):
            raise TypeError(f"Константа должна быть Rational, получено {type(const)}")

        if self.is_zero_mat:   # если матрица нулевая, то ее не умножаем на константу, результат она же
            return self

        if const == ZERO_RATIONAL:
            return Matrix.zero(n)

        if const == ONE_RATIONAL:
            return self

        res = [
            [self.arr[i][j] * const for j in range(n)]   # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)


    def trace(self):
        """
        Вычисение следа матрицы
        O(n)
        """
        n = self.size
        res = ZERO_RATIONAL
        for i in range(n):
            res += self.arr[i][i]
        return res


    def transpose(self):
        """
        Возвращает новую транспонированную матрицу
        Строки и столбцы меняются местами
        O(n^2)
        """
        n = self.size
        res = [
            [self.arr[j][i] for j in range(n)]  # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)

# =============================================== MATMUL ====================================================

    # 3 Multiplications (future __matmul__ = @)
    def naive_mul(self, other):
        """
        Умножение двух матриц по определению -> O(n^3)

        Пометки про производительность (подробнее в research.py / benchmarking.py):
        * смог посчитать умножение матриц 40x40, но долго
        * посчитал для матрицы 100х100 но минут 5-7 примерно, а метод Штрассена не смог посчитать за 15 минут...
        * относительно быстро для n <= 10, метод Штрассена ощутимо дольше считает
        => Используем
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")
        n = self.size
        
        # Небольшие оптимизации
        if self.is_one_mat: return other  # умножение на E
        if other.is_one_mat: return self

        if self.is_zero_mat: return Matrix.zero(n) # умножение на 0-matrix
        if other.is_zero_mat: return Matrix.zero(n)


        res = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (self.arr[i][k] == ZERO_RATIONAL) or (other.arr[k][j] == ZERO_RATIONAL):  # оптимизация: проверка на ноль вместо алгоритмов умножения и сложения дробей
                        continue
                    res[i][j] += self.arr[i][k] * other.arr[k][j]

        return Matrix(n, res)


    def blocked_mul(self, other, block_size=None):
        """
        Блочное умножение матриц

        * В теории должен давать виыгрыш на больших матрицах
        благодаря локальности памяти и лучшей кэш-эффективности,
        так как обращений к памяти меньше, тк достается блоками
        * На практике работает либо так же, как наивное умножение, либо хуже,
        так как затраты на создание range, работу min,
        что не перекрывет возвможный выигрыш по работе с памятью
        => Не используем
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")

        n = self.size

        # Небольшие оптимизации
        if self.is_one_mat: return other  # умножение на E
        if other.is_one_mat: return self

        if self.is_zero_mat: return Matrix.zero(n)  # умножение на 0-matrix
        if other.is_zero_mat: return Matrix.zero(n)


        if block_size is None:
            # Автоподбор
            if self.size <= 16:
                block_size = 4
            elif self.size <= 32:
                block_size = 8
            elif self.size <= 64:
                block_size = 16
            else:
                block_size = 32


        res = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
        for ii in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for jj in range(0, n, block_size):
                    # дошли до блока
                    for i in range(ii, min(ii + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            a = self.arr[i][k]
                            if a == ZERO_RATIONAL:
                                continue
                            for j in range(jj, min(jj + block_size, n)):
                                b = other.arr[k][j]
                                if b == ZERO_RATIONAL:
                                    continue
                                res[i][j] = res[i][j] + a * b

        return Matrix(n, res)


    def Strassen_mul(self, other):
        """
        Умножение двух матриц по алгоритму Штрассена

        Алгоритм:
        1. Проверяем размеры
        2. Находим m = ближайшая степень двойки >= n
        3. Дополняем A и B нулями до (m x m), если m != n
        4. Вызываем recursive_strassen_power_of_two(A', B')
        5. Обрезаем результат до (n x n), если m != n
        6. Возвращаем Matrix

        Сложность:
        T(n) = 7 * T(n/2) + O(n^2)  # 7 рекурсивных вызовов вместо 8, как в случае с рекурсивной реализации наивного умножения
        O(n^2) уходит на сложение, вычитание (18 шт), расширение и обрезку матриц
        глубина рекурсии = log2(n)
        T(n) = O(n^log2(7)) = O(n^2.81) - решение рекурентного соотношения, через рассмотрение всей суммы элементов дерева рекурсии
        = сумма геометрической прогресcии.

        НО на практике огромная константа за счет организации вычисений, рекурсии,
        а так же больше затраты на расширение и обрезку матриц размера != степень двойки
        Считает дольше, чем наивное умножение.
        => Не используем
        """

        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")

        n = self.size

        # Ближайшая степень двойки
        m = 1
        while m < n:
            m <<= 1

        # Дополняем матрицы (работаем со списками, чтобы не создавать много объектов Matrix)
        def pad(arr, size):
            zero = Rational(Integer(0, 1, [0]), Natural(1, [1]))
            result = [[zero] * size for _ in range(size)]
            for i in range(len(arr)):
                for j in range(len(arr)):
                    result[i][j] = arr[i][j]
            return result


        if m != n:  # если надо дополнять нулями
            A = pad(self.arr, m)
            B = pad(other.arr, m)

        else:   # не надо заполнять нулями, просто копируем
            A = self.arr
            B = other.arr


        # Основная рекурсивная функция
        def strassen_recursive(A, B, size):
            if size == 1:
                # Базовый случай
                return [[A[0][0] * B[0][0]]]

            half = size // 2

            # Вспомогательные функции для работы с блоками: извлечение блока, сложение и вычитание
            def get_block(mat, r, c):
                return [row[c:c + half] for row in mat[r:r + half]]

            def add_blocks(X, Y):
                return [[X[i][j] + Y[i][j] for j in range(len(X[0]))]
                        for i in range(len(X))]

            def sub_blocks(X, Y):
                return [[X[i][j] - Y[i][j] for j in range(len(X[0]))]
                        for i in range(len(X))]


            # Извлекаем блоки
            A11 = get_block(A, 0, 0)
            A12 = get_block(A, 0, half)
            A21 = get_block(A, half, 0)
            A22 = get_block(A, half, half)

            B11 = get_block(B, 0, 0)
            B12 = get_block(B, 0, half)
            B21 = get_block(B, half, 0)
            B22 = get_block(B, half, half)

            # Вычисляем произведения согласно идее алгоритма(7 шт)
            P1 = strassen_recursive(A11, sub_blocks(B12, B22), half)    # P1 = A11 * (B12 - B22)
            P2 = strassen_recursive(add_blocks(A11, A12), B22, half)    # P2 = (A11 + A12) * B22
            P3 = strassen_recursive(add_blocks(A21, A22), B11, half)    # P3 = (A21 + A22) * B11
            P4 = strassen_recursive(A22, sub_blocks(B21, B11), half)    # P4 = A22 * (B21 - B11)
            P5 = strassen_recursive(add_blocks(A11, A22), add_blocks(B11, B22), half)   # P5 = (A11 + A22) * (B11 + B22)
            P6 = strassen_recursive(sub_blocks(A12, A22), add_blocks(B21, B22), half)   # P6 = (A12 - A22) * (B21 + B22)
            P7 = strassen_recursive(sub_blocks(A11, A21), add_blocks(B11, B12), half)   # P7 = (A11 - A21) * (B11 + B12)

            # Вычисляем блоки результата
            # C11 = P5 + P4 - P2 + P6
            C11 = add_blocks(P5, P4)
            C11 = sub_blocks(C11, P2)
            C11 = add_blocks(C11, P6)

            # C12 = P1 + P2
            C12 = add_blocks(P1, P2)

            # C21 = P3 + P4
            C21 = add_blocks(P3, P4)

            # C22 = P5 + P1 - P3 - P7
            C22 = add_blocks(P5, P1)
            C22 = sub_blocks(C22, P3)
            C22 = sub_blocks(C22, P7)

            # Объединяем блоки
            result = [[None] * size for _ in range(size)]
            for i in range(half):
                for j in range(half):
                    result[i][j] = C11[i][j]
                    result[i][j + half] = C12[i][j]
                    result[i + half][j] = C21[i][j]
                    result[i + half][j + half] = C22[i][j]

            return result

        # Запускаем рекурсию
        result_full = strassen_recursive(A, B, m)

        # Убираем нули, если надо
        if len(result_full) != n:
            result_trimmed = [row[:n] for row in result_full[:n]]
            return Matrix(n, result_trimmed)

        return Matrix(n, result_full)

#================================================= DET =================================================

    # DETERMINANTS 4 ways
    def det_formula(self):
        # Для малых размерностей считаем через формулы, посчитанные вручную -> O(1)
        n = self.size

        if n == 1:
            return self.arr[0][0]

        if n == 2:
            # ad - bc
            a, b = self.arr[0][0], self.arr[0][1]
            c, d = self.arr[1][0], self.arr[1][1]
            return a * d - b * c

        if n == 3:
            # треугольники (правило Сарруса)
            a, b, c = self.arr[0]
            d, e, f = self.arr[1]
            g, h, i = self.arr[2]
            return a * e * i  +  b * f * g  +  c * d * h  -  c * e * g  -  b * d * i  -  a * f * h

        return None


    def det_definition(self):
        """
        Вычисление определителя квадратной матрицы
        по определению через знакопеременную функцию:
        det(A) = Σ_σ sgn(σ) ∏ a_{i,σ(i)}

        Очень плохая сложность O(n^2 * n!)
        Но для n <= 3 константно, а для больших n безумно неэффективно
        => Не используем
        """
        n = self.size
        if n <= 3:
            return self.det_formula()

        # Для n > 3 используем перебор перестановок
        # Генерируем все перестановки рекурсивно
        permutations = []
        used = [False] * n

        def generate_permutations(current):    # O(n * n!), тк n! - всего перестановок, O(n) - на копирование каждой
            if len(current) == n:   # граничный случай
                permutations.append(current[:])    # O(n)
                return

            for i in range(n):    # рекурсивный случай
                if not used[i]:
                    used[i] = True
                    current.append(i)
                    generate_permutations(current)
                    current.pop()
                    used[i] = False

        generate_permutations([])

        det_sum = ZERO_RATIONAL

        # Проход по всем перестановкам (их n! штук)
        for perm in permutations:
            # Вычисляем знак перестановки через инверсии
            inversions = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if perm[i] > perm[j]:
                        inversions += 1

            # sign = +1 если чётное число инверсий, -1 если нечётное
            sign_numerator = Integer(0, 1, [1]) if inversions % 2 == 0 else Integer(1, 1, [1])
            sign_rational = Rational(sign_numerator, Natural(1, [1]))

            product = ONE_RATIONAL

            # Вычисляем произведение a_{i, perm[i]}
            for i in range(n):
                product = product * self.arr[i][perm[i]]

            # Добавляем к сумме
            term = sign_rational * product
            det_sum = det_sum + term

        return det_sum


    def det_Laplace(self):  # оптимален где много нулей. но все равно Гаусс быстрее
        """
        Вычисление определителя квадратной матрицы
        через рекурсивное разложение по строке/столбцу (метод Лапласа)

        Сложность: T(n) = n * T(n - 1) + O(n^2) - рекурсивное соотношение,
        O(n^2) - на проход по всей матрице с целью поиска строк/столбцов, где много нулей.
        * Отсюда сложность O(n!) - для худшего случая, где нет строк/столбцов с нулями.
        * Для лучшего лучая (треугольная/диагональная матрица, где много нулей)
        можно добиться кубической сложности:
        T(n) = T(n - 1) + O(n^2) => T(n) = T(1) + O(1^2) + O(2^2) + ... + O(n^2) ~ n(n+1)(2n+1)/6 = O(n^3))
        но это крайне редкая ситуация, причем по методу Гаусса она все равно будет быстрее в большинстве случаев,
        в противном разница все равно не велика

        Граничный случай рекурсии n = 2, что уменьшает кол-во вычислений, но все равно остается факториал в худшем случае
        => Не используем
        """

        n = self.size
        if n <= 3:
            return self.det_formula()

        # Вспомогательная рекурсивная функция
        def det_recursive(rows, cols):
            """
            Вычисляет определитель подматрицы по спискам индексов,
            чтобы не тратить ресурсы на копирование  (n^2 уходит на это)
            """
            size = len(rows)

            # Базовые случаи
            if size == 1:
                return self.arr[rows[0]][cols[0]]

            if size == 2:
                i1, i2 = rows[0], rows[1]
                j1, j2 = cols[0], cols[1]
                return self.arr[i1][j1] * self.arr[i2][j2] - self.arr[i1][j2] * self.arr[i2][j1]

            # Поиск строки или столбца с максимальным количеством нулей
            # Это уменьшит количество рекурсивных вызовов (оптимимзация)

            # Проверяем строки  O(n^2)
            best_row = None
            max_zeros_in_row = -1

            for idx, row_idx in enumerate(rows):
                zeros_count = 0
                for col_idx in cols:
                    if self.arr[row_idx][col_idx] == ZERO_RATIONAL:
                        zeros_count += 1

                if zeros_count > max_zeros_in_row:
                    max_zeros_in_row = zeros_count
                    best_row = (row_idx, idx)  # (индекс в матрице, позиция в списке rows)

            # Проверяем столбцы  O(n^2)
            best_col = None
            max_zeros_in_col = -1

            for idx, col_idx in enumerate(cols):
                zeros_count = 0
                for row_idx in rows:
                    if self.arr[row_idx][col_idx] == ZERO_RATIONAL:
                        zeros_count += 1

                if zeros_count > max_zeros_in_col:
                    max_zeros_in_col = zeros_count
                    best_col = (col_idx, idx)  # (индекс в матрице, позиция в списке cols)

            # Выбираем что лучше: разложение по строке или по столбцу
            expand_by_row = (max_zeros_in_row >= max_zeros_in_col)

            result = ZERO_RATIONAL

            if expand_by_row:
                # Разложение по строке
                row_idx, row_pos = best_row

                # Новые строки (без текущей)
                new_rows = rows[:row_pos] + rows[row_pos + 1:]

                for col_pos, col_idx in enumerate(cols):
                    element = self.arr[row_idx][col_idx]
                    if element == ZERO_RATIONAL:
                        continue  # Пропускаем нулевые элементы - их вклад нулевой (оптимизация)

                    # Знак: (-1)^(row_pos + col_pos)
                    sign_power = row_pos + col_pos
                    sign_numerator = Integer(0, 1, [1]) if sign_power % 2 == 0 else Integer(1, 1, [1])
                    sign = Rational(sign_numerator, Natural(1, [1]))

                    # Новые столбцы (без текущего)
                    new_cols = cols[:col_pos] + cols[col_pos + 1:]

                    # Рекурсивный вызов для минора
                    minor_det = det_recursive(new_rows, new_cols)

                    # Вклад в определитель
                    term = sign * element * minor_det
                    result = result + term
            else:
                # Разложение по столбцу
                col_idx, col_pos = best_col

                # Новые столбцы (без текущего)
                new_cols = cols[:col_pos] + cols[col_pos + 1:]

                for row_pos, row_idx in enumerate(rows):
                    element = self.arr[row_idx][col_idx]
                    if element == ZERO_RATIONAL:
                        continue  # Пропускаем нулевые элементы

                    # Знак: (-1)^(row_pos + col_pos)
                    sign_power = row_pos + col_pos
                    sign_numerator = Integer(0, 1, [1]) if sign_power % 2 == 0 else Integer(1, 1, [1])
                    sign = Rational(sign_numerator, Natural(1, [1]))

                    # Новые строки (без текущей)
                    new_rows = rows[:row_pos] + rows[row_pos + 1:]

                    # Рекурсивный вызов для минора
                    minor_det = det_recursive(new_rows, new_cols)

                    # Вклад в определитель
                    term = sign * element * minor_det
                    result = result + term

            return result

        # Начинаем со всех строк и столбцов
        all_rows = list(range(n))
        all_cols = list(range(n))

        return det_recursive(all_rows, all_cols)


    def det_Gauss(self):
        """
        Определитель методом Гаусса через приведение к треугольному (ступенчатому) виду

        Вычитание из одной строки другой, умноженной на константу 'k' по ходу метода Гаусса не меняют определитель.
        Достаточно рассмотреть преобразование для матрицы 2х2,
        так как к этому случаю можно свести при разложении определителя по строке/столбцу:
        det[[a, b], [c, d]] = ad - bc
        det[[a - kc, b - kd], [c, d]] = ad - kcd - bc + kcd = ad - bc

        O(n^3). Самое оптимальное из реализованных методов для n > 3
        * смог посчитать определеитель для случайной матрицы 30x30
        => Используем

        Возвращает: произведение диагональных элементов с учётом перестановок строк
        """
        n = self.size
        if n <= 3:
            return self.det_formula()


        # Создаём копию матрицы, чтобы не портить оригинал
        matrix = [row[:] for row in self.arr]           # в конце алгоритма она станет ступенчатой

        swaps = 0  # Счётчик перестановок строк

        # Начинаем с определителя = 1
        det = ONE_RATIONAL

        for i in range(n):
            # Поиск ненулевого элемента в i-м столбце (ведущего)
            pivot_row = i

            while pivot_row < n and matrix[pivot_row][i] == ZERO_RATIONAL:
                pivot_row += 1

            if pivot_row == n:
                # Весь столбец нулевой -> определитель = 0
                return ZERO_RATIONAL

            if pivot_row != i:
                # Меняем строки местами
                matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
                swaps += 1  # Увеличиваем счётчик перестановок

            # Ведущий элемент
            pivot = matrix[i][i]
            # Умножаем определитель на ведущий элемент
            det = det * pivot
            if det == ZERO_RATIONAL:
                return ZERO_RATIONAL

            # Обнуляем элементы под ведущим
            for j in range(i + 1, n):
                if matrix[j][i] != ZERO_RATIONAL:
                    # Коэффициент для вычитания
                    factor = matrix[j][i] / pivot

                    # Вычитаем из строки j строку i, умноженную на factor
                    for k in range(i, n):
                        matrix[j][k] = matrix[j][k] - factor * matrix[i][k]

        # Учёт знака от перестановок строк: (-1)^swaps
        sign_numerator = Integer(0, 1, [1]) if swaps % 2 == 0 else Integer(1, 1, [1])
        sign_rational = Rational(sign_numerator, Natural(1, [1]))

        return sign_rational * det

# ================================================= INVERSE =================================================

    # 2 Inverse matrix algorithms
    def Gauss_inv(self):
        """
        Поиск обратной матрицы по методу Гаусса
        [A | E] ~ [E | A^-1], E - единичная

        Сложность: O(n^3), (лучше, чем метод союзной матрицы)
        * за +- адекватное время считает для n <= 12
        * за ~4 минуты вычислил обратную для 20х20, подробнее в research
        => Используем

        Алгоритм:
        1. Создаём [A | E]
        2. Прямой ход: приводим A к верхнетреугольному виду
        3. Обратный ход: приводим A к единичной матрице
        4. Получаем [E | A^-1]

        Оптимальнее объединить прямой и обратный ход, и за 1 проход сводить к единичной
        """
        n = self.size

        # Проверка определителя (вырожденные матрицы необратимы)
        det = self.det_Gauss()
        if det == ZERO_RATIONAL:
            raise ValueError("Матрица вырождена (det = 0), обратной не существует")

        # Шаг 1: Создаём расширенную матрицу [A | E]
        # Размер n х (2n): слева исходная A, справа единичная E
        augmented = []

        for i in range(n):
            row = []
            # Копируем строку из A
            row.extend(self.arr[i])
            # Добавляем правую часть: единичную матрицу
            for j in range(n):
                if i == j:
                    row.append(ONE_RATIONAL)  # 1 на диагонали
                else:
                    row.append(ZERO_RATIONAL)  # 0 вне диагонали
            augmented.append(row)

        # Шаг 2 + 3: Прямой и обратный ход - приводим к виду единичной матрицы
        for i in range(n):
            # Поиск ведущего элемента (ненулевого) в i-м столбце
            pivot_row = i
            while pivot_row < n and augmented[pivot_row][i] == ZERO_RATIONAL:
                pivot_row += 1

            if pivot_row == n:
                # Не должно случиться, т.к. det != 0, но на всякий случай
                raise ValueError("Матрица вырождена в процессе вычисления")

            # Если ведущий элемент не на диагонали, меняем строки местами
            if pivot_row != i:
                augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

            # Нормировка i-й строки: делаем элемент [i][i] = 1
            pivot = augmented[i][i]
            for j in range(i, 2 * n):
                augmented[i][j] = augmented[i][j] / pivot

            # Обнуление i-го столбца в других строках
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    if factor != ZERO_RATIONAL:    # чтобы не выполнять лишнее бессмысленное действие
                        for j in range(i, 2 * n):
                            augmented[k][j] = augmented[k][j] - factor * augmented[i][j]

        # Шаг 4: Извлекаем обратную матрицу из правой части
        inverse_data = []
        for i in range(n):
            row = []
            for j in range(n):
                # Берём элементы из правой половины (столбцы n..2n-1)
                row.append(augmented[i][n + j])
            inverse_data.append(row)

        return Matrix(n, inverse_data)


    def adjugate_inv(self):
        """
        Обратная матрица методом алгебраических дополнений
        Формула: A^-1 = (1/det(A)) * adj(A), где adj(A) - союзная матрица

        Сложность: O(n^5) - вычисление n^2 определителей размера (n-1): n^2 * (n-1)^3 ~ n^5 (так как считаем определитель за куб по методу Гаусса)
        Если не по методу Гаусса, то n^2 * (n - 1)! = n^2 * n! / n = n * n! (еще хуже)

        Считает за приемлемое время только для n <= 7
        => Не используем
        """
        n = self.size

        # Шаг 1: Вычисляем определитель
        det = self.det_Gauss()
        if det == ZERO_RATIONAL:
            raise ValueError("Матрица вырождена (det = 0), обратной не существует")

        # Шаг 2: Создаём матрицу алгебраических дополнений
        adjoint_data = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                # Вычисляем минор M_ij (матрица без i-й строки и j-го столбца)
                minor_data = []
                for row_idx in range(n):
                    if row_idx == i:
                        continue  # Пропускаем i-ю строку
                    row = []
                    for col_idx in range(n):
                        if col_idx == j:
                            continue  # Пропускаем j-й столбец
                        row.append(self.arr[row_idx][col_idx])
                    minor_data.append(row)

                minor = Matrix(n - 1, minor_data)

                # Вычисляем определитель минора
                minor_det = minor.det_Gauss()

                # Знак алгебраического дополнения: (-1)^(i+j)
                sign_power = i + j
                sign = ONE_RATIONAL if sign_power % 2 == 0 else Rational(Integer(1, 1, [1]), Natural(1, [1]))

                # Алгебраическое дополнение
                cofactor = sign * minor_det

                # В союзной матрице элементы транспонируются: adj[i][j] = C_ji
                adjoint_data[j][i] = cofactor

        # Шаг 3: Делим союзную матрицу на определитель
        # A^-1 = (1/det) * adj(A)
        inverse_data = [[adjoint_data[i][j] / det for j in range(n)] for i in range(n)]

        return Matrix(n, inverse_data)


# ================================================= POW =================================================

    # 2 Powerings + edge cases (future __pow__)
    def naive_pow(self, power: int):
        """
        Возведение матрицы в степень power
        через умножение (power - 1) раз -> O(power) умножений
        (неэффективно => не используем)
        """
        if not isinstance(power, int):
            raise TypeError(f"Степень должна быть целым числом, получено {type(power)}")

        n = self.size

        if power == 0:
            return Matrix.identity(n)

        if power < 0:  # Отрицательные степени: A^(-k) = (A^(-1))^k
            try:
                base = self.Gauss_inv()
            except ValueError as e:
                raise ValueError(f"Нельзя возвести в отрицательную степень: {e}")

            power = -power      # => power > 0 -> general case

        else:
            base = self

        res = base
        for i in range(power - 1):
            res = res.naive_mul(base)       # используем то произведение, что быстрeе (по определению)

        return res


    def quick_powering(self, power: int):
        """
        Возведение матрицы в степень power
        через алгоритм быстрого возведения в степень -> O(log |power|) умножений
        Эффективнее обычного + легкость реализации + универсален
        => Используем

        Здесь есть проверки на граничные случаи, кроме диагональности (для этого отдельный метод)
        """

        if not isinstance(power, int):
            raise TypeError(f"Степень должна быть целым числом, получено {type(power)}")
        n = self.size


        # Проверки граничных случаев, упрощающих вычисления
        if self.is_one_mat:   # E^k = E ∀ k
            return self

        if power == 0:
            return Matrix.identity(n)

        if self.is_zero_mat and power > 0:
            return self

        if power == 1:
            return self

        if power == -1:
            return self.Gauss_inv()

        # Проверка на идемпотентность: A*A = A => A^k = A, k >= 2   ### Принято решение убрать, тк проверка дорогая, а встречаются они редко
        # if power >= 2:
        #     if self.naive_mul(self) == self:   # O(n^3), но для идемпотентных экономия
        #         return self


        if power < 0:  # Отрицательные степени: A^(-k) = (A^(-1))^k
            try:
                base = self.Gauss_inv()
            except ValueError as e:
                raise ValueError(f"Нельзя возвести в отрицательную степень: {e}")

            power = -power  # => power > 0 -> general case

        else:
            base = self

        # Основной алгоритм бинарного возведения в степень: base ^ power
        res = Matrix.identity(n)

        while power > 0:
            if power & 1:  # Если степень нечётная
                res = res.naive_mul(base)
            base = base.naive_mul(base)  # Возводим в квадрат
            power >>= 1  # Делим степень на 2

        return res


    # возведение в степень особых матриц + проверка на особые матрицы (неуниверсальные методы)

    def is_identity(self) -> bool:
        """
        Проверка матрицы на единичную. Применение:
        Результат проверки сохраняется в конструкторе, как поле,
        так как это удобно использовать при умножении, сложении матриц и возведении в степень.
        Сложность O(n^2), но есть ранняя остановка при первом нарушении соответствия.
        """
        n = self.size
        for i in range(n):
            for j in range(n):
                expected = ONE_RATIONAL if i == j else ZERO_RATIONAL
                if self.arr[i][j] != expected:
                    return False

        return True


    def is_zero_matrix(self):
        """
        Проверка матрицы на нулевую. Применение:
        Результат проверки сохраняется в конструкторе как поле,
        так как это удобно использовать при умножении, сложении матриц и возведении в степень.
        Сложность O(n^2), но есть ранняя остановка при первом нарушении соответствия.
        """
        n = self.size
        for i in range(n):
            for j in range(n):
                if self.arr[i][j] != ZERO_RATIONAL:
                    return False

        return True


    def is_diag(self):
        """
        Проверка матрицы на диагональную для оптимизации возведения в степень.
        O(n^2) + early stopping
        """
        n = self.size
        for i in range(n):
            for j in range(n):
                if i != j and self.arr[i][j] != ZERO_RATIONAL:
                    return False

        return True


    def pow_diag(self, power: int):
        """
        Алгоритм возведения в степень диагональной матрицы
        O(n) возведений рациональных чисел в степень (тоже реализовано бинарно в Rational), копирование - O(n^2)
        Метод вызывается только после проверки на диагональность.
        => Используем для диагональных/диагонализируемых
        """
        n = self.size

        # Проверка, вдруг диагональная матрица - скалярная (const * E) -> O(n)
        is_scalar = True
        pivot = self.arr[0][0]
        for i in range(1, n):
            if self.arr[i][i] != pivot:
                is_scalar = False
                break

        # Скалярная матрица, 1 возведение дроби в степень, тк (const * E)^k = const^k * E^k = const^k * E
        if is_scalar:
            pivot_powered = pivot ** power
            res = [
                [
                    pivot_powered if i == j else ZERO_RATIONAL
                    for j in range(n)
                ]
                for i in range(n)
            ]

        # Обычная диагональная не скалярная матрица, n возведений дроби в степень
        else:
            res = [
                [
                    self.arr[i][j] ** power if i == j else ZERO_RATIONAL
                    for j in range(n)
                ]
                for i in range(n)
            ]
        return Matrix(n, res)


    def get_eigen(self):
        """
        Попытка вычислить собственные числа и вектора для матриц 2 х 2,
        если нет иррациональности в ходе поиска

        Для матрицы 2 х 2 вида A = [[a, b], [c, d]] распишем характеристическое уравнение:
        A * v = lmbd * v  <=> lmbd - собственное значение, v - собственный вектор (v != 0)
        (A - lmbd * E) * v = 0, v != 0. Значит преобразование матрицей (A - lmbd * E) схлопывыет пространство,
        так как ненулевой вектор v после преобразования стал нуль-вектором, значит есть линейная зависимость, значит
        det(A - lmbd * E) = 0. Распишем определитель и приравняем к нулю:
        (a - lmbd) * (d - lmbd) - bc = ad - bc + lmbd^2 - lmbd (a + d)
        detA = ad - bc, trA = a + d, тогда
        X(lmbd) = lmbd^2 - trA * lmbd + detA    -   Характеристическое уравнение
        D = trA^2 - 4detA,  чтобы были рациональные корни (точные, не приближенные), надо чтобы корень из данного выражения был рациональным

        Метод возвращает: (eigenvalues, eigenvectors) или None если есть иррациональности
        eigenvalues: список из 2 Rational
        eigenvectors: список из 2 списков [Rational, Rational]
        """

        n = self.size
        if n != 2:
            raise ValueError("Метод работает только для матриц 2 x 2")

        a, b, c, d = self.arr[0][0], self.arr[0][1], self.arr[1][0], self.arr[1][1]
        tr = a + d
        det = a * d - b * c

        D = tr * tr - det * Rational(Integer(0, 1, [4]), Natural(1, [1]))

        sqrt_D = self._get_sqrt(D)

        if not sqrt_D:
            print("Иррациональность в корне из дискриминанта")    # info output
            return None

        # Ищем собственные значния - корни характеристического уравнения
        two = Rational(Integer(0, 1, [2]), Natural(1, [1]))
        lmbd1 = (tr + sqrt_D) / two
        lmbd2 = (tr - sqrt_D) / two

        eigenvalues = [lmbd1, lmbd2]

        # Теперь ищем собственные векторы
        eigenvectors = []

        for lmbd in eigenvalues:
            # Решаем (A - lmbd * E) * v = 0

            # Матрица A - lmbd * E
            a_lmbd = a - lmbd
            d_lmbd = d - lmbd

            # Находим ненулевое решение системы:
            # [a-lmbd,   b] [x] = [0]
            # [c,   d-lmbd] [y]   [0]

            # Рассмотрим случаи:

            # Случай 1: если b != 0, можно взять y = 1, тогда x = -b/(a-lmbd)
            if b != ZERO_RATIONAL:
                if a_lmbd != ZERO_RATIONAL:
                    x = -b / a_lmbd
                    y = ONE_RATIONAL
                else:
                    # a-lmbd = 0, значит уравнение: b*y = 0 => y = 0 (тривиально)
                    # Попробуем использовать второе уравнение
                    x = ONE_RATIONAL
                    y = ZERO_RATIONAL if c == ZERO_RATIONAL else -c / d_lmbd

            # Случай 2: b = 0, но c != 0
            elif c != ZERO_RATIONAL:
                # Можно взять x = 1, тогда из второго уравнения: c + (d-lmbd)*y = 0
                x = ONE_RATIONAL
                if d_lmbd != ZERO_RATIONAL:
                    y = -c / d_lmbd
                else:
                    # d-lmbd = 0, значит уравнение: c*x = 0 => x = 0 (тривиально)
                    y = ONE_RATIONAL

            # Случай 3: b = 0 и c = 0 (диагональная матрица)
            else:
                # Матрица диагональная: [[a, 0], [0, d]]
                # Собственные значения: lmbd1 = a, lmbd2 = d

                if lmbd == a:
                    # Собственный вектор для lmbd = a: [1, 0]
                    x = ONE_RATIONAL
                    y = ZERO_RATIONAL
                else:  # lmbd = d
                    # Собственный вектор для lmbd = d: [0, 1]
                    x = ZERO_RATIONAL
                    y = ONE_RATIONAL

            # Добавляем вектор
            eigenvectors.append([x, y])

        return eigenvalues, eigenvectors


    @staticmethod
    def _get_sqrt(q: Rational):
        """
        Попытка извлечения корня из рационального числа.
        Необходимо для поиска корня из дискриминанта

        q = a/b => sqrt(q) = sqrt(a/b) = (sqrt(a)) / (sqrt(b))
        """

        # Небольшая оптимизация: квадрат целого числа не заканчивается на 2, 3, 7, 8
        for forbidden_num in [2, 3, 7, 8]:
            if q.numerator.A[-1] == forbidden_num or q.denominator.A[-1] == forbidden_num:
                return None

        # для удобства оба переведем в int
        a = int(q.numerator.show())
        b = int(q.denominator.show())

        if a < 0:
            return None
        if int(a ** 0.5) ** 2 == a:
            sqrt_a = int(a ** 0.5)
        else:
            return None

        if int(b ** 0.5) ** 2 == b:
            sqrt_b = int(b ** 0.5)
        else:
            return None

        s = 0 if sqrt_a > 0 else 1
        a_sqrt_arr = [int(elem) for elem in str(abs(sqrt_a))]
        integer_sqrt_a = Integer(s, len(a_sqrt_arr) - 1, a_sqrt_arr)

        b_sqrt_arr = [int(elem) for elem in str(abs(sqrt_b))]
        natural_sqrt_b = Natural(len(b_sqrt_arr) - 1, b_sqrt_arr)

        return Rational(integer_sqrt_a, natural_sqrt_b)


    def diagonalize_2x2(self):
        """
        Диагонализация матрицы 2x2 над Q, если возможно

        Возвращает: (P, D) такие что A = P * D * P^(-1)
        или None если нельзя диагонализовать над Q
        """
        n = self.size
        if n != 2:
            raise ValueError("Метод работает только для матриц 2 x 2")

        result = self.get_eigen()
        if result is None:
            return None

        eigenvalues, eigenvectors = result

        # Проверяем, что собственные векторы линейно независимы
        v1, v2 = eigenvectors[0], eigenvectors[1]

        # Проверка линейной независимости: определитель матрицы из векторов != 0
        det_v = v1[0] * v2[1] - v1[1] * v2[0]
        if det_v == ZERO_RATIONAL:
            print("Собственные векторы линейно зависимы")
            return None

        # Матрица P из собственных векторов
        P = Matrix(2, [
            [v1[0], v2[0]],
            [v1[1], v2[1]]
        ])

        # Диагональная матрица D
        D = Matrix(2, [
            [eigenvalues[0], ZERO_RATIONAL],
            [ZERO_RATIONAL, eigenvalues[1]]
        ])

        return P, D


    def diag_and_pow_2x2(self, power):
        """
        Интеграция с предыдущим методом diagonalize_2x2
        После удавшейся диагонализации можно оптимизировать возведение в степень
        A ~ D => A = P * D * P^-1 =>
        A^k = (P * D * P^-1)^k = P * D^k * P^-1

        Свойство (почему можно так диагонализировать):
        A * P = P * D, где P - матрица в стобцах которой айгенвектора, D - диагональная матрица из айгензначений
        A * [v1, v2] = [v1, v2] * diag(lmbd1, lmbd2) и если это раскрыть, то получится как раз определение айген векторов и чисел:
        A * v_i = lmbd_i * v_i

        При провале диагонализации fallback на бинарное возведение в степень

        На практике порой реально попадаются случаи без иррациональности, и оно действительно считает быстрее
        К примеру, 9.33 сек vs 25.56 сек (quick_pow) для возведения матрицы 2х2 в 500 степень
        Главный минус в том, что несложно посчитать можно только для 2х2 матриц, так как аналитический поиск собственных значений и векторов
        очень усложняется с увеличением размера матрицы.
        Так же ограничением является рассмотрение только рациональных собственных значений, которые для 2x2 можно быстро посчитать аналитически
        """

        result = self.diagonalize_2x2()
        if result is None:
            print("Диагонализация не удалась. Вычисление через quick_powering")   # info output
            return self.quick_powering(power)

        P, D = result
        P_inv = P.Gauss_inv()

        D_powered = D.pow_diag(power)
        temp = P.naive_mul(D_powered)

        return temp.naive_mul(P_inv)


    def analytical_2x2_pow(self, power):
        """
        Возведение матрицы 2x2 в степень через аналитические формулы (Следствие из теоремы Гамильтона-Кэли).
        При выходе из поля рациональных чисел откат на бинарное возведение в степень.

        Откуда берутся формулы:
        1) Теорема Гамильтона-Кэли: любая квадратная матрица удовлетворяет своему характеристическому уравнению X.
        То есть X(A) = 0.
        В свою очередь характеристический полином можно записать в виде
        lmbd^n + c_n-1 * lmbd^(n-1) + ... + c_1 * lmbd + c_0
        2) Тогда для матрицы 2х2 A = [[a, b], [c, d]] по Гамильтону-Кэли:
        X(lmbd) = lmbd^2 - (a+d)lmbd + (ad - bc) = 0
        X(A) = A^2 - (a+d)*A + (ad - bc)*E = 0
        A^2 = (a+d)*A - (ad - bc)*E
        Отсюда получаем, что A^n = alpha*A + beta*E для матриц 2х2
        3) Поиск коэффициентов alpha, beta:
        Av = lmbd*v, по индукции A^n * v = lmbd^n * v
        A^n * v = (alpha*A + beta) * v = aplha * Av + beta * v = (alpha * lmbd + beta) * v
        Приравняем два полученных A^n * v, получаем coeff1 * v = coeff2 * v => coeff1 = coeff2:
        lmbd^n = (alpha * lmbd + beta)

        4) Находим два собственных значения, подставляем их в уравнение
        и чтобы найти коэффициенты решаем сисему методом подстановки.
        В ходе решения возникает знаменатель lmbd1-lmbd2, что дает деление на ноль при кратных корнях,
        поэтому в этом случае применяем предельный переход при lmbd1 -> lmbd2 и раскрытие неопределенности [0/0] через Лопиталя

        * На практике работает чаще быстрее, чем диагонализация, так как тут нет обращения матрицы и матричных произведений.
        * Оба метода имеют два возведения в степень рационального числа.
        * !!! Также на практике было замечено, что для матрицы [[a, b], [c, d]] с с=0 диагонализация работает значительно быстрее,
        так как в коде это отдельная ветка условий, в которой нахождение собственных векторов проще, и матрица P также проще, что ускоряет ее обращение

        Сравнению аналитического метода , диагонализации и бинарного возведения в степень в случае иррациональности посвещены отдельные графики в исследовании.
        """
        n = self.size
        if n != 2:
            raise ValueError("Метод работает только для матриц 2 x 2")

        # Константы
        zero = Rational(Integer(0, 1, [0]), Natural(1, [1]))
        one = Rational(Integer(0, 1, [1]), Natural(1, [1]))
        two = Rational(Integer(0, 1, [2]), Natural(1, [1]))
        four = Rational(Integer(0, 1, [4]), Natural(1, [1]))

        if power < 0:
            inv = self.Gauss_inv()
            return inv.analytical_2x2_pow(-power)

        # Базовые случаи
        if power == 0:
            return Matrix.identity(2)
        if power == 1:
            return self

        a, b, c, d = self.arr[0][0], self.arr[0][1], self.arr[1][0], self.arr[1][1]

        # Вычисляем собственные значения
        tr = a + d  # след
        det = a * d - b * c  # определитель

        # Дискриминант: D = tr^2 - 4*det
        D = tr * tr - det * four

        # Пробуем извлечь точный квадратный корень
        sqrt_D = self._get_sqrt(D)
        if sqrt_D is None:
            # Дискриминант не является полным квадратом рационального числа
            print("Аналитические вычисления не удались. Вычисление через quick_powering")
            return self.quick_powering(power)

        # Собственные значения
        lmbd1 = (tr + sqrt_D) / two
        lmbd2 = (tr - sqrt_D) / two

        # Случай 1: различные собственные значения
        if lmbd1 != lmbd2:
            # lmbd1^power и lmbd2^power
            lmbd1_pow = lmbd1 ** power
            lmbd2_pow = lmbd2 ** power

            # Вычисляем lmbd1^(power-1) и lmbd2^(power-1)
            if power > 0:
                lmbd1_pow_m1 = lmbd1 ** (power - 1)
                lmbd2_pow_m1 = lmbd2 ** (power - 1)
            else:
                # Для power = 0 обрабатываем особо
                lmbd1_pow_m1 = one / lmbd1 if lmbd1 != zero else zero
                lmbd2_pow_m1 = one / lmbd2 if lmbd2 != zero else zero

            denominator = lmbd1 - lmbd2

            # alpha = (lmbd1^power - lmbd2^power) / (lmbd1 - lmbd2)
            alpha = (lmbd1_pow - lmbd2_pow) / denominator

            # beta = -lmbd1*lmbd2 * (lmbd1^(power-1) - lmbd2^(power-1)) / (lmbd1 - lmbd2)
            beta = (zero - lmbd1 * lmbd2) * (lmbd1_pow_m1 - lmbd2_pow_m1) / denominator

            # A^power = alpha * A + beta * E
            result = Matrix(2, [
                [alpha * a + beta, alpha * b],
                [alpha * c, alpha * d + beta]
            ])

        else:  # lmbd1 = lmbd2 = lmbd
            lmbd = lmbd1
            lmbd_pow = lmbd ** power

            if power == 0:
                lmbd_pow_minus_1 = zero
            else:
                lmbd_pow_minus_1 = lmbd ** (power - 1)

            # Формула для кратных корней:
            # alpha = n * lmbd^(n-1), beta = -(1 - n)*lmbd^n, n = power

            # A^power = power * lmbd^(power-1) * A - (power-1) * lmbd^power * E
            power_rat = Rational(Integer(0, 1, [power]), Natural(1, [1]))
            power_minus_1_rat = Rational(Integer(0, 1, [power - 1]), Natural(1, [1]))

            coeff1 = power_rat * lmbd_pow_minus_1  # power * lmbd^(power-1)
            coeff2 = power_minus_1_rat * lmbd_pow  # (power-1) * lmbd^power

            # формула coeff1 * A - coeff2 * E
            # Поэтому вычитаем coeff2 на диагонали
            result = Matrix(2, [
                [coeff1 * a - coeff2, coeff1 * b],
                [coeff1 * c, coeff1 * d - coeff2]
            ])

        return result


    # NEW APPROACH: Hamilton-Cayley Theorem

    def _get_characteristic_polynomial(self):
        """
        Вычисление характеристического полинома для матриц размера 2, 3, 4 по заранее выведенным формулам
        Для больших размеров вычисление коэффициентов характеристического полинома через метод Фаддеева-Леверье

        Если расписывать определители det(A - lmbd*E), можно заметить закономерность:
        Коэффициент при lmbd^(n-k) в характеристическом полиноме равен (-1)^k * (сумма всех главных миноров размера k)
        """
        n = self.size

        if n == 2:
            a, b, c, d = self.arr[0][0], self.arr[0][1], self.arr[1][0], self.arr[1][1]
            # Вычисляем след и определитель
            tr = a + d
            det = a * d - b * c

            # p(x) = x^2 - tr*x + det
            return Polynomial(2, [ONE_RATIONAL, -tr, det])

        elif n == 3:
            a, b, c = self.arr[0][0], self.arr[0][1], self.arr[0][2]
            d, e, f = self.arr[1][0], self.arr[1][1], self.arr[1][2]
            g, h, i = self.arr[2][0], self.arr[2][1], self.arr[2][2]

            # След
            tr = a + e + i

            # Сумма главных миноров 2x2
            m1 = a * e - b * d  # минор для элемента (3,3)
            m2 = a * i - c * g  # минор для элемента (2,2)
            m3 = e * i - f * h  # минор для элемента (1,1)
            sum_minors = m1 + m2 + m3

            # Определитель
            det = self.det3x3(a, b, c, d, e, f, g, h, i)

            # p(x) = x^3 - tr*x^2 + sum_minors*x - det
            return Polynomial(3, [ONE_RATIONAL, -tr, sum_minors, -det])

        elif n == 4:
            a, b, c, d = self.arr[0][0], self.arr[0][1], self.arr[0][2], self.arr[0][3]
            e, f, g, h = self.arr[1][0], self.arr[1][1], self.arr[1][2], self.arr[1][3]
            i, j, k, l = self.arr[2][0], self.arr[2][1], self.arr[2][2], self.arr[2][3]
            m, n, o, p = self.arr[3][0], self.arr[3][1], self.arr[3][2], self.arr[3][3]

            # След
            tr = a + f + k + p

            # Сумма главных миноров 3x3
            m1 = self.det3x3(f, g, h,  # минор для элемента (1,1)
                        j, k, l,
                        n, o, p)

            m2 = self.det3x3(a, c, d,  # минор для элемента (2,2)
                        i, k, l,
                        m, o, p)

            m3 = self.det3x3(a, b, d,  # минор для элемента (3,3)
                        e, f, h,
                        m, n, p)

            m4 = self.det3x3(a, b, c,  # минор для элемента (4,4)
                        e, f, g,
                        i, j, k)

            sum_minors_3x3 = m1 + m2 + m3 + m4

            # Сумма всех C(4, 2) = 6 главных миноров 2x2
            s1 = self.det2x2(a, b, e, f) + self.det2x2(a, c, i, k) + self.det2x2(a, d, m, p)
            s2 = self.det2x2(f, g, j, k) + self.det2x2(f, h, n, p) + self.det2x2(k, l, o, p)
            sum_minors_2x2 = s1 + s2

            det = self.det_Gauss()

            # p(x) = x^4 - tr*x^3 + sum_minors_2x2*x^2 - sum_minors_3x3*x + det
            return Polynomial(4, [ONE_RATIONAL, -tr, sum_minors_2x2, -sum_minors_3x3, det])

        else:
            # Метод Фаддеева-Леверье, так как для больших размеров труднее расписывать формулы вручную
            return self.Faddeev_LeVerrier_algorithm()


    # Вспомогательные методы для вычисления определителей для метода _get_characteristic_polynomial
    @staticmethod
    def det2x2(a, b, c, d):
        """Определитель матрицы 2x2 [[a,b],[c,d]]"""
        return a * d - b * c
    @staticmethod
    def det3x3(a, b, c, d, e, f, g, h, i):
        """Определитель матрицы 3x3 [[a,b,c],[d,e,f],[g,h,i]] (правило Сарруса)"""
        return (a * e * i + b * f * g + c * d * h) - (c * e * g + b * d * i + a * f * h)


    def Faddeev_LeVerrier_algorithm(self):
        """
        Итеративный алгоритм для нахождения характеристического полинома без вычисления определителей

        Его сложность O(n^4) так как n раз вычисляется матричное произведение сложностью O(n^3)
        """
        n = self.size

        B = Matrix.identity(n)    # B0 = E
        coeffs = [ONE_RATIONAL]   # c0 = 1

        for k in range(1, n + 1):
            # Вычисляем A*B_{k-1}
            AB = self.naive_mul(B)

            # c_k = -tr(AB) / k
            tr = AB.trace()
            k_rat = Rational(Integer(0, 0, [int(el) for el in str(k)]), Natural(0, [1]))
            c_k = -tr / k_rat
            coeffs.append(c_k)

            if k < n:
                # B_k = AB + c_k*E
                B = AB + Matrix.identity(n).multiply_by_const(c_k)

        return Polynomial(n, coeffs)


    def Hamilton_Cayley_pow(self, power: int):
        """
        Нахождение A^power с использованием теоремы Гамильтона-Кэли:

        Суть метода:
        Пусть есть характеристический полином p(x) для матриц небольшого размера, найденный в методе _get_characteristic_polynomial
        Его степень равна размеру матрицы.
        Найдем x^power mod p(x) = r(x), тогда существует q(x) такое что x^n = q(x)*p(x) + r(x)
        Подставим x := A (исходная матрица)
        Получим A^power = q(A) * p(A) + r(A). А по теореме Гамильтона-Кэли p(A) = 0, тогда
        A^power = r(A). Мы свели задачу возведения матрицы в степень к нахождению остатка от деления x^power на характеристический полином.
        Также стоит отметить, что deg(r(x)) < n, где n - размер матрицы.
        Это значит, что вычисление полинома с рациональными коэффициентами от матрицы не составит особой вычислительной трудности, так как размер матрицы небольшой.

        Ограничение метода:
        1) Сложность вычисления характеристического полинома возрастает с увеличением размера матрицы, поэтому рассматриваем небольшие размеры
        2) Необходимо, чтобы поиск остатка от деления многочленов был оптимальным по сложности

        Преимущества:
        В отличие от рассмотренных ранее методов диагонализации и аналитики для матриц 2х2,
        данный метод более универсален, так как он не опирается на собственные значения, которые могут быть иррациональными
        """
        if power < 0:
            inv = self.Gauss_inv()
            return inv.Hamilton_Cayley_pow(-power)


        characteristic_poly = self._get_characteristic_polynomial()

        # NOT OPTIMIZED APPROACH: Использование неоптимизированных алгоритмов для общего случая деления полиномов
        # powered_x = Polynomial(power, [ONE_RATIONAL] + [ZERO_RATIONAL for _ in range(power)] )
        # rest_poly = powered_x % characteristic_poly

        rest_coeffs = self._poly_mod_pow_fast(power, characteristic_poly.C)

        # Вычисление полинома от матрицы (последовательное вычисление степеней)
        n = self.size
        result = Matrix.zero(n)
        powered_X = Matrix.identity(n)  # X^0
        deg = len(rest_coeffs) - 1

        # Идём от младшего коэффициента к старшему
        for i in range(deg, -1, -1):
            coeff = rest_coeffs[i]

            if coeff != ZERO_RATIONAL:
                result = result + powered_X.multiply_by_const(coeff)

            # Подготавливаем следующую степень (если не последняя итерация)
            if i > 0:
                powered_X = powered_X.naive_mul(self)

        return result


    # OPTIMIZED POLYNOMIAL METHODS
    def _poly_mod_pow_fast(self, power: int, mod_coeffs: list):
        """
        Оптимизированное вычисление x^power mod p(x)
        Суть: вычисление остатка не по классическому алгоритму вида
        a mod b = a - b * (a // b),
        а вычисление x^n бинарным возведением в степень, но вычисления производятся по модулю p(x)

        mod_coeffs: [1, c_{n-1}, ..., c_0] (от старшего к младшему)
        Возвращает: [r_{n-1}, ..., r_0]
        """
        n = len(mod_coeffs) - 1  # степень полинома

        # Базовый случай: полином степени 0
        if n == 0:
            return [ZERO_RATIONAL]  # любой полином mod константа = 0

        # Представляем x как вектор длины n
        x_vec = [ZERO_RATIONAL] * n
        x_vec[-2] = ONE_RATIONAL  # предпоследний элемент = коэф. при x^1

        # Результат: 1 = 0*x^{n-1} + ... + 0*x + 1
        result = [ZERO_RATIONAL] * n
        result[-1] = ONE_RATIONAL  # последний элемент = 1

        base = x_vec[:]
        p = power

        while p > 0:
            if p & 1:
                result = self._poly_mod_mul_fast(result, base, mod_coeffs)
            base = self._poly_mod_mul_fast(base, base, mod_coeffs)
            p >>= 1

        return result


    @staticmethod
    def _poly_mod_mul_fast(a: list, b: list, mod_coeffs: list):
        """
        Быстрое умножение многочленов по модулю
        Вспомогательный для _poly_mod_pow_fast
        """
        n = len(a)  # длина векторов = степень модуля
        degree = len(mod_coeffs) - 1  # степень модуля

        # 1. Умножение (максимальная степень 2n-2)
        prod = [ZERO_RATIONAL] * (2 * n - 1)

        for i in range(n):
            if a[i] == ZERO_RATIONAL:
                continue
            for j in range(n):
                if b[j] == ZERO_RATIONAL:
                    continue
                prod[i + j] = prod[i + j] + a[i] * b[j]

        # 2. Приведение по модулю
        # mod_coeffs = [1, c_{degree-1}, ..., c0]
        # x^degree ≡ -c_{degree-1}x^{degree-1} - ... - c0 mod p(x)
        for i in range(2 * n - degree - 1): # проход по степеням, которые >= degree, их нужно заменить на остаток
            if prod[i] != ZERO_RATIONAL:
                coeff = prod[i]
                # Обнуляем текущую позицию
                prod[i] = ZERO_RATIONAL

                # Вычитаем coeff * x^i * p(x)
                for j in range(degree + 1):
                    prod[i + j] = prod[i + j] - coeff * mod_coeffs[j]

        # 3. Берем только последние n элементов (степени < n)
        return prod[-n:]


    @staticmethod
    def identity(n: int):
        """
        Создание единичной матрицы размера n
        Нейтральный элемент по умножению
        """
        arr = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
        for i in range(n):
            arr[i][i] = ONE_RATIONAL
        return Matrix(n, arr)


    @staticmethod
    def zero(n: int):
        """
        Создание нулевой матрицы
        Нейтральный элемент по сложению, но вырожденная
        """
        arr = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
        return Matrix(n, arr)


    def __str__(self):
        """
        Строковое представление матрицы
        для удобного восприятия при выводе в консоль
        """

        res = ""
        for i in range(self.size):
            res += " ".join([str(elem) for elem in self.arr[i]])
            res += "\n"

        return res


    def __eq__(self, other):
        """Сравнение матриц по их составляющему"""
        if not isinstance(other, Matrix) or self.size != other.size:
            return False
        for i in range(self.size):
            for j in range(self.size):
                if self.arr[i][j] != other.arr[i][j]:
                    return False
        return True



class MatrixPolynomial:
    def __init__(self, degree: int, coefficients: list[Matrix]):
        pivot = coefficients[0].size
        for i in range(len(coefficients)):
            if coefficients[i].size != pivot:
                raise ValueError("Все коэффициенты полинома должны быть одного размера")

        if degree != (len(coefficients) - 1):
            raise ValueError("Данные degree и coefficients не соответствуют друг другу")

        self.degree = degree              # степень многочлена
        self.coefficients = coefficients  # массив коэффициентов (от старшего к младшему)
        self.size = coefficients[0].size

        # Предварительный анализ полинома на "заполненность": разреженный или плотный
        self._analyze_polynomial()


    def _analyze_polynomial(self):
        """Анализ полинома для выбора оптимального метода вычисления"""
        zero_matrix = Matrix.zero(self.size)
        zero_count = 0

        for coeff in self.coefficients:
            if coeff == zero_matrix:
                zero_count += 1

        total_terms = len(self.coefficients)
        self.zero_ratio = zero_count / total_terms if total_terms > 0 else 0

        # Правило: чем выше степень, тем ниже порог для бинарного метода
        if self.degree >= 100:  # Очень высокая степень
            threshold = 0.5     # 50% нулей => бинарный
        elif self.degree >= 50: # Высокая степень
            threshold = 0.6     # 60% нулей => бинарный
        elif self.degree >= 20: # Средняя степень
            threshold = 0.7     # 70% нулей => бинарный
        else:                   # Низкая степень
            threshold = 0.8     # 80% нулей → бинарный

        self.use_binary = self.zero_ratio >= threshold


    def __call__(self, X: Matrix):
        """
        Вычисление многочлена от матрицы

        Рассматриваем 2 типа полиномов: разреженные и плотные.
        И в зависимости от степени полинома выбираем подход:
        * бинарный
        * последовательный
        """

        if X.size != self.size:
            raise ValueError(f"Размер X ({X.size}) не совпадает с размером коэффициентов ({self.size})")

        if self.use_binary:
            return self.call_binary_sparse(X)
        else:
            return self.call_sequential(X)


    def call_sequential(self, X: Matrix):
        """
        Последовательное вычисление - оптимально для плотных полиномов.
        Вычисляет степени последовательно, переиспользуя результаты.
        """
        n = self.size
        zero_matrix = Matrix.zero(n)
        result = Matrix.zero(n)
        powered_X = Matrix.identity(n)  # X^0

        # Идём от младшего коэффициента к старшему
        for i in range(self.degree, -1, -1):
            coeff = self.coefficients[i]

            if coeff != zero_matrix:
                result = result + coeff.naive_mul(powered_X)

            # Подготавливаем следующую степень (если не последняя итерация)
            if i > 0:
                powered_X = powered_X.naive_mul(X)

        return result


    def call_binary_sparse(self, X: Matrix):
        """
        Бинарное вычисление - оптимально для разреженных полиномов
        Для каждой степени, матрица при которой не нулевая, вычисляет X^power бинарно.
        """
        n = self.size
        zero_matrix = Matrix.zero(n)
        result = Matrix.zero(n)

        for i, coeff in enumerate(self.coefficients):
            if coeff != zero_matrix:
                power = self.degree - i

                if power == 0:
                    X_power = Matrix.identity(n)
                elif power == 1:
                    X_power = X
                else:
                    X_power = X.quick_powering(power)

                result = result + coeff.naive_mul(X_power)

        return result





# TEMP TESTING + usage
from contextlib import contextmanager
import time
@contextmanager
def timer(name=""):
    """Измеряет время выполнения блока кода для промежуточного бенчмаркинга"""
    start = time.perf_counter()
    yield   # выполняется код внутри блока with
    duration = time.perf_counter() - start
    print(f"[{name}] Время: {duration} сек")

def create_test_matrix(n, max_num=10):
    """Создание тестовой матрицы (n x n) с рациональными числами"""
    import random

    arr = []
    for i in range(n):
        row = []
        for j in range(n):
            a = random.randint(-max_num, max_num)
            b = random.randint(1, max_num)

            sign = 1 if a < 0 else 0
            digits = [int(d) for d in str(abs(a))]
            numerator = Integer(sign, len(digits), digits)

            denom_digits = [int(d) for d in str(b)]
            denominator = Natural(1, denom_digits)

            row.append(Rational(numerator, denominator))
        arr.append(row)

    return Matrix(n, arr)



if __name__ == "__main__":

    # for size in [5, 6, 7]:
    #     A = create_test_matrix(size, 5)
    #
    #     with timer("q"):
    #         (A.quick_powering(50))
    #     with timer("H-C"):
    #         (A.Hamilton_Cayley_pow(50))
    #     print()

    # A = create_test_matrix(15, 5)
    # print(A)

    # with timer("D"):
    #     A.diag_and_pow_2x2(200)
    # with timer("A"):
    #     A.analytical_2x2_pow(200)
    # with timer("Q"):
    #     A.quick_powering(30)
    # with timer("H-C"):
    #     A.Hamilton_Cayley_pow(30)
    ...
