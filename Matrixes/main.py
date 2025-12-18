"""
Финальная версия класса работы с невырожденными квадратными матрицами над Q
Содержит реализации и подходы, которые являются самыми оптимальными согласно замерам производительности,
а также переопределенные операторы +, -, * (на константу), @ (матричное произведение) и тд
"""

from N.Natural import Natural
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

        self.size = size  # размерность квадратной матрицы, int
        self.arr = arr  # массив строк матрицы, строка - массив рациональных чисел

        # При создании вычисляем статистику про матрицу, чтобы использовать ее при
        # умножении, сложении, возведении в степень (дает выигрыш во времени в этих случаях + имеет ранний выход из циклов проверки)
        self.is_zero_mat = self._is_zero_matrix()
        self.is_one_mat = self._is_identity()


    __slots__ = ('size', 'arr', 'is_zero_mat', 'is_one_mat')


    def __add__(self, other):
        """
        Сложение двух матриц (одного размера)
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")

        if self.is_zero_mat:  # Небольшая оптимизация: проверки на нулевые матрицы, O(1) так как кэшировано
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

        if other.is_zero_mat:  # A - 0 = A
            return self

        n = self.size
        res = [
            [self.arr[i][j] - other.arr[i][j] for j in range(n)]  # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)

    def __neg__(self):
        """Унарный минус: -A"""
        return self * Rational(Integer(1, 1, [1]), Natural(1, [1]))  # умножение на -1, O(n^2)


    def __mul__(self, other):   # A * const, const from Q
        """
        Умножение матрицы на константу
        """
        n = self.size
        if not isinstance(other, Rational):
            raise TypeError(f"Константа должна быть Rational, получено {type(other)}")

        if self.is_zero_mat:  # если матрица нулевая, то ее не умножаем на константу, результат она же
            return self

        if other == ZERO_RATIONAL:
            return Matrix.zero(n)

        if other == ONE_RATIONAL:
            return self

        res = [
            [self.arr[i][j] * other for j in range(n)]  # list comprehension, O(n^2)
            for i in range(n)
        ]

        return Matrix(n, res)


    def __rmul__(self, other):  # const * A
        """Умножение константы на матрицу слева"""
        return self.__mul__(other)  # используем уже реализованное умножение на константу, но справа


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


    def __matmul__(self, other):
        """
        Умножение двух матриц по определению -> O(n^3)
        * самый оптимальный из трех рассмотренных
        """
        if self.size != other.size:
            raise ValueError(f"Матрицы разного размера: {self.size} и {other.size}")
        n = self.size

        # Небольшие оптимизации
        if self.is_one_mat: return other  # умножение на E
        if other.is_one_mat: return self

        if self.is_zero_mat: return Matrix.zero(n)  # умножение на 0-matrix
        if other.is_zero_mat: return Matrix.zero(n)

        res = [[ZERO_RATIONAL for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (self.arr[i][k] == ZERO_RATIONAL) or (other.arr[k][j] == ZERO_RATIONAL):  # оптимизация: проверка на ноль вместо алгоритмов умножения и сложения дробей
                        continue
                    res[i][j] += self.arr[i][k] * other.arr[k][j]

        return Matrix(n, res)
    

    def det(self):
        """
        Определитель методом Гаусса через приведение к треугольному (ступенчатому) виду
        O(n^3). Самое оптимальное из реализованных методов для n > 3
        """
        n = self.size
        if n <= 3:
            return self._det_formula()

        # Создаём копию матрицы, чтобы не портить оригинал
        matrix = [row[:] for row in self.arr]  # в конце алгоритма она станет ступенчатой

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
    
    def _det_formula(self):
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
            # треугольники
            a, b, c = self.arr[0]
            d, e, f = self.arr[1]
            g, h, i = self.arr[2]
            return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h

        return None


    def inv(self):
        """
        Поиск обратной матрицы по методу Гаусса
        [A | E] ~ [E | A^-1], E - единичная

        Сложность: O(n^3), (лучше, чем метод союзной матрицы)
    
        Алгоритм:
        1. Создаём [A | E]
        2. Прямой ход: приводим A к верхнетреугольному виду
        3. Обратный ход: приводим A к единичной матрице
        4. Получаем [E | A^-1]

        Оптимальнее объединить прямой и обратный ход, и за 1 проход сводить к единичной
        """
        n = self.size

        # Проверка определителя (вырожденные матрицы необратимы)
        det = self.det()
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
                    if factor != ZERO_RATIONAL:  # чтобы не выполнять лишнее бессмысленное действие
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

    
    def __pow__(self, power: int):
        """
        Финальный алгоритм возведения матрицы в челочисленную степень со всеми провеками и оптимальными подходами
        """
        if not isinstance(power, int):
            raise TypeError(f"Степень должна быть целым числом, получено {type(power)}")
        n = self.size

        # Проверка на возведение уже диагональной матрицы в степень
        if self._is_diag():
            return self._pow_diag(power)
        
        if n == 2:
            # В зависимости от коэффициента 'c' делаем попытку возведения через диагонализацию или через аналитические формулы.
            # В обоих методах в случаях иррациональности и других ошибках реализован откат на надежный _quick_powering.
            if self.arr[1][0] == ZERO_RATIONAL:     # c == 0
                return self._diag_and_pow_2x2(power)
            
            return self._analytical_2x2_pow(power)
        
        return self._quick_powering(power)


    def _quick_powering(self, power: int):
        """
        Возведение матрицы в степень power
        через алгоритм быстрого возведения в степень -> O(log |power|) умножений
        Эффективнее обычного + легкость реализации + универсален

        Здесь есть проверки на граничные случаи, кроме диагональности (для этого отдельный метод)
        """

        if not isinstance(power, int):
            raise TypeError(f"Степень должна быть целым числом, получено {type(power)}")
        n = self.size

        # Проверки граничных случаев, упрощающих вычисления
        if self.is_one_mat:  # E^k = E ∀ k
            return self

        if power == 0:
            return Matrix.identity(n)

        if self.is_zero_mat and power > 0:
            return self

        if power == 1:
            return self

        if power == -1:
            return self.inv()

        if power < 0:  # Отрицательные степени: A^(-k) = (A^(-1))^k
            try:
                base = self.inv()
            except ValueError as e:
                raise ValueError(f"Нельзя возвести в отрицательную степень: {e}")

            power = -power  # => power > 0 -> general case

        else:
            base = self

        # Основной алгоритм бинарного возведения в степень: base ^ power
        res = Matrix.identity(n)

        while power > 0:
            if power & 1:  # Если степень нечётная
                res = res @ base
            base = base @ base  # Возводим в квадрат
            power >>= 1  # Делим степень на 2

        return res


    def _is_identity(self) -> bool:
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


    def _is_zero_matrix(self):
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


    def _is_diag(self):
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


    def _pow_diag(self, power: int):
        """
        Алгоритм возведения в степень диагональной матрицы
        O(n) возведений рациональных чисел в степень (тоже реализовано бинарно в Rational), копирование - O(n^2)
        Метод вызывается только после проверки на диагональность.
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
        X(lmbd) = lmbd^2 - trA * lmbd + detA    -    Характеристическое уравнение
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
            print("Иррациональность в корне из дискриминанта")
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

        # Мини оптимизация: квадрат целого числа не заканчивается на 2, 3, 7, 8
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
        Диагонализация матрицы 2x2 над Q если возможно

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


    def _diag_and_pow_2x2(self, power):
        """
        Интеграция с предыдущим методом diagonalize_2x2
        После удавшейся диагонализации можно оптимизировать возведение в степень
        A ~ D => A = P * D * P^-1 =>
        A^k = (P * D * P^-1)^k = P * D^k * P^-1

        Свойство (почему можно так диагонализировать):
        A * P = D * P, где P - матрица в стобцах которой айгенвектора, D - диагональная матрица из айгензначений
        A * [v1, v2] = diag(lmbd1, lmbd2) * [v1, v2] и если это раскрыть, то получится как раз определение айген векторов и чисел:
        A * v_i = lmbd_i * v_i

        При провале диагонализации fallback на бинарное возведение в степень

        На практике порой реально попадаются случаи без иррациональности и оно действительно считает быстрее
        К примеру, 9.33 сек vs 25.56 сек для возведения матрицы 2х2 в 500 степень
        Главный минус в том, что несложно посчитать можно только для 2х2 матриц, так как аналитический поиск собственных значений и векторов
        очень усложняется с увеличением размера матрицы.
        Так же ограничением является рассмотрение только рациональных значений, которые в 2x2 хоть как-то можно посчитать аналитически
        
        * Здесь, в финальной реализации, применяется для матриц, у которых c=0, так как это дает выигрыш по времени согласно исследованию
        """

        result = self.diagonalize_2x2()
        if result is None:
            print("Диагонализация не удалась. Вычисление через _quick_powering")
            return self._quick_powering(power)

        P, D = result
        P_inv = P.inv()

        D_powered = D._pow_diag(power)

        return P @ D_powered @ P_inv


    def _analytical_2x2_pow(self, power):
        """
        Возведение матрицы 2x2 в степень через аналитические формулы.
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

        Сравнению аналитческого метода, диагонализации и бинарного возведения в степень в случае иррациональности посвещены отдельные графики в исследовании.

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
            inv = self.inv()
            return inv._analytical_2x2_pow(-power)

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
            print("fallback from analytical to quick")     # справочный иинформационный вывод
            return self._quick_powering(power)

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

        self.degree = degree  # степень многочлена
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
        zero_ratio = zero_count / total_terms if total_terms > 0 else 0

        # Правило: чем выше степень, тем ниже порог для бинарного метода
        if self.degree >= 100:  # Очень высокая степень
            threshold = 0.5     # 50% нулей => бинарный
        elif self.degree >= 50: # Высокая степень
            threshold = 0.6     # 60% нулей => бинарный
        elif self.degree >= 20: # Средняя степень
            threshold = 0.7     # 70% нулей => бинарный
        else:                   # Низкая степень
            threshold = 0.8     # 80% нулей → бинарный

        self.use_binary = zero_ratio >= threshold


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
            return self._call_binary_sparse(X)
        else:
            return self._call_sequential(X)


    def _call_sequential(self, X: Matrix):
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
                result = result + coeff @ powered_X

            # Подготавливаем следующую степень (если не последняя итерация)
            if i > 0:
                powered_X = powered_X @ X

        return result


    def _call_binary_sparse(self, X: Matrix):
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
                    X_power = X ** power    # ** = __pow__ - оптимизированное возведение в степень

                result = result + coeff @ X_power

        return result