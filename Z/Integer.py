from N.Natural import Natural

class Integer:
    def __init__(self, s, n, A):
        self.s = s  # int знак числа (1 — минус, 0 — плюс)
        self.len = n  # int len(A)-1
        self.A = A  # [] массив из int   123 -> [1, 2, 3], -123 -> [1, 2, 3]

    def ABS_Z_Z(self):
        """
        Сделала: Имховик Наталья
        Определение абсолютной величины числа
        Возвращает целое
        """
        # Устанавливаем знак +
        return Integer(0, self.len, self.A[:])

    def SGN_Z_D(self):
        """
        Сделал: Чумаков Никита Ярославович
        Определение знака целого числа:
        -1 — положительное
         0 — равно нулю
         1 — отрицательное
        """
        # Проверим, что число не ноль
        if all(d == 0 for d in self.A):  # Если массив нулей -> z = 0
            return 0
        # Если знак отрицательный (s == 1)
        elif self.s == 1:
            return 1
        # Иначе положительное
        else:
            return -1

    def MUL_ZM_Z(self):
        """
        Сделал: Захаренко Александр
        Умножение целого на (-1)
        """

        # Проверям, не пытаемся ли заменить знак у нуля
        if self.A == [0]: return self  # для нуля возвращаем его же, не меняя знак

        # Иначе, число не ноль
        new_s = 0 if self.s == 1 else 1  # меняем знак числа
        return Integer(new_s, self.len, self.A)  # формируем новое целое число с противоположным знаком

    def __sub__(self, other):
        """
        Выполнил: Сурин Максим
        Вычитание целых чисел: self - other
        """

        # Если вычитаем ноль - возвращаем self
        if other.SGN_Z_D() == 0:
            return self

        # Если из нуля вычитаем число - возвращаем -other
        if self.SGN_Z_D() == 0:
            return other.MUL_ZM_Z()

        # Получаем модули чисел
        abs_self = self.ABS_Z_Z()
        abs_other = other.ABS_Z_Z()

        # Создаем натуральные числа для операций
        n1 = Natural(abs_self.len, abs_self.A[:])
        n2 = Natural(abs_other.len, abs_other.A[:])

        cmp_result = n1.COM_NN_D(n2)

        # Случай 1: оба положительные (a - b)
        if self.s == 0 and other.s == 0:
            if cmp_result >= 0:  # a >= b
                result = n1 - n2
                return Integer(0, result.len, result.A)
            else:  # a < b
                result = n2 - n1
                return Integer(1, result.len, result.A)

        # Случай 2: положительное - отрицательное (a - (-b) = a + b)
        if self.s == 0 and other.s == 1:
            result = n1 + n2
            return Integer(0, result.len, result.A)

        # Случай 3: отрицательное - положительное (-a - b = -(a + b))
        if self.s == 1 and other.s == 0:
            result = n1 + n2
            return Integer(1, result.len, result.A)

        # Случай 4: отрицательное - отрицательное (-a - (-b) = -a + b = b - a)
        if self.s == 1 and other.s == 1:
            if cmp_result == 0:  # a == b
                return Integer(0, 0, [0])
            elif cmp_result < 0:  # a < b → b - a > 0
                result = n2 - n1
                return Integer(0, result.len, result.A)
            else:  # a > b → b - a < 0
                result = n1 - n2
                return Integer(1, result.len, result.A)

    def __mul__(self, other):
        """
        Сделал: Соколовский Артём
        Умножение целых чисел: self * other.
        """

        sign = 1 if self.s != other.s else 0

        if self.SGN_Z_D() == 0 or other.SGN_Z_D() == 0:
            return Integer(0, 0, [0])

        # Используем ABS_Z_Z для получения модулей
        abs_self = self.ABS_Z_Z()
        abs_other = other.ABS_Z_Z()

        n1 = Natural(abs_self.len, abs_self.A[:])
        n2 = Natural(abs_other.len, abs_other.A[:])
        mul_result = n1 * n2

        return Integer(sign, mul_result.len, mul_result.A)

    def __floordiv__(self, other):
        """
        Сделал: Соколовский Артём
        Деление целых чисел (self / other).
        """

        if other.SGN_Z_D() == 0:
            raise ZeroDivisionError("Деление на ноль в целых числах")

        result_sign = 1 if self.s != other.s else 0

        # Используем ABS_Z_Z для получения модулей вместо ABS_Z_N
        abs_self = self.ABS_Z_Z()
        abs_other = other.ABS_Z_Z()

        # Создаем натуральные числа из модулей
        n1 = Natural(abs_self.len, abs_self.A[:])
        n2 = Natural(abs_other.len, abs_other.A[:])

        if n1.COM_NN_D(n2) == -1:
            return Integer(0, 0, [0])

        quotient = n1 // n2
        return Integer(result_sign, quotient.len, quotient.A)

    def __mod__(self, other):
        """
        Богданов Никита Константинович
        Остаток от деления целого числа self на целое число other
        """
        if other.SGN_Z_D() == 0:
            raise ValueError('Нельзя делить на ноль.')

        # Вместо // используем прямой вызов __truediv__
        quotient = self//other

        # Вычисляем произведение делителя и частного
        product = other * quotient

        # Вычисляем остаток
        remainder = self - product

        return remainder

    def __add__(self, other):
        """
        Выполнил: Сурин Максим
        Сложение целых чисел
        """

        # Если первое число - ноль, возвращаем второе
        if self.SGN_Z_D() == 0:
            return other

        # Если второе число - ноль, возвращаем первое
        if other.SGN_Z_D() == 0:
            return self

        # Получаем модули чисел
        abs_self = self.ABS_Z_Z()
        abs_other = other.ABS_Z_Z()

        # Создаем натуральные числа из модулей
        n1 = Natural(abs_self.len, abs_self.A[:])
        n2 = Natural(abs_other.len, abs_other.A[:])

        # Если числа одного знака - складываем модули и устанавливаем общий знак
        if self.s == other.s:
            sum_of_mods = n1 + n2
            return Integer(self.s, sum_of_mods.len, sum_of_mods.A)

        # Если числа разных знаков - вычитаем из большего меньшее, берём знак большего
        if self.s != other.s:
            cmp_result = n1.COM_NN_D(n2)

            if cmp_result >= 0:  # |self| >= |other|
                sub_of_mods = n1 - n2
                # Знак берем от первого числа
                return Integer(self.s, sub_of_mods.len, sub_of_mods.A)
            else:  # |self| < |other|
                sub_of_mods = n2 - n1
                # Знак берем от второго числа
                return Integer(other.s, sub_of_mods.len, sub_of_mods.A)

    def show(self):
        s=""
        if self.s == 1:
            s="-"
        s=s+"".join(list(map(str, self.A)))
        return s