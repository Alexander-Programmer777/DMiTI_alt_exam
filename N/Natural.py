class Natural:
    def __init__(self, n, A):
        self.A = A  # [] массив из int   123 -> [1, 2, 3]
        self.len = n  # int len(A)-1
        # Автоматически убираем ведущие нули при создании
        while len(self.A) > 1 and self.A[0] == 0:
            self.A.pop(0)
        # Корректируем длину после удаления нулей
        self.len = len(self.A) - 1

        # Если после удаления нулей массив пуст, создаем [0]

        if len(self.A) == 0:
            self.A = [0]
            self.len = 0

    def COM_NN_D(self, other):
        """
        Сделал: Соколовский Артём
        Сравнение двух натуральных чисел (self и other).

        Возвращает:
            1  — если self > other
            0  — если self == other
            -1 — если self < other
        """
        A = self.A[:]
        B = other.A[:]

        # Убираем ведущие нули
        while len(A) > 1 and A[0] == 0:
            A.pop(0)
        while len(B) > 1 and B[0] == 0:
            B.pop(0)

        # Сравнение по длине
        if len(A) > len(B):
            return 1
        elif len(A) < len(B):
            return -1

        # Поразрядное сравнение
        for da, db in zip(A, B):
            if da > db:
                return 1
            elif da < db:
                return -1

        return 0

    def NZER_N_B(self):
        """
        Богданов Никита Константинович
        Проверка на ноль натурального числа
        """
        # Число равно нулю, если оно состоит из одной цифры и эта цифра 0

        A = self.A[:]
        while len(A) > 1 and A[0] == 0:
            A.pop(0)

        return not (self.len == 0 and self.A[0] == 0)

    def ADD_1N_N(self):
        """
        Выполнил: Сурин Максим
        Добавление 1 к натуральному числу
        """

        """ 
        Запись числа справа налево;
        прибавление единицы к разряду единиц  
        """
        rev_num = self.A.copy()[::-1]
        rev_num[0] += 1

        """ Перенос единиицы при переполнении разряда """
        for i in range(self.len + 1):
            if rev_num[i] == 10:
                rev_num[i] = 0
                if i < self.len:
                    rev_num[i + 1] += 1
                else:
                    rev_num.append(1)

        return Natural(len(rev_num) - 1, rev_num[::-1])

    def __add__(self, other):
        """
        Сделал: Соколовский Артём
        Сложение двух натуральных чисел: self + other.
        """
        A = self.A[::-1]
        B = other.A[::-1]
        res = []
        carry = 0

        for i in range(max(len(A), len(B))):
            da = A[i] if i < len(A) else 0
            db = B[i] if i < len(B) else 0
            s = da + db + carry
            res.append(s % 10)
            carry = s // 10

        if carry:
            res.append(carry)

        res.reverse()
        return Natural(len(res) - 1, res)

    def __sub__(self, other):
        """
        Сделал: Соколовский Артём
        Вычитание натуральных чисел: self - other (при self >= other).
        """
        if self.COM_NN_D(other) == -1:
            raise ValueError("SUB_NN_N: self < other")

        A = self.A[::-1]
        B = other.A[::-1]
        res = []
        borrow = 0

        for i in range(len(A)):
            da = A[i]
            db = B[i] if i < len(B) else 0
            diff = da - db - borrow
            if diff < 0:
                diff += 10
                borrow = 1
            else:
                borrow = 0
            res.append(diff)

        while len(res) > 1 and res[-1] == 0:
            res.pop()

        res.reverse()
        return Natural(len(res) - 1, res)

    def MUL_ND_N(self, int):
        """
        Выполнил: Сурин Максим
        Умножение натурального числа на цифру
        """

        """ Запись числа справа налево """
        rev_num = self.A.copy()[::-1]

        """ Домножение каждого разряда на цифру """
        for i in range(self.len + 1):
            rev_num[i] *= int

        """ Перенос при переполнении разряда """
        for i in range(self.len + 1):
            if rev_num[i] > 9:
                """ 
                В текущем разряде сохраняем единицы,
                в следующий переносим десятки
                """
                if i < self.len:
                    rev_num[i + 1] += rev_num[i] // 10
                else:
                    rev_num.append(rev_num[i] // 10)
                rev_num[i] %= 10

        return Natural(len(rev_num) - 1, rev_num[::-1])

    def MUL_Nk_N(self, k):
        """
        Богданов Никита Константинович
        Умножение натурального числа на 10^k
        """

        if not isinstance(k, Natural):
            # Если k - int, преобразуем в Natural
            if k == 0:
                k_natural = Natural(0, [0])
            elif k < 10:
                k_natural = Natural(0, [k])
            else:
                k_digits = [int(d) for d in str(k)]
                k_natural = Natural(len(k_digits) - 1, k_digits)
        else:
            k_natural = k

        # Если число равно нулю, возвращаем ноль
        if not self.NZER_N_B():
            return Natural(0, [0])

        # Получаем значение k как целое число
        k_value = 0
        multiplier = 1
        for i in range(k_natural.len, -1, -1):
            k_value += k_natural.A[i] * multiplier
            multiplier *= 10

        new_A = self.A + [0] * k_value
        return Natural(len(new_A) - 1, new_A)

    def __mul__(self, other):
        """
        Богданов Никита Константинович
        Умножение натуральных чисел
        """
        if not isinstance(other, Natural):
            raise TypeError("The multipliers must be Natural")

        # Если одно из чисел равно нулю, возвращаем ноль
        if not self.NZER_N_B() or not other.NZER_N_B():
            return Natural(0, [0])

        # Используем упрощенный алгоритм умножения
        result = Natural(0, [0])

        # Проходим по всем цифрам второго числа справа налево
        for i in range(len(other.A)):
            # Индекс цифры (от младшей к старшей)
            digit_index = len(other.A) - 1 - i
            digit = other.A[digit_index]

            # Умножаем self на цифру
            temp_product = self.MUL_ND_N(digit)

            # Сдвигаем на i позиций влево (умножаем на 10^i)
            if i > 0:
                # Создаем Natural для сдвига
                shift_natural = Natural(0, [i]) if i < 10 else Natural(1, [i // 10, i % 10])
                temp_product = temp_product.MUL_Nk_N(shift_natural)

            # Складываем с результатом
            result = result + temp_product

        return result

    def SUB_NDN_N(self, int, other):
        """
        Выполнил: Сурин Максим
        Вычитание из натурального другого натурального, умноженного на цифру для случая с неотрицательным результатом
        """

        """ Домножение второго числа на цифру """
        other_num = other.MUL_ND_N(int)

        """ В случае отрицательного результата возвращаем ноль """
        if self.COM_NN_D(other_num) == -1:
            return Natural(0, [0])

        return self - other_num

    def DIV_NN_Dk(self, other) -> (int, int):
        """
        Сделал: Захаренко Александр
        Вычисление первой цифры деления большего натурального
        на меньшее, домноженное на 10^k,
        где k - номер позиции этой цифры (номер считается с нуля)
        """

        if self.COM_NN_D(other) == -1:
            return 0, 0

        k = self.len - other.len
        if k < 0:
            return 0, 0

        # ИСПРАВЛЕНО: создаем Natural для k
        if k < 10:
            k_natural = Natural(0, [k])
        else:
            k_digits = [int(d) for d in str(k)]
            k_natural = Natural(len(k_digits) - 1, k_digits)

        other_shifted = other.MUL_Nk_N(k_natural)

        # Проверяем, что other_shifted не больше self
        if self.COM_NN_D(other_shifted) == -1:
            k -= 1
            if k < 0:
                return 0, 0
            if k < 10:
                k_natural = Natural(0, [k])
            else:
                k_digits = [int(d) for d in str(k)]
                k_natural = Natural(len(k_digits) - 1, k_digits)
            other_shifted = other.MUL_Nk_N(k_natural)
        #
        for digit in range(9, 0, -1):
            temp = other_shifted.MUL_ND_N(digit)
            if self.COM_NN_D(temp) != -1:
                return digit, k

        return 0, 0

    # вместо DIV_NN_N (переопределяем Целочисленное деление //)
    def __floordiv__(self, other):
        """
        Сделал: Захаренко Александр
        Неполное частное от деления первого натурального числа
        на второе с остатком (делитель отличен от нуля)
        """

        # Проверка деления на ноль
        if all(x == 0 for x in other.A):
            raise ZeroDivisionError("Division by zero")

        # Если делимое меньше делителя, возвращаем 0
        if self.COM_NN_D(other) == -1:
            return Natural(0, [0])

        # Определяем максимальную длину результата
        max_length = self.len - other.len + 1
        result_digits = [0] * max_length  # массив для цифр результата

        current = Natural(self.len, self.A.copy())  # текущий остаток

        # Пока текущий остаток >= other
        while current.COM_NN_D(other) != -1:
            # Получаем очередную цифру и её позицию
            digit, k = current.DIV_NN_Dk(other)

            # Если цифра 0, значит деление завершено
            if digit == 0:
                break

            # Записываем цифру в результат на соответствующую позицию
            result_digits[k] = digit

            # Вычитаем: current = current - digit * other * 10^k
            other_shifted = other.MUL_Nk_N(k)  # other * 10^k
            current = current.SUB_NDN_N(digit, other_shifted)

            # Если остаток стал нулевым, завершаем
            if all(x == 0 for x in current.A):
                break

        # Убираем ведущие нули
        while len(result_digits) > 1 and result_digits[-1] == 0:
            result_digits.pop()

        # Разворачиваем массив (т.к. у нас старшие разряды в конце)
        result_digits.reverse()

        return Natural(len(result_digits), result_digits)


    def __mod__(self, other):
        """
        Сделал: Захаренко Александр
        Остаток от деления первого натурального числа
        на второе натуральное (делитель отличен от нуля)
        """
        # Проверка деления на ноль
        if not other.NZER_N_B():
            raise ZeroDivisionError("Division by zero")

        # Если делимое меньше делителя, возвращаем само делимое
        if self.COM_NN_D(other) == -1:
            return Natural(self.len, self.A.copy())

        # Вычисляем частное
        quotient = self // other

        # Вычисляем произведение other * quotient
        product = other * quotient

        # Вычисляем остаток
        remainder = self - product

        return remainder

    def GCF_NN_N(self, other):
        """
        Сделал: Чумаков Никита Ярославович
        НОД (наибольший общий делитель) двух натуральных чисел.
        """
        # Создаём копии, чтобы не испортить исходные
        A = Natural(self.len, self.A[:])
        B = Natural(other.len, other.A[:])

        while B.NZER_N_B():  # пока B != 0
            # Проверяем, какое больше
            cmp = A.COM_NN_D(B)

            if cmp == -1:  # A < B -> поменяем их местами
                A, B = B, A

            # Теперь A >= B, можно брать остаток
            R = A % B
            A, B = B, R

        return A

    def LCM_NN_N(self, other):
        """
        Сделала: Имховик Наталья
        Нахождение НОК натуральных чисел
        Используем НОК(a, b) = (a * b) / НОД(a, b)
        Возвращает натуральное
        """
        gcd = self.GCF_NN_N(other)
        return self * other // gcd

    def show(self):
        s="".join(list(map(str, self.A)))
        return s