from Q.Rational import Rational
from Z.Integer import Integer
from N.Natural import Natural
from Matrixes.main import Matrix, MatrixPolynomial


def parse_rational_from_str(s: str) -> Rational:
    """
    Парсит строку в Rational
    Форматы: "5", "2/3", "-4/7", "-5"
    """
    s = s.strip()

    if '/' in s:
        num_str, den_str = s.split('/')

        # Обработка отрицательных дробей
        num_str = num_str.strip()
        den_str = den_str.strip()

        # Определяем знак числителя
        is_negative = num_str.startswith('-')
        if is_negative:
            num_str = num_str[1:]  # убираем минус

        # Преобразуем строки в массивы цифр
        num_digits = [int(d) for d in num_str]
        den_digits = [int(d) for d in den_str]

        # Создаем Integer с правильным знаком
        num_sign = 1 if is_negative else 0
        num = Integer(num_sign, len(num_digits) - 1, num_digits)

        # Знаменатель всегда натуральное (положительное)
        den = Natural(len(den_digits) - 1, den_digits)

        return Rational(num, den)
    else:
        # Целое число
        s = s.strip()
        is_negative = s.startswith('-')
        if is_negative:
            s = s[1:]  # убираем минус

        digits = [int(d) for d in s]

        num_sign = 1 if is_negative else 0
        num = Integer(num_sign, len(digits) - 1, digits)
        den = Natural(1, [1])  # знаменатель = 1

        return Rational(num, den)


def parse_matrix_string(matrix_str: str) -> Matrix:
    """
    Парсит строковое представление матрицы
    Поддерживаемые форматы:
    1) [a] или просто a для 1x1
    2) [[a,b],[c,d]]
    3) a b; c d
    4) a b
       c d
    """
    matrix_str = matrix_str.strip()

    if not matrix_str:
        raise ValueError("Пустой ввод матрицы")

    # ===== СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ МАТРИЦЫ 1x1 =====
    # Убираем только внешние скобки для проверки
    test_str = matrix_str.strip()

    # Убираем возможные внешние скобки
    if test_str.startswith('[') and test_str.endswith(']'):
        test_str = test_str[1:-1].strip()
        # Если были двойные скобки [[...]], убираем еще один слой
        if test_str.startswith('[') and test_str.endswith(']'):
            test_str = test_str[1:-1].strip()

    # Проверяем, может это просто число или дробь
    # Сначала проверим, нет ли в строке запятых, точек с запятой или переносов
    has_complex_format = False
    if (',' in matrix_str and '],[' not in matrix_str) or ';' in matrix_str or '\n' in matrix_str:
        has_complex_format = True

    # Если нет сложного форматирования и test_str не пустой
    if not has_complex_format and test_str:
        try:
            # Пробуем распарсить как число (уже без скобок)
            rational = parse_rational_from_str(test_str)
            # Создаем матрицу 1x1
            return Matrix(1, [[rational]])
        except:
            # Если не получилось - продолжаем обычный парсинг
            pass

    # ===== ОБЫЧНЫЙ ПАРСИНГ =====

    # Разделяем по переносам строк
    lines = [line.strip() for line in matrix_str.split('\n') if line.strip()]

    # Если есть несколько строк после split по \n - это формат a b\nc d
    if len(lines) > 1:
        rows = []
        for line in lines:
            # Убираем скобки если есть
            line = line.replace('[', '').replace(']', '').strip()
            if not line:
                continue

            # Разделяем элементы: сначала пробелы, потом запятые
            elements = []
            # Пробуем пробелы
            if ' ' in line:
                elements = [elem.strip() for elem in line.split() if elem.strip()]
            # Если не вышло, пробуем запятые
            if not elements and ',' in line:
                elements = [elem.strip() for elem in line.split(',') if elem.strip()]
            # Если вообще нет разделителей - это один элемент
            if not elements and line:
                elements = [line]

            if elements:
                row = [parse_rational_from_str(elem) for elem in elements]
                rows.append(row)

        if not rows:
            raise ValueError("Не удалось распознать матрицу")

        n = len(rows)
        # Проверяем, что все строки имеют одинаковую длину
        first_len = len(rows[0]) if rows else 0
        for i, row in enumerate(rows):
            if len(row) != first_len:
                raise ValueError(f"Строка {i + 1} имеет {len(row)} элементов, ожидается {first_len}")

        # Проверяем квадратность
        if first_len != n:
            raise ValueError(f"Матрица не квадратная: {n} строк по {first_len} элементов")

        return Matrix(n, rows)


    # Если вся матрица в одной строке
    single_line = matrix_str

    # Формат [[a,b],[c,d]]
    if single_line.startswith('[') and '],[' in single_line:
        if single_line.startswith('[[') and single_line.endswith(']]'):
            content = single_line[2:-2]
            rows_str = content.split('],[')
        else:
            raise ValueError("Неверный формат матрицы вида [[a,b],[c,d]].")

    # Формат a b; c d
    elif ';' in single_line:
        rows_str = single_line.split(';')

    else:
        raise ValueError(
            "Неправильный формат матрицы.\n\n"
            "Используйте один из форматов:\n"
            "1) [[1,2],[3,4]]\n"
            "2) 1 2; 3 4\n"
            "3) 1 2\\n3 4 (каждая строка матрицы на новой строке)"
        )

    # Парсим строки для форматов с явным разделением строк
    rows = []
    for row_str in rows_str:
        row_str = row_str.strip()
        if not row_str:
            continue

        # Разделяем элементы строки
        elements = []
        # Сначала пробелы
        if ' ' in row_str:
            elements = [elem.strip() for elem in row_str.split() if elem.strip()]
        # Потом запятые
        if not elements and ',' in row_str:
            elements = [elem.strip() for elem in row_str.split(',') if elem.strip()]

        if elements:
            row = [parse_rational_from_str(elem) for elem in elements]
            rows.append(row)

    if not rows:
        raise ValueError("Не удалось распознать матрицу")

    n = len(rows)
    # Проверяем квадратность
    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(f"Строка {i + 1} имеет {len(row)} элементов, ожидается {n}")

    return Matrix(n, rows)



def parse_matrix_polynomial(poly_str: str) -> MatrixPolynomial:
    """
    Парсит полином вида: A*X^n + B*X^(n-1) + ... + C
    где A, B, C - матрицы
    """
    poly_str = poly_str.strip()
    if not poly_str:
        raise ValueError("Пустой полином")

    terms = []

    # Разделяем на слагаемые
    temp_terms = []
    current = ""
    bracket_depth = 0

    for char in poly_str:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1

        if char == '+' and bracket_depth == 0:
            if current.strip():
                temp_terms.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        temp_terms.append(current.strip())

    # Если не нашли плюсов, значит один член
    if not temp_terms:
        temp_terms = [poly_str]

    # Парсим каждый член
    for term in temp_terms:
        term = term.strip()
        if not term:
            continue

        # Находим степень X
        if 'X^' in term:
            parts = term.split('X^')
            if len(parts) != 2:
                raise ValueError(f"Неверный формат члена: {term}")

            matrix_str = parts[0].strip()
            # Убираем знак умножения если есть
            if matrix_str.endswith('*'):
                matrix_str = matrix_str[:-1].strip()

            power = int(parts[1].strip())
        elif '*X' in term:
            parts = term.split('*X')
            if len(parts) != 2:
                raise ValueError(f"Неверный формат члена: {term}")

            matrix_str = parts[0].strip()
            power = 1
        else:
            # Свободный член (X^0)
            matrix_str = term.strip()
            power = 0

        # Парсим матрицу
        matrix = parse_matrix_string(matrix_str)
        terms.append((power, matrix))

    # Сортируем по степени (от старшей к младшей)
    terms.sort(key=lambda x: x[0], reverse=True)

    # Создаем список коэффициентов
    max_power = terms[0][0]
    coefficients = [Matrix.zero(terms[0][1].size) for _ in range(max_power + 1)]

    for power, matrix in terms:
        coefficients[max_power - power] = matrix

    return MatrixPolynomial(max_power, coefficients)