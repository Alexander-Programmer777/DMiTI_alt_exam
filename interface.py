import tkinter as tk
from tkinter import ttk, messagebox

from Matrixes.main import Matrix
from matrix_parser import parse_matrix_string, parse_matrix_polynomial
from parser import *
from P.Polynomial import Polynomial
from TRANS.TRANS_Q_P import TRANS_Q_P


class CalculatorSelector:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Выбор калькулятора")

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = int(screen_width * 0.3)
        window_height = int(screen_height * 0.5)

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.window.minsize(int(screen_width * 0.25), int(screen_height * 0.4))  # Минимальный размер
        self.window.resizable(True, True)

        self.create_widgets()

    def run(self):
        """Запускает приложение"""
        self.window.mainloop()

    def create_widgets(self):
        title_label = tk.Label(self.window, text="Выберите тип калькулятора",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)

        # Кнопки для выбора типа калькулятора
        button_style = {'font': ('Arial', 14), 'height': 2, 'width': 20}

        natural_btn = tk.Button(self.window, text="Натуральные числа",
                                command=self.open_natural_calculator, **button_style)
        natural_btn.pack(pady=5)

        integer_btn = tk.Button(self.window, text="Целые числа",
                                command=self.open_integer_calculator, **button_style)
        integer_btn.pack(pady=5)

        rational_btn = tk.Button(self.window, text="Рациональные числа",
                                 command=self.open_rational_calculator, **button_style)
        rational_btn.pack(pady=5)

        polynomial_btn = tk.Button(self.window, text="Полиномы",
                                   command=self.open_polynomial_calculator, **button_style)
        polynomial_btn.pack(pady=5)


        # Добавляем новые кнопки
        matrix_btn = tk.Button(self.window, text="Матрицы",
                               command=self.open_matrix_calculator, **button_style)
        matrix_btn.pack(pady=5)

        poly_matrix_btn = tk.Button(self.window, text="Полином от матрицы",
                                    command=self.open_polynomial_matrix_calculator, **button_style)
        poly_matrix_btn.pack(pady=5)


    def open_matrix_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = MatrixCalculator()
        calculator.window.geometry(geometry)
        calculator.run()

    def open_polynomial_matrix_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = PolynomialMatrixCalculator()
        calculator.window.geometry(geometry)
        calculator.run()

    def open_natural_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = Calculator("natural")
        calculator.window.geometry(geometry)
        calculator.run()

    def open_integer_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = Calculator("integer")
        calculator.window.geometry(geometry)
        calculator.run()

    def open_rational_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = Calculator("rational")
        calculator.window.geometry(geometry)
        calculator.run()

    def open_polynomial_calculator(self):
        geometry = self.window.geometry()
        self.window.destroy()
        calculator = Calculator("polynomial")
        calculator.window.geometry(geometry)
        calculator.run()


class Calculator:
    def __init__(self, calc_type):
        self.calc_type = calc_type
        self.window = tk.Tk()

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = int(screen_width * 0.22)
        window_height = int(screen_height * 0.51)

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.window.minsize(window_width, window_height)
        self.window.resizable(True, True)

        titles = {
            "natural": "Калькулятор натуральных чисел",
            "integer": "Калькулятор целых чисел",
            "rational": "Калькулятор рациональных чисел",
            "polynomial": "Калькулятор полиномов"
        }
        self.window.title(titles.get(calc_type, "Калькулятор"))

        self.expression = ""

        self.create_widgets()

    def create_widgets(self):
        # Создаем основной фрейм с правильным выравниванием
        main_frame = tk.Frame(self.window)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Центрируем содержимое по горизонтали
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.columnconfigure(4, weight=1)

        type_label = tk.Label(main_frame,
                              text=f"Тип: {self.get_calc_type_name()}",
                              font=('Arial', 10, 'bold'),
                              fg='blue')
        type_label.grid(row=0, column=0, columnspan=5, pady=(10, 5), sticky='ew')

        # Создаем фрейм для Text виджета и скроллбара
        entry_frame = tk.Frame(main_frame)
        entry_frame.grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky='ew')

        # ЗАМЕНА Entry на Text виджет с горизонтальным скроллбаром
        self.display = tk.Text(entry_frame, font=('Arial', 14), height=2, wrap='none',
                               padx=5, pady=5, bg='white', fg='black')
        self.display.pack(side='top', fill='x', expand=True)

        # ГОРИЗОНТАЛЬНЫЙ скроллбар для прокрутки в бок
        h_scrollbar = ttk.Scrollbar(entry_frame, orient='horizontal', command=self.display.xview)
        h_scrollbar.pack(side='bottom', fill='x')
        self.display.config(xscrollcommand=h_scrollbar.set)

        # БЛОКИРОВКА ВВОДА С КЛАВИАТУРЫ
        self.display.bind('<Key>', lambda e: 'break')  # Блокируем все клавиши
        self.display.bind('<Button-1>', lambda e: 'break')  # Блокируем выделение мышкой

        info_label = tk.Label(main_frame, text="Введите выражение", font=('Arial', 10))
        info_label.grid(row=2, column=0, columnspan=5, pady=(0, 10))

        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=5, sticky='nsew', pady=10)

        main_frame.rowconfigure(3, weight=1)

        for i in range(5):
            button_frame.columnconfigure(i, weight=1)
        for i in range(5):
            button_frame.rowconfigure(i, weight=1)

        buttons = [
            '7', '8', '9', '/', 'x²',
            '4', '5', '6', '*', 'x³',
            '1', '2', '3', '-', 'xⁿ',
            '0', 'x', '=', '+', 'C',
            '(', ')', '%', '//', '<—'
        ]

        row = 0
        col = 0

        for button in buttons:
            if button == '=':
                cmd = self.show_result
            elif button == 'C':
                cmd = self.clear
            elif button == '<—':
                cmd = self.backspace
            elif button in ['x²', 'x³', 'xⁿ']:
                cmd = lambda x=button: self.add_power(x)
            else:
                cmd = lambda x=button: self.add_to_expression(x)

            btn = tk.Button(
                button_frame,
                text=button,
                font=('Arial', 12),
                command=cmd
            )

            btn.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')

            col += 1
            if col > 4:
                col = 0
                row += 1

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.grid(row=4, column=0, columnspan=5, sticky='ew', pady=10)

        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        back_btn = tk.Button(
            bottom_frame,
            text="Назад к выбору",
            font=('Arial', 12),
            command=self.back_to_selector,
            height=2,
            bg='lightgreen'
        )
        back_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))

        calc_btn = tk.Button(
            bottom_frame,
            text="Вычислить",
            font=('Arial', 12),
            command=self.show_result,
            height=2,
            bg='lightblue'
        )
        calc_btn.grid(row=0, column=1, sticky='ew', padx=(5, 0))

        self.update_display()

    def get_calc_type_name(self):
        """Возвращает читаемое название типа калькулятора"""
        names = {
            "natural": "Натуральные числа",
            "integer": "Целые числа",
            "rational": "Рациональные числа",
            "polynomial": "Полиномы"
        }
        return names.get(self.calc_type, "Неизвестный тип")

    def add_to_expression(self, value):
        """Добавляет символ к выражению"""
        if self.calc_type in ["natural", "integer"] and len(value) == 1 and value[0] == '/':
            messagebox.showwarning("Ошибка", "В данном калькуляторе нельзя использовать операцию деления")
            return
        elif self.calc_type in ["rational", "polynomial"] and value[:2] == '//':
            messagebox.showwarning("Ошибка",
                                   "В данном калькуляторе нельзя использовать операцию целочисленного деления")
            return
        elif self.calc_type in ["rational"] and value[0] == '%':
            messagebox.showwarning("Ошибка", "В данном калькуляторе нельзя использовать операцию остатка от деления")
            return

        self.expression += str(value)
        self.update_display()

    def add_power(self, power_type):
        """Добавляет степень переменной"""
        if self.calc_type in ["natural", "integer", "rational"]:
            messagebox.showwarning("Ошибка", "В данном калькуляторе нельзя использовать переменную 'x'")
            return
        if power_type == 'x²':
            self.expression += 'x^2'
        elif power_type == 'x³':
            self.expression += 'x^3'
        elif power_type == 'xⁿ':
            self.expression += 'x^'
        self.update_display()

    def clear(self):
        """Очищает выражение"""
        self.expression = ""
        self.update_display()

    def backspace(self):
        """Очищает последний символ выражения"""
        self.expression = self.expression[:-1]
        self.update_display()

    def show_result(self):
        """Получаем результат"""
        if self.expression:
            try:
                # Вычисляем результат в зависимости от типа калькулятора
                result = self.process_expression(self.expression)
                # Выводим результат
                self.clear()
                self.expression = str(result)
                self.update_display()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка вычисления: {e}")
        else:
            messagebox.showwarning("Предупреждение", "Введите выражение")

    def update_display(self):
        """Обновляет отображение выражения"""
        self.display.config(state='normal')
        self.display.delete(1.0, tk.END)
        self.display.insert(1.0, self.expression)
        self.display.config(state='disabled')

        # Автоматически прокручиваем к концу
        self.display.see(tk.END)

    def process_expression(self, expr):
        """Обрабатывает выражение в зависимости от типа калькулятора"""
        # Здесь будет происходить передача строки с выражением
        # в соответствующий модуль обработки

        if self.calc_type == "natural":
            ans = eval_rpn_n(to_rpn(expr))
            return f"{ans.show()}"
        elif self.calc_type == "integer":
            ans = eval_rpn_z(to_rpn(expr))
            return f"{ans.show()}"
        elif self.calc_type == "rational":
            ans = eval_rpn_q(to_rpn(expr))
            return f"{ans.show()}"
        elif self.calc_type == "polynomial":
            ans = eval_rpn_p(to_rpn(expr))
            if type(ans) != Polynomial:
                ans = TRANS_Q_P(ans)
            return f"{ans.show()}"

        return 'answer'

    def back_to_selector(self):
        """Возврат к окну выбора калькулятора"""
        geometry = self.window.geometry()
        self.window.destroy()
        selector = CalculatorSelector()
        selector.window.geometry(geometry)
        selector.window.mainloop()

    def run(self):
        """Запускает приложение"""
        self.window.mainloop()


class MatrixCalculator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Калькулятор матриц")

        # Получаем размеры экрана
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Адаптивные размеры окна (50% экрана)
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.7)

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.window.minsize(int(screen_width * 0.4), int(screen_height * 0.6))

        # Настройка шрифтов
        self.large_font = ('Arial', 12)
        self.medium_font = ('Arial', 11)
        self.small_font = ('Arial', 10)
        self.matrix_font = ('Courier New', 11)  # Моноширинный для матриц

        self.create_widgets()

    def create_widgets(self):
        # Главный контейнер с прокруткой
        main_container = tk.Frame(self.window)
        main_container.pack(fill='both', expand=True)

        # Добавляем полосы прокрутки
        canvas = tk.Canvas(main_container)
        scrollbar_y = tk.Scrollbar(main_container, orient='vertical', command=canvas.yview)
        scrollbar_x = tk.Scrollbar(main_container, orient='horizontal', command=canvas.xview)

        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Упаковываем с прокруткой
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar_y.pack(side='right', fill='y')
        scrollbar_x.pack(side='bottom', fill='x')


        # Привязываем колесо мыши для прокрутки
        # Привязываем колесо мыши для прокрутки на моем чудесном macOS
        def _on_mousewheel(event):
            # Для macOS с Magic Mouse
            if hasattr(event, 'delta'):
                canvas.yview_scroll(int(-1 * (event.delta)), "units")
            # Для Linux (кнопки 4 и 5)
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        # Привязываем все возможные события прокрутки мультиплатформенно!
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows, macOS с обычной мышью
        canvas.bind_all("<Button-4>", _on_mousewheel)  # Linux
        canvas.bind_all("<Button-5>", _on_mousewheel)  # Linux

        # Особое событие для тачпадов macOS и Magic Mouse
        try:
            # Некоторые версии Tkinter поддерживают это событие
            canvas.bind_all("<TouchpadScroll>", _on_mousewheel)
        except:
            pass  # Игнорируем если событие не поддерживается


        # Теперь создаем виджеты внутри scrollable_frame
        main_frame = scrollable_frame

        # Заголовок
        tk.Label(main_frame, text="Калькулятор матриц",
                 font=('Arial', 16, 'bold')).pack(pady=20)

        # Инструкция
        instruction = """Введите матрицы в одном из форматов:
        1) [[1,2],[3,4]]
        2) 1 2; 3 4
        3) 1 2 3
           4 5 6
           7 8 9"""

        tk.Label(main_frame, text=instruction, font=self.small_font,
                 justify='left', bg='lightyellow', padx=10, pady=10).pack(fill='x', padx=20, pady=10)

        # Фрейм для матриц A и B
        matrices_frame = tk.Frame(main_frame)
        matrices_frame.pack(fill='x', padx=20, pady=10)

        # Матрица A
        matrix_a_frame = tk.LabelFrame(matrices_frame, text="Матрица A", font=self.medium_font)
        matrix_a_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Text с прокруткой для матрицы A
        a_text_frame = tk.Frame(matrix_a_frame)
        a_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.matrix_a_text = tk.Text(a_text_frame, height=8, width=30,
                                     font=self.matrix_font, wrap='none')
        self.matrix_a_text.pack(side='left', fill='both', expand=True)

        a_scroll_y = tk.Scrollbar(a_text_frame, command=self.matrix_a_text.yview)
        a_scroll_y.pack(side='right', fill='y')
        self.matrix_a_text.config(yscrollcommand=a_scroll_y.set)

        a_scroll_x = tk.Scrollbar(matrix_a_frame, orient='horizontal', command=self.matrix_a_text.xview)
        a_scroll_x.pack(side='bottom', fill='x')
        self.matrix_a_text.config(xscrollcommand=a_scroll_x.set)

        # Матрица B
        matrix_b_frame = tk.LabelFrame(matrices_frame, text="Матрица B", font=self.medium_font)
        matrix_b_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Text с прокруткой для матрицы B
        b_text_frame = tk.Frame(matrix_b_frame)
        b_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.matrix_b_text = tk.Text(b_text_frame, height=8, width=30,
                                     font=self.matrix_font, wrap='none')
        self.matrix_b_text.pack(side='left', fill='both', expand=True)

        b_scroll_y = tk.Scrollbar(b_text_frame, command=self.matrix_b_text.yview)
        b_scroll_y.pack(side='right', fill='y')
        self.matrix_b_text.config(yscrollcommand=b_scroll_y.set)

        b_scroll_x = tk.Scrollbar(matrix_b_frame, orient='horizontal', command=self.matrix_b_text.xview)
        b_scroll_x.pack(side='bottom', fill='x')
        self.matrix_b_text.config(xscrollcommand=b_scroll_x.set)

        # Настраиваем расширение колонок
        matrices_frame.columnconfigure(0, weight=1)
        matrices_frame.columnconfigure(1, weight=1)

        # Кнопки операций
        ops_frame = tk.LabelFrame(main_frame, text="Операции", font=self.medium_font)
        ops_frame.pack(fill='x', padx=20, pady=10)

        # Первый ряд кнопок
        row1_frame = tk.Frame(ops_frame)
        row1_frame.pack(fill='x', padx=10, pady=5)

        operations1 = [
            ('A + B', self.add_matrices, 'lightgreen'),
            ('A - B', self.subtract_matrices, 'lightcoral'),
            ('A × B', self.multiply_matrices, 'lightblue'),
            ('det(A)', self.determinant_a, 'yellow'),
        ]

        for i, (text, command, color) in enumerate(operations1):
            btn = tk.Button(row1_frame, text=text, command=command,
                            font=self.medium_font, bg=color, width=12, height=2)
            btn.pack(side='left', padx=5, pady=5, expand=True)

        # Второй ряд кнопок
        row2_frame = tk.Frame(ops_frame)
        row2_frame.pack(fill='x', padx=10, pady=5)

        operations2 = [
            ('A⁻¹', self.inverse_a, 'orange'),
            ('Aᵀ', self.transpose_a, 'lightcyan'),
            ('trace(A)', self.trace_a, 'violet'),
            ('A²', lambda: self.power_a(2), 'pink'),
        ]

        for i, (text, command, color) in enumerate(operations2):
            btn = tk.Button(row2_frame, text=text, command=command,
                            font=self.medium_font, bg=color, width=12, height=2)
            btn.pack(side='left', padx=5, pady=5, expand=True)

        # Третий ряд для специальных операций
        row3_frame = tk.Frame(ops_frame)
        row3_frame.pack(fill='x', padx=10, pady=5)

        # Кнопка для произвольной степени
        self.power_var = tk.StringVar(value="3")
        power_frame = tk.Frame(row3_frame)
        power_frame.pack(side='left', padx=5, pady=5, expand=True)

        tk.Button(power_frame, text="A^", command=self.power_custom,
                  font=self.medium_font, bg='lightyellow', width=3).pack(side='left')

        power_entry = tk.Entry(power_frame, textvariable=self.power_var,
                               width=5, font=self.medium_font, justify='center')
        power_entry.pack(side='left', padx=2)

        # Поле результата
        result_frame = tk.LabelFrame(main_frame, text="Результат", font=self.medium_font)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Text с двойной прокруткой для результата
        result_text_frame = tk.Frame(result_frame)
        result_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_text_frame, height=12,
                                   font=self.matrix_font, wrap='none', state='disabled')
        self.result_text.pack(side='left', fill='both', expand=True)

        result_scroll_y = tk.Scrollbar(result_text_frame, command=self.result_text.yview)
        result_scroll_y.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=result_scroll_y.set)

        result_scroll_x = tk.Scrollbar(result_frame, orient='horizontal', command=self.result_text.xview)
        result_scroll_x.pack(side='bottom', fill='x')
        self.result_text.config(xscrollcommand=result_scroll_x.set)

        # Кнопки управления
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill='x', padx=20, pady=10)

        control_buttons = [
            ('Очистить всё', self.clear_all, 'lightcoral'),
            ('Сохранить в A', self.save_to_a, 'lightblue'),
            ('Сохранить в B', self.save_to_b, 'lightblue'),
            ('Назад', self.back_to_selector, 'lightgreen'),
        ]

        for text, command, color in control_buttons:
            btn = tk.Button(control_frame, text=text, command=command,
                            font=self.medium_font, bg=color, height=2)
            btn.pack(side='left', padx=5, pady=5, expand=True)

        self.window.update_idletasks()  # Обновляем геометрию

        # Динамически вычисляем отступы
        def update_padding():
            width = self.window.winfo_width()

            # Формула: (общая_ширина - ширина_контента) / 2
            content_width = 600
            padding = max(20, (width - content_width) // 2)

            # Применяем ко всем фреймам
            matrices_frame.pack_configure(padx=padding)
            ops_frame.pack_configure(padx=padding)
            result_frame.pack_configure(padx=padding)
            control_frame.pack_configure(padx=padding)

        # Вызываем сразу и при изменении размера
        update_padding()
        self.window.bind('<Configure>', lambda e: update_padding())


    def parse_matrix_input(self, text_widget):
        """Парсит матрицу из текстового поля с обработкой ошибок"""
        text = text_widget.get("1.0", tk.END).strip()
        if not text:
            raise ValueError("Введите матрицу")

        try:
            return parse_matrix_string(text)
        except Exception as e:
            # Показываем подробную ошибку
            error_msg = f"Ошибка парсинга матрицы:\n{str(e)}\n\n"
            raise ValueError(error_msg)

    def show_result(self, result):
        """Показывает результат в текстовом поле"""
        self.result_text.config(state='normal')
        self.result_text.delete("1.0", tk.END)

        # Добавляем сам результат
        if isinstance(result, Matrix):
            matrix_str = str(result)
            self.result_text.insert(tk.END, matrix_str)
        else:
            self.result_text.insert(tk.END, str(result))

        self.result_text.config(state='disabled')
        # Прокручиваем в начало
        self.result_text.see("1.0")

    def add_matrices(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            b = self.parse_matrix_input(self.matrix_b_text)
            result = a + b
            self.show_result(result)
        except Exception as e:
            messagebox.showerror("Ошибка сложения", str(e))

    def subtract_matrices(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            b = self.parse_matrix_input(self.matrix_b_text)
            result = a - b
            self.show_result(result)
        except Exception as e:
            messagebox.showerror("Ошибка вычитания", str(e))

    def multiply_matrices(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            b = self.parse_matrix_input(self.matrix_b_text)
            result = a @ b
            self.show_result(result)
        except Exception as e:
            messagebox.showerror("Ошибка умножения", str(e))

    def determinant_a(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a.det()
            self.show_result(f"det(A) = {result}")
        except Exception as e:
            messagebox.showerror("Ошибка вычисления определителя", str(e))

    def inverse_a(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a.inv()
            self.show_result(result)
        except Exception as e:
            messagebox.showerror("Ошибка вычисления обратной матрицы", str(e))

    def transpose_a(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a.transpose()
            self.show_result(result)
        except Exception as e:
            messagebox.showerror("Ошибка транспонирования", str(e))

    def power_a(self, power):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a ** power
            self.show_result(result)
        except Exception as e:
            messagebox.showerror(f"Ошибка возведения в степень {power}", str(e))

    def trace_a(self):
        try:
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a.trace()
            self.show_result(f"trace(A) = {result}")
        except Exception as e:
            messagebox.showerror("Ошибка вычисления следа", str(e))

    def clear_all(self):
        self.matrix_a_text.delete("1.0", tk.END)
        self.matrix_b_text.delete("1.0", tk.END)
        self.result_text.config(state='normal')
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state='disabled')

    def save_to_a(self):
        try:
            result_text = self.result_text.get("1.0", tk.END).strip()
            if result_text:
                # Извлекаем только матрицу из результата (игнорируем заголовки)
                lines = result_text.split('\n')
                matrix_lines = []
                for line in lines:
                    # Ищем строки с числами или скобками (сами элементы матрицы)
                    if any(c.isdigit() or c in '[]/-' for c in line.strip()):
                        matrix_lines.append(line.strip())

                if matrix_lines:
                    matrix_str = '\n'.join(matrix_lines)
                    self.matrix_a_text.delete("1.0", tk.END)
                    self.matrix_a_text.insert("1.0", matrix_str)
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def save_to_b(self):
        try:
            result_text = self.result_text.get("1.0", tk.END).strip()
            if result_text:
                # Извлекаем только матрицу из результата (игнорируем заголовки)
                lines = result_text.split('\n')
                matrix_lines = []
                for line in lines:
                    # Ищем строки с числами или скобками (сами элементы матрицы)
                    if any(c.isdigit() or c in '[]/-' for c in line.strip()):
                        matrix_lines.append(line.strip())

                if matrix_lines:
                    matrix_str = '\n'.join(matrix_lines)
                    self.matrix_b_text.delete("1.0", tk.END)
                    self.matrix_b_text.insert("1.0", matrix_str)
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def back_to_selector(self):
        geometry = self.window.geometry()
        self.window.destroy()
        selector = CalculatorSelector()
        selector.window.geometry(geometry)
        selector.window.mainloop()

    def power_custom(self):
        try:
            power_str = self.power_var.get().strip()
            if not power_str:
                raise ValueError("Введите степень")

            power = int(power_str)
            a = self.parse_matrix_input(self.matrix_a_text)
            result = a ** power
            self.show_result(result)
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректная степень: {e}")
        except Exception as e:
            messagebox.showerror("Ошибка возведения в степень", str(e))

    def run(self):
        self.window.mainloop()



class PolynomialMatrixCalculator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Полином от матрицы")

        # Адаптивные размеры
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.8)

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.window.minsize(int(screen_width * 0.6), int(screen_height * 0.7))

        # Шрифты
        self.title_font = ('Arial', 14, 'bold')
        self.normal_font = ('Arial', 11)
        self.matrix_font = ('Courier New', 11)

        self.create_widgets()

    def create_widgets(self):
        # Главный контейнер с прокруткой
        main_container = tk.Frame(self.window)
        main_container.pack(fill='both', expand=True)

        canvas = tk.Canvas(main_container)
        scrollbar_y = tk.Scrollbar(main_container, orient='vertical', command=canvas.yview)
        scrollbar_x = tk.Scrollbar(main_container, orient='horizontal', command=canvas.xview)

        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar_y.pack(side='right', fill='y')
        scrollbar_x.pack(side='bottom', fill='x')

        # для прокрутки
        def _on_mousewheel(event):
            # Для macOS с Magic Mouse
            if hasattr(event, 'delta'):
                canvas.yview_scroll(int(-1 * (event.delta)), "units")
            # Для Linux (кнопки 4 и 5)
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        # Привязываем все возможные события прокрутки
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        try:
            canvas.bind_all("<TouchpadScroll>", _on_mousewheel)
        except:
            pass

        main_frame = scrollable_frame

        # Заголовок
        tk.Label(main_frame, text="Вычисление полинома от матрицы",
                 font=self.title_font).pack(pady=20)

        # Инструкция
        instruction = """Формат полинома: Aₙ*Xⁿ + Aₙ₋₁*Xⁿ⁻¹ + ... + A₁*X + A₀
        где Aᵢ - матричные коэффициенты, X - матрица-аргумент

        Примеры:
        1) [[1,0],[0,1]]*X^2 + [[0,1],[1,0]]*X + [[2,0],[0,2]]
        2) [[1,2],[3,4]]*X^3 + [[5,6],[7,8]]*X + [[9,10],[11,12]]

        Можно использовать пробелы для удобства чтения."""

        tk.Label(main_frame, text=instruction, font=('Arial', 10),
                 justify='left', bg='lightyellow', padx=10, pady=10,
                 wraplength=600).pack(fill='x', padx=20, pady=10)

        # Ввод полинома
        poly_frame = tk.LabelFrame(main_frame, text="Полином P(X)", font=self.normal_font)
        poly_frame.pack(fill='x', padx=20, pady=10)

        # Text с прокруткой для полинома
        poly_text_frame = tk.Frame(poly_frame)
        poly_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.poly_text = tk.Text(poly_text_frame, height=6, font=self.matrix_font, wrap='none')
        self.poly_text.pack(side='left', fill='both', expand=True)

        poly_scroll_y = tk.Scrollbar(poly_text_frame, command=self.poly_text.yview)
        poly_scroll_y.pack(side='right', fill='y')
        self.poly_text.config(yscrollcommand=poly_scroll_y.set)

        poly_scroll_x = tk.Scrollbar(poly_frame, orient='horizontal', command=self.poly_text.xview)
        poly_scroll_x.pack(side='bottom', fill='x')
        self.poly_text.config(xscrollcommand=poly_scroll_x.set)

        # Ввод матрицы X
        matrix_frame = tk.LabelFrame(main_frame, text="Матрица X", font=self.normal_font)
        matrix_frame.pack(fill='x', padx=20, pady=10)

        # Text с прокруткой для матрицы
        matrix_text_frame = tk.Frame(matrix_frame)
        matrix_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.matrix_text = tk.Text(matrix_text_frame, height=6, font=self.matrix_font, wrap='none')
        self.matrix_text.pack(side='left', fill='both', expand=True)

        matrix_scroll_y = tk.Scrollbar(matrix_text_frame, command=self.matrix_text.yview)
        matrix_scroll_y.pack(side='right', fill='y')
        self.matrix_text.config(yscrollcommand=matrix_scroll_y.set)

        matrix_scroll_x = tk.Scrollbar(matrix_frame, orient='horizontal', command=self.matrix_text.xview)
        matrix_scroll_x.pack(side='bottom', fill='x')
        self.matrix_text.config(xscrollcommand=matrix_scroll_x.set)

        # Кнопки примеров
        example_frame = tk.Frame(main_frame)
        example_frame.pack(pady=10)

        examples = [
            ("Пример 1", self.load_example1),
            ("Пример 2", self.load_example2),
            ("Пример 3", self.load_example3),
        ]

        for text, command in examples:
            btn = tk.Button(example_frame, text=text, command=command,
                            font=self.normal_font, bg='lightblue')
            btn.pack(side='left', padx=5)

        # Кнопка вычисления
        tk.Button(main_frame, text="Вычислить P(X)", command=self.calculate,
                  font=('Arial', 12, 'bold'), bg='green', fg='white',
                  height=2, width=20).pack(pady=20)

        # Результат
        result_frame = tk.LabelFrame(main_frame, text="Результат P(X)", font=self.normal_font)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Text с двойной прокруткой для результата
        result_text_frame = tk.Frame(result_frame)
        result_text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_text_frame, height=15, font=self.matrix_font,
                                   wrap='none', state='disabled')
        self.result_text.pack(side='left', fill='both', expand=True)

        result_scroll_y = tk.Scrollbar(result_text_frame, command=self.result_text.yview)
        result_scroll_y.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=result_scroll_y.set)

        result_scroll_x = tk.Scrollbar(result_frame, orient='horizontal', command=self.result_text.xview)
        result_scroll_x.pack(side='bottom', fill='x')
        self.result_text.config(xscrollcommand=result_scroll_x.set)

        # Информация о полиноме
        self.info_label = tk.Label(main_frame, text="", font=('Arial', 10), fg='blue')
        self.info_label.pack(pady=5)

        # Кнопки управления
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=20)

        tk.Button(control_frame, text="Очистить", command=self.clear,
                  font=self.normal_font, bg='lightcoral', width=15).pack(side='left', padx=10)
        tk.Button(control_frame, text="Назад", command=self.back_to_selector,
                  font=self.normal_font, bg='lightgreen', width=15).pack(side='left', padx=10)

        self.window.update_idletasks()  # Обновляем геометрию

        # Центрирование
        def update_padding():
            width = self.window.winfo_width()

            # Формула: (общая_ширина - ширина_контента) / 2
            # Для полиномов контент шире - ставим больше
            content_width = 600
            padding = max(20, (width - content_width) // 2)

            # Применяем ко всем фреймам
            poly_frame.pack_configure(padx=padding)
            matrix_frame.pack_configure(padx=padding)
            example_frame.pack_configure(padx=padding)
            result_frame.pack_configure(padx=padding)
            control_frame.pack_configure(padx=padding)

        # Вызываем сразу и при изменении размера
        update_padding()
        self.window.bind('<Configure>', lambda e: update_padding())


    def load_example1(self):
        """Полином 2 степени: X² + 2X + 3E"""
        self.poly_text.delete("1.0", tk.END)
        self.poly_text.insert("1.0", "[[1,0],[0,1]]*X^2 + [[2,0],[0,2]]*X + [[3,0],[0,3]]")

        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert("1.0", "[[1,2],[3,4]]")

    def load_example2(self):
        """Полином 3 степени"""
        self.poly_text.delete("1.0", tk.END)
        self.poly_text.insert("1.0", "[[1,1],[0,1]]*X^3 + [[2,0],[0,2]]*X^2 + [[0,1],[1,0]]*X + [[3,0],[0,3]]")

        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert("1.0", "[[2,1],[1,2]]")

    def load_example3(self):
        """Линейный полином: 2X + E"""
        self.poly_text.delete("1.0", tk.END)
        self.poly_text.insert("1.0", "[[2,0],[0,2]]*X + [[1,0],[0,1]]")

        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert("1.0", "[[1,0],[0,1]]")

    def calculate(self):
        try:
            # Парсим полином
            poly_str = self.poly_text.get("1.0", tk.END).strip()
            if not poly_str:
                raise ValueError("Введите полином")

            # Используем парсер полиномов
            matrix_poly = parse_matrix_polynomial(poly_str)

            # Показываем информацию о полиноме
            self.info_label.config(
                text=f"Полином степени {matrix_poly.degree}, размер матриц: {matrix_poly.size}x{matrix_poly.size}"
            )

            # Парсим матрицу X
            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            if not matrix_str:
                raise ValueError("Введите матрицу X")

            X = parse_matrix_string(matrix_str)

            # Проверяем совместимость размеров
            if X.size != matrix_poly.size:
                raise ValueError(
                    f"Несовместимые размеры: полином работает с {matrix_poly.size}x{matrix_poly.size}, "
                    f"а матрица X имеет размер {X.size}x{X.size}"
                )

            # Вычисляем полином от матрицы
            result = matrix_poly(X)

            # Показываем результат
            self.show_result(result, matrix_poly, X)

        except Exception as e:
            messagebox.showerror("Ошибка вычисления", str(e))
            import traceback
            traceback.print_exc()

    def show_result(self, result, poly, X):
        """Показывает результат с детальной информацией"""
        self.result_text.config(state='normal')
        self.result_text.delete("1.0", tk.END)

        # Форматируем вывод
        output = "=" * 60 + "\n"
        output += "РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ ПОЛИНОМА ОТ МАТРИЦЫ\n"
        output += "=" * 60 + "\n\n"

        # Результат
        output += "\nРезультат P(X):\n"
        output += str(result)

        self.result_text.insert("1.0", output)
        self.result_text.config(state='disabled')
        self.result_text.see("1.0")

    def clear(self):
        self.poly_text.delete("1.0", tk.END)
        self.matrix_text.delete("1.0", tk.END)
        self.result_text.config(state='normal')
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state='disabled')
        self.info_label.config(text="")

    def back_to_selector(self):
        geometry = self.window.geometry()
        self.window.destroy()
        selector = CalculatorSelector()
        selector.window.geometry(geometry)
        selector.window.mainloop()

    def run(self):
        self.window.mainloop()


# Запуск приложения начинается с выбора калькулятора
if __name__ == "__main__":
     CalculatorSelector().run()
