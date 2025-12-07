import tkinter as tk
from tkinter import ttk, messagebox
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


# Запуск приложения начинается с выбора калькулятора
if __name__ == "__main__":
    selector = CalculatorSelector().run()
