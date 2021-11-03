"""
Лабораторная работа по созданию и визуализации работы модели персептрона для распознавания 2 типов изображений (сердце и елочка). Изображения для тренировки и распознавания рисуются пользователем на форме.
"""

from tkinter import *
import numpy as np


class Paint(Frame):
    N = 20
    x = w = np.zeros((N, N), dtype=int)
    res = 0

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.color = "black"
        self.brush_size = 5
        self.setUI()

    def clear(self):
        self.canv.delete("all")
        for line in range(0, 200, 10):  # range(start, stop, step)
            self.canv.create_line([(line, 0), (line, 200)], fill='black', tags='grid_line_w')

        for line in range(0, 200, 10):
            self.canv.create_line([(0, line), (200, line)], fill='black', tags='grid_line_h')

        # обнуление массива при очистке изображения
        self.x = np.zeros((self.N, self.N), dtype=int)

    def draw(self, event):
        # отрисовка сетки для рисования
        for line in range(0, 200, 10):  # range(start, stop, step)
            print(self.x)
            self.canv.create_line([(line, 0), (line, 200)], fill='black', tags='grid_line_w')

        for line in range(0, 200, 10):
            self.canv.create_line([(0, line), (200, line)], fill='black', tags='grid_line_h')

        # заливка квадратиков при рисовании
        self.canv.create_rectangle(event.x // 10 * 10 - 10, event.y // 10 * 10 - 10, event.x // 10 * 10,
                                   event.y // 10 * 10, fill=self.color, outline=self.color)
        if event.x // 10 < 20 and event.y // 10 < 20:
            self.x[event.x // 10][event.y // 10] = 1

    def delta(self, w, x):

        # шаг 0 - проинициализировать весовые коэффициенты небольшими случайными значениями
        # выполняется один раз, если файл с весами пуст
        with open("WData.txt", "r") as fobj:
            try:
                # если весовые коэффициенты сохранены в файл предыдущими итерациями, то считываем их
                numbers = np.fromfile(fobj, count=self.N * self.N, sep=';').reshape(self.N, self.N)
                self.w = np.array(numbers)

            except ValueError:
                # если весовые коэффициенты не заданы в файле - инициализируем случайными значениями
                numbers = self.setRndW(self.N)
                # и записываем в файл
                self.save2File(numbers)
            print(self.w)

        # шаг 1 - подать на вход один из обучающих векторов и вычислить выход Y с помощью функции активации
        resLearn = self.activate(np.sum(np.asarray(w * x)))

        # шаг 2 - если выход правильный, перейти на шаг 4. Иначе вычислить ошибку
        if (self.res != resLearn):
            delt = self.res - resLearn
            # шаг 3 - скорректировать весовые коэффициенты
            for i in range(self.N):
                for j in range(self.N):
                    self.w[i][j] = self.w[i][j] + 0.5 * delt * self.x[i][j]
            # и сохраняем в файл
            self.save2File(self.w)

        # шаг 4 - шаги 1-3 повторяются для всех обучающих векторов при каждом нажатии на кнопку "Обучение"

        if round(resLearn) == 1:
            # label для отображения результата
            lbl = Label(self, text="Елочка")
            lbl.grid(column=0, row=0)
        else:
            lbl = Label(self, text="Сердце")
            lbl.grid(column=0, row=0)
        # print(resLearn)
        return resLearn

    def setRndW(self, N):
        return np.random.uniform(-0.3, 0.3, (N, N))

    def setTrue(self):
        self.res = 1

    def setFalse(self):
        self.res = 0

    def activate(self, s):
        return 1 / (1 + np.e ** (-0.3 * s))

    def save2File(self, numbers):
        file_handler = open("WData.txt", "w")
        for i in range(self.N):
            for j in range(self.N):
                file_handler.write(str(numbers[i][j]) + ";")
        file_handler.close()

    def check(self):
        # считываем значение весов из файла
        # необходимо для возможности использования кнопки "Проверить" сразу после запуска программы)
        with open("WData.txt", "r") as fobj:
            try:
                # если весовые коэффициенты сохранены в файл предыдущими итерациями, то считываем их
                numbers = np.fromfile(fobj, count=self.N * self.N, sep=';').reshape(self.N, self.N)
                self.w = np.array(numbers)

            except ValueError:
                # если весовые коэффициенты не заданы в файле - инициализируем случайными значениями
                numbers = self.setRndW(self.N)
                # и записываем в файл
                self.save2File(numbers)

        resLearn = self.activate(np.sum(np.asarray(self.w * self.x)))

        if round(resLearn) == 1:
            # label для отображения результата
            lbl = Label(self, text="Елочка")
            lbl.grid(column=0, row=0)
        else:
            lbl = Label(self, text="Сердце")
            lbl.grid(column=0, row=0)
        # print(resLearn)

    def setUI(self):

        width = 200
        height = 200

        self.parent.title("Pythonicway PyPaint")  # Устанавливаем название окна
        self.pack(fill=BOTH, expand=1)  # Размещаем активные элементы на родительском окне

        # Даем третьему столбцу возможность растягиваться, благодаря чему кнопки не будут разъезжаться при ресайзе
        self.columnconfigure(3, weight=1)

        # Создаем поле для рисования, устанавливаем белый фон
        self.canv = Canvas(self, bg="white")
        # Прикрепляем канвас методом grid. Он будет находится в 3м ряду, первой колонке, и будет занимать 7 колонок,
        # задаем отступы по X и Y в 3 пикселей, и заставляем растягиваться при растягивании всего окна
        self.canv.grid(row=2, column=0, columnspan=7,
                       padx=3, pady=3,
                       sticky=E + W + S + N)
        # Привязываем обработчик к канвасу
        # <B1-Motion> означает "при движении зажатой левой кнопки мыши" вызывать функцию draw
        self.canv.bind("<B1-Motion>", self.draw)

        # задание кнопок и их расположение
        clear_btn = Button(self, text="Очистить", width=8,
                           command=lambda: self.clear())
        clear_btn.grid(row=0, column=1, sticky=W)

        learn_btn = Button(self, text="Обучение", width=8,
                           command=lambda: self.delta(self.w, self.x))
        learn_btn.grid(row=0, column=2, sticky=W)

        true_btn = Button(self, text="Елочка", width=8,
                          command=lambda: self.setTrue())
        true_btn.grid(row=1, column=1, sticky=W)

        false_btn = Button(self, text="Сердце", width=8,
                           command=lambda: self.setFalse())
        false_btn.grid(row=1, column=2, sticky=W)

        check_btn = Button(self, text="Проверить", width=8,
                           command=lambda: self.check())
        check_btn.grid(row=1, column=0, sticky=W)

        # отрисовка сетки для изображения
        for line in range(0, width, 10):  # range(start, stop, step)
            self.canv.create_line([(line, 0), (line, height)], fill='black', tags='grid_line_w')

        for line in range(0, height, 10):
            self.canv.create_line([(0, line), (width, line)], fill='black', tags='grid_line_h')



def main():
    root = Tk()
    root.geometry("220x270+100+100")
    app = Paint(root)
    root.mainloop()


if __name__ == '__main__':
    main()
