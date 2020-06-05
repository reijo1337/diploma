import os
from datetime import timedelta
from time import time
from tkinter import Tk, Label, Button, Frame
from tkinter.ttk import Combobox
from tkinter import filedialog as fd
from keras.models import load_model
from PIL import Image, ImageTk
import cv2
from PIL import ImageTk
import  numpy as np

from classifier.capsnet import CapsNet
from preprocessing.detector import skin_detector


class MyFirstGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Демонстрация метода распознавания жестовых символов")
        self.root.geometry("1000x600")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

        self.build_left()
        self.build_center()
        self.build_right()

    def build_left(self):
        self.left = Frame(self.root, borderwidth=1, relief="solid")
        self.left.pack(side="left", expand=True, fill="both")

        self.left_choose_frame = Frame(self.left)
        self.left_choose_frame.pack(side="top")
        self.left_choose_label = Label(self.left_choose_frame, text="Способ получения изображения:")
        self.left_choose_label.pack(side="left")
        self.left_choose_combo = Combobox(self.left_choose_frame,
                                          values=['Загрусить с диска', 'Снять с web-камеры'])
        self.left_choose_combo.current(0)
        self.left_choose_combo.bind("<<ComboboxSelected>>", self.switch_input)
        self.left_choose_combo.pack(side="left")

        self.left_input_frame = Frame(self.left)
        self.left_input_frame.pack(side="top", expand=True, fill="both")
        self.file_input()

    def build_center(self):
        self.center = Frame(self.root, borderwidth=1, relief="solid")
        self.center.pack(side="left", expand=True, fill="both")
        self.center_original_label = Label(self.center, text="Исходное изображение:")
        self.center_original_label.pack()
        self.center_original_image = Label(self.center)
        self.center_original_image.pack()
        self.center_processed_label = Label(self.center, text="Предобработанное изображение:")
        self.center_processed_label.pack()
        self.center_processed_image = Label(self.center)
        self.center_processed_image.pack()

    def build_right(self):
        self.right = Frame(root, borderwidth=1, relief="solid")
        self.right.pack(side="left", expand=True, fill="both")
        self.choose_model = Button(self.right, text="Загрузить модель с диска", command=self.load_model)
        self.choose_model.pack()
        self.result = Label(self.right)
        self.result.pack()

    def load_model(self):
        filename = fd.askopenfilename(filetypes=(("Файлы моделей keras", "*.h5"),
                                                 ("Все файлы", "*.*")))
        print(filename)

        self.model = CapsNet((60, 60, 1), 10, 3)
        self.model.load_weights(filename)
        filename, _ = os.path.split(filename)
        filename, _ = os.path.split(filename)
        filename, _ = os.path.split(filename)
        filename = os.path.join(filename, "classes")
        
        if self.processed_image is not None:
            self.process_image()

    def process_image(self):
        w, h = self.input_image.shape[0], self.input_image.shape[1]
        scale = w / 250.
        w /= scale
        w = int(w)
        h /= scale
        h = int(h)

        start = time()
        self.processed_image = skin_detector(self.input_image)
        data = cv2.resize(self.processed_image, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
        data = data.reshape(1, 60, 60, 1)
        y_pred = self.model.predict(data, batch_size=100)
        b = np.zeros_like(y_pred)
        b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        y_pred = np.around(b).astype(np.int)
        end = time() - start

        orig = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGBA)
        orig = cv2.resize(orig, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        orig = Image.fromarray(orig)
        origtk = ImageTk.PhotoImage(image=orig)
        self.center_original_image.imgtk = origtk
        self.center_original_image.configure(image=origtk)

        proc = cv2.resize(self.processed_image, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        proc = Image.fromarray(proc)
        proctk = ImageTk.PhotoImage(image=proc)
        self.center_processed_image.imgtk = proctk
        self.center_processed_image.configure(image=proctk)

        result_test = f'Вывод классификатора: {y_pred[0]}\nКласс жеста: 9\nВремя обработки: {str(timedelta(seconds=end))}'
        self.result.text = result_test
        self.result.configure(text=result_test)
        
    def open_image(self):
        filename = fd.askopenfilename(filetypes=(("Файлы PNG", "*.png"),
                                                 ("Файлы JPEG", "*.jpg"),
                                                 ("Файлы BMP", "*.bmp"),
                                                 ("Все файлы", "*.*")))
        self.input_image = cv2.imread(filename)
        self.process_image()

    def file_input(self):
        for widget in self.left_input_frame.winfo_children():
            widget.destroy()
        self.dont_show_webcam = True
        file_button = Button(self.left_input_frame, text="Открыть файл", command=self.open_image)
        file_button.pack()

    def webcam_input(self):
        for widget in self.left_input_frame.winfo_children():
            widget.destroy()
        self.dont_show_webcam = False
        self.webcam = Label(self.left_input_frame)
        self.webcam.pack(expand=True, fill="both")
        webcam_button = Button(self.left_input_frame, text="Сделать снимок", command=self.snap)
        webcam_button.pack()
        self.show_frame()

    def switch_input(self, event):
        if self.left_choose_combo.get() == 'Загрусить с диска':
            self.file_input()
        else:
            self.webcam_input()

    def show_frame(self):
        if self.dont_show_webcam:
            return
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.frame = frame[0:300, 0:200]
        cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.webcam.imgtk = imgtk
        self.webcam.configure(image=imgtk)
        self.webcam.after(10, self.show_frame)

    def snap(self):
        self.input_image = self.frame
        self.process_image()


if __name__ == "__main__":
    root = Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()