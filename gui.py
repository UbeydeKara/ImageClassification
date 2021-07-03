from tkinter import filedialog
import cv2
from tkinter import *
import tkinter as tk
from PIL import ImageTk
from keras.preprocessing import image
import model as Model

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.wm_title("Image Classification")

        # ------ Image Capture ------------------------------------------------------------

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.interval = 30

        # ------ Creating elements ------------------------------------------------------------

        # Panels
        left_panel = PanedWindow()
        left_panel.grid(row="0", column="0", sticky="N")
        right_panel = PanedWindow()
        right_panel.grid(row="0", column="1", sticky="N")

        # ------ Widgets - Left Panel ------------------------------------------------------------

        self.info_list = Listbox(left_panel, width=50, height=18, font=("Arial", 10))
        scrollbar = Scrollbar(left_panel)
        self.info_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_list.yview)

        title1_lbl = tk.Label(left_panel, text="Train", font=("Helvetica", 24))
        train_btn = tk.Button(left_panel, text="Load Train Set",
                                   command=lambda: self.load_click(), padx=10, pady=10)

        self.count_lbl = tk.Label(left_panel)
        self.train_lbl = tk.Label(left_panel, wraplength=400)
        self.test_lbl = tk.Label(left_panel, wraplength=400)
        self.start_btn = tk.Button(left_panel, command=lambda: self.train_click(),
                                   text="Start Training", padx=10, pady=10, state='disabled')

        # Grid structure
        title1_lbl.grid(row=0, pady=5, padx=50)
        train_btn.grid(row=1, pady=5, padx=50, sticky="W")

        self.count_lbl.grid(row=2, pady=2, padx=50, sticky="W")
        self.train_lbl.grid(row=3, pady=2, padx=50, sticky="W")
        self.test_lbl.grid(row=4, pady=2, padx=50, sticky="W")
        self.start_btn.grid(row=5, pady=5, padx=50, sticky="W")

        self.info_list.grid(row=6, pady=10, padx=50)
        scrollbar.grid(row=6, pady=10, padx=50, sticky="nse")

        # ------ Widgets - Right Panel ------------------------------------------------------------

        title2_lbl = tk.Label(right_panel, text="Test", font=("Helvetica", 24))
        self.test_btn = tk.Button(right_panel, text="Select Image", command=lambda: self.img_view(),
                             padx=10, pady=10, state='normal')
        self.cam_btn = tk.Button(right_panel, text="Select From Camera", command=lambda: self.img_capture(),
                             padx=10, pady=10, state='normal')
        self.canvas = Canvas(right_panel, width=300, height=300, bg="white")
        self.predict_lbl = tk.Label(right_panel, font=("Arial", 12))

        # Grid structure
        title2_lbl.grid(row=0, pady=5, padx=50)
        self.test_btn.grid(row=1, pady=5, padx=50, sticky="W")
        self.cam_btn.grid(row=1, pady=5, padx=50, sticky="E")
        self.canvas.grid(row=2, pady=5, padx=50)
        self.predict_lbl.grid(row=3, pady=5, padx=50, sticky="W")

    def center_window(self, width, height):
        # get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def load_click(self):
        img_dir = filedialog.askdirectory()

        if img_dir:
            model.create_ds(img_dir)

            train_str = ""
            test_str = ""
            i = 0

            for class_name in model.classes:
                train_str = "{}{} ({}) ".format(train_str, class_name,
                                                 str(model.train_num.get(i)))
                test_str = "{}{} ({}) ".format(test_str, class_name,
                                                str(model.test_num.get(i)))
                i = i + 1

            self.count_lbl.config(text="{} class found".format(str(model.train_gen.num_classes)))
            self.train_lbl.config(text="Train={}".format(train_str))
            self.test_lbl.config(text="Test={}".format(test_str))
            self.start_btn.config(state='normal')

    def train_click(self):
        self.interval = -1
        self.info_list.insert(END, "[...]")
        self.info_list.update()

        for i in range(self.info_list.size(), self.info_list.size() + 1):
            model.train_model()
            self.info_list.insert(i - 1,
                                  "Epoch = " + str(i - 1) + " Train = {:.0%}".format(model.info.history['accuracy'][0]) +
                                  ", Loss = {:.0%}".format(model.info.history['loss'][0]) +
                                  ", Test = {:.0%}".format(model.info.history['val_accuracy'][0]))
            self.info_list.see(END)
            self.info_list.update()

        self.info_list.delete(END)

        self.cam_btn.config(state='normal')
        self.test_btn.config(state='normal')
        model.save_weights()

    def img_view(self):
        self.interval = -1
        img_name = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg'), ('Image Files', '*.jpeg'),
                                                     ('Image Files', '*.png')])

        if img_name:
            canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
            img_canvas = image.load_img(img_name, target_size=canvas_size)
            img = ImageTk.PhotoImage(img_canvas)
            self.canvas.create_image(0, 0, anchor=NW, image=img)
            self.canvas.image = img

            prediction, score = model.predict_img(img_canvas)
            self.predict_lbl.config(text="Estimated class: {} ({:.2f}%)"
                                    .format(prediction, score))

    def img_capture(self):
        _, frame = self.cam.read()
        RGB_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        RGB_img = image.array_to_img(RGB_arr)

        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        img_canvas = RGB_img.resize(canvas_size)
        img = ImageTk.PhotoImage(img_canvas)
        self.canvas.create_image(0, 0, anchor=NW, image=img)
        self.canvas.image = img

        prediction, score = model.predict_img(img_canvas)

        self.predict_lbl.config(text="Estimated class: {} ({:.2f}%)"
                                .format(prediction, score))

        # loop
        if self.interval == 30:
            self.after(self.interval, self.img_capture)
        else:
            self.interval = 30


model = Model.CNN_Model()
app = App()
app.center_window(900, 650)
app.resizable(0, 0)
mainloop()