import tkinter as tk

from PIL import ImageTk
from keras.preprocessing.image import load_img

window = tk.Tk()
window.geometry("500x500")

image_frame = tk.Label(window).pack()

display_text = tk.StringVar()

display = tk.Label(window, textvariable=display_text).pack()


def display_ui_image(path, x, y):
    global image_frame

    image = load_img(path, target_size=(x, y))

    img = ImageTk.PhotoImage(image)
    if image_frame is None:
        image_frame = tk.Label(window, image=img).pack()
    else:
        image_frame.configure(image=img)

    window.mainloop()


def add_text(text):
    global display_text

    s = display_text.get()
    s += text
    display_text.set(s)


def set_text(text):
    global display_text

    s = display_text.get()
    s = text
    display_text.set(s)


def update_image(image):
    global image_frame

    image_frame.configure(image=image)
    image_frame.image = image
