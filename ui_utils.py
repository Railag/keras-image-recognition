import tkinter as tk

from PIL import ImageTk
from keras.preprocessing.image import load_img

window = tk.Tk()
window.geometry("1600x700")

default_image = load_img('madrid.jpg', target_size=(224, 224))
img = ImageTk.PhotoImage(image=default_image)
image_frame = tk.Label(window, image=img)
image_frame.place(x=650, y=20)

display_text = tk.StringVar()

display = tk.Label(window, textvariable=display_text, anchor="w", width="500")
display.place(x=20, y=400)


def display_ui_image(path, x, y):
    global image_frame
    global img

    image = load_img(path, target_size=(x, y))

    img = ImageTk.PhotoImage(image)
    if image_frame is None:
        image_frame = tk.Label(window, image=img)
    else:
        image_frame.configure(image=img)

    window.update()


def add_text(text):
    global display_text

    s = display_text.get()
    s += str(text) + "\n"
    display_text.set(s)

    window.update()


def set_text(text):
    global display_text

    s = display_text.get()
    s = text
    display_text.set(s)

    window.update()


def update_image(image):
    global image_frame

    image_frame.configure(image=image)
    image_frame.image = image

    window.update()
