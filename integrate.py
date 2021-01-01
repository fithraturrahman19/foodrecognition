from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

# Set Base Page Root
root = Tk()
root.title('Calorie Estimator')
root.geometry("350x450")
color = '#e0c68b'
root.configure(bg=color)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)

# Base Image , Food Name, Food Calorie
food_name = 'Food_name'
food_calorie = 'Food_calories'
certainty = 'Certainty'

# predict_import():
# import necessary libraries
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import heapq
import pandas as pd
import numpy as np

# read food classes file
classes_file = open("classes.txt", "r")
class_complete = classes_file.read().split('\n')
class_list = class_complete
class_list.pop()

# load trained model
reconstructed_model = load_model("101class_model_mobilenet_full10.h5")

def predict_application():
    # load image to predict
    img_path = path3
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    re_preds = reconstructed_model.predict(x)

    # output food class to console
    # print(max(re_preds[0]), np.argmax(re_preds[0]))
    the_food = class_list[np.argmax(re_preds[0])]
    print('food type: '+str(the_food))

    # output certainty
    certainty_value = max(re_preds[0])
    print('certainty: '+str(certainty))

    # output calorie to console
    data_calories = pd.read_csv('calories.csv')
    calories_value = data_calories.at[data_calories['category_name'].eq(the_food).idxmax(),'calorie']
    print('calorie: '+str(calories_value)+' kcal')

    # pass as output
    return([str(the_food), str(calories_value), str(certainty_value)])

# Check Calorie Function
def check_calorie():
    [food_name, food_calorie, certainty] = predict_application()
    food_name = food_name.replace("_"," ")
    food_name = food_name.capitalize()
    foodNameLabel.config(text=('Food Type:'+'  '+food_name))
    foodCalorieLabel.config(text=('Food Calorie:'+'  '+food_calorie))
    certaintyLabel.config(text=('Certainty:'+'  '+certainty))
    
# Browse and Display Image Function
def browse_file():
    global browse_img
    global path3
    path3 = filedialog.askopenfilename(initialdir="/Learning/Python/py-objdetection/objdet",
                                       title="Select A File", filetypes=(("jpeg files", "*.jpg"),
                                                                         ("jpeg files", "*.jpeg"),
                                                                         ("all files", "*.*")))
    browse_img = Image.open(path3)
    browse_img = browse_img.resize((300,300))
    browse_img = ImageTk.PhotoImage(browse_img)
    canvasImage.itemconfig(image_id, image=browse_img)

# Create Title Label
titleLabel = Label(root, text="Calorie Estimation")
titleLabel.configure(font='Terminal',bg=color)
titleLabel.grid(row=0,column=0,columnspan=2,sticky='we',padx=20,pady=5)

# Canvas for Image
canvasImage = Canvas(root,width=300,height=300,borderwidth=5,bg='white') 
canvasImage.grid(row=1,column=0,columnspan=2,sticky='')
image_id = canvasImage.create_image(0,0, anchor='nw')

# Create Select Button
selectButton = Button(root, text="Choose Food Image", command=browse_file)
selectButton.grid(row=2,column=0,sticky='we',padx=20,pady=5)

# Create Check Calorie Button
calorieButton = Button(root, text="Check Calorie", command=check_calorie)
calorieButton.grid(row=2,column=1,sticky='we',padx=20,pady=5)

# Display food name
foodNameLabel = Label(root)
foodNameLabel.config(text=('Food Type:'+'  '+food_name))
foodNameLabel.grid(row=3,column=0,columnspan=2,sticky='we',padx=20,pady=5)

# Display food calories
foodCalorieLabel = Label(root)
foodCalorieLabel.config(text=('Food Calorie:'+'  '+food_calorie))
foodCalorieLabel.grid(row=4,column=0,columnspan=2,sticky='we',padx=20,pady=5)

# Display food calories
certaintyLabel = Label(root)
certaintyLabel.config(text=('Certainty:'+'  '+certainty))
certaintyLabel.grid(row=5,column=0,columnspan=2,sticky='we',padx=20,pady=5)

# Loop
root.mainloop()