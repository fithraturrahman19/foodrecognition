# import necessary libraries
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import heapq
import pandas as pd
import numpy as np
import csv

# read food classes file
classes_file = open("classes.txt", "r")
class_complete = classes_file.read().split('\n')
class_list = class_complete
class_list.pop()
#print(len(class_list))

# read food classes file
labels_file = open("labels.txt", "r")
label_complete = labels_file.read().split('\n')
label_list = label_complete
#label_list.pop()
#print(len(label_list))

score = 0
i = 0

# load trained model
# reconstructed_model = load_model("101class_model_mobilenet_run5.h5")
# reconstructed_model = load_model("101class_model_mobilenet_full10.h5")
# reconstructed_model = load_model("101class_model_mobilenet_full10.h5")
reconstructed_model = load_model("101class_model_mobilenet_last10.h5")

# prepare dictionary
import os
directory = 'test_images/'

name_list = []

for filename in os.listdir(directory):
    # load image to predict
    img_path = 'test_images/'+filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    re_preds = reconstructed_model.predict(x)

    # output food class to console
    # print(max(re_preds[0]), np.argmax(re_preds[0]))
    the_food = class_list[np.argmax(re_preds[0])]
    # print('food type: '+str(the_food))

    # output certainty
    certainty = max(re_preds[0])
    # print('certainty: '+str(certainty))

    # output calorie to console
    data_calories = pd.read_csv('calories.csv')
    calories_value = data_calories.at[data_calories['category_name'].eq(the_food).idxmax(),'calorie']
    # print('calorie: '+str(calories_value)+' kcal')

    # print(i)
    if(label_list[i] == the_food):
        score = score+1
    name_list.append(the_food)
    i = i+1

#print(len(name_list))
print('score: '+str(score)+'%')

name_list_df = pd.DataFrame(name_list)
name_list_df.to_csv('test_image_result.csv')