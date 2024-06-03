import tensorflow as tf
from keras.api.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import pickle
lables = []
pic = 0
with open(r'.\models\label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
def show(model):
    test_images = []
    test_labels = []
    path = r'one'
    dir_list = os.listdir(path)
    
    for i in dir_list:
        dir = os.path.join(path, i)
        file_list = os.listdir(dir)
        for j in file_list:
            files = os.path.join(dir, j)
            img = cv2.imread(files)
            img = cv2.resize(img, (64,64))
            img = np.array(img, dtype=np.float32)
            img = img/255
            test_images.append(img)
            test_labels.append(i)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    preds = model.predict(X_test)
    predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))
    print('\n'*4)
    print(predicted_labels[pic])
    plt.imshow(X_test[pic])
    plt.title(f"Label: {predicted_labels[pic]}")
    plt.show()

# Загрузка модели
model = tf.keras.models.load_model(r'.\models\animalsModel.h5')

# Вызов функции для тестирования
show(model)
