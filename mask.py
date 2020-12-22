import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_data = ImageDataGenerator(rescale=1/255)

train_generator = train_data.flow_from_directory(
        'dataset',  
        target_size=(150, 150),  
        batch_size=10,
        class_mode='categorical')


modeling = model.fit(
      train_generator,
      epochs=7,
      verbose=1)

#model.save('F:\mask_model_91_1')
#model = tf.keras.models.load_model('F:\mask_model_91_1')

'''for Testing a single image'''
path="test/t1.jpeg"
cascade='haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade)
image = cv2.imread(path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
#print("Found {0} faces!".format(len(faces)))
x1=0

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (150, 150))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    x1 = model.predict(face)[0]
    j = np.unravel_index(x1.argmax(), x1.shape)
    #print(j)
    if j==(0,):
        label="aligned"
    elif j==(1,):
        label="unaligned"
    else:
        label="no mask"
    color = (0, 255, 0) if label == "aligned" else (0, 0, 255)
    cv2.putText(image, label, (x, y+h - 20),cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 5)
plt.imshow(image)
outfile="foo"+"2.png"
cv2.imwrite(outfile,image)

''' For Testing Multiple Images '''
image_path = "test"
for i in os.listdir(image_path):
    path = os.path.join(image_path, i)
    cascade='haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade)
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    x1=0
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (150, 150))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        x1 = model.predict(face)[0]
        j = np.unravel_index(x1.argmax(), x1.shape)
        print(j)
        if j==(0,):
            label="aligned"
        elif j==(1,):
            label="unaligned"
        else:
            label="no mask"
        color = (0, 255, 0) if label == "aligned" else (0, 0, 255)
        cv2.putText(image, label, (x, y+h - 20),cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 5)
    outfile="test_op/"+i+".png"
    cv2.imwrite(outfile,image)
        







