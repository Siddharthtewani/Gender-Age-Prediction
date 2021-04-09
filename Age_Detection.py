#%%
import cv2 , datetime , os , warnings ,glob , random , re 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
print("Done")
#%%
image_files=[f for f in glob.glob(r'C:\Users\siddh\Desktop\Projects\Gender&AgeClassification\datafiles\Image-Dataset-for-Age-Prediction\images' + '/*' , recursive=True)]
random.shuffle(image_files)
print("Done")
# %%
age=[]
age_label=[]
for img in image_files:
    data=re.findall("[*\d]{0,3}[_\d]",img)
    age.append(data[0])
    
for i in age:
    x = i.split("_")
    
    age_label.append(x[0])

# %%
data_images=[]
for img in image_files:
    image=cv2.imread(img)
    image=cv2.resize(image,(96,96))
    image=img_to_array(image)
    data_images.append(image)
#%%
data_images=np.array(data_images,dtype=np.float64)/255.0 
age_label=np.array(age_label,dtype=np.float64)
# %%

# cv2.imshow("Image",data_images[2])
# print(age_label[2])
# cv2.waitKey()
# cv2.destroyAllWindows()
# %%
x_train,x_test,y_train,y_test=train_test_split(data_images,age_label,test_size=0.3,random_state=100)
# %%
int_dim=(96,96,3)
y_test=np.array(y_test,dtype=np.float64)
y_train=np.array(y_train,dtype=np.float64)
# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# %%
def bulid_model(height, width, depth):
    model=keras.Sequential()

    model.add(keras.layers.Conv2D(500,(5,5),input_shape=(height ,width, depth)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(500,(3,3)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(550,(5,5)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(0.2))




    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1000))
    model.add(keras.layers.Activation("relu"))

    return model

# %%
model=bulid_model(height=96, width=96, depth=3)
model.summary()
# %%
opt=keras.optimizers.Adam(learning_rate=0.1)
mse = tf.keras.losses.MeanSquaredError()
m = tf.keras.metrics.MeanSquaredError()
model.compile(optimizer=opt,
                loss=mse,
                metrics=m)
# # %%
# from tensorflow.keras.models import  load_model
# model=load_model("age_detection_latestmodel1.model")
# %%
model.fit(x_train, y_train, batch_size=64,
                        validation_data=(x_test,y_test),
                        epochs=1)
# %%
model.evaluate(x_test,y_test)

# %%
model.save('age_detection_latestmodel1.model')
# %%
model.fit(x_train, y_train, batch_size=64,
                        validation_data=(x_test,y_test),
                        epochs=3)
# %%
model.summary()
# %%
x_train
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
