#%%
import cv2 , datetime , os , warnings ,glob , random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
print("Done")
#%%
img_dim=(96,96,3)
data=[]
labels=[]

#%%
image_files=[f for f in glob.glob(r'C:\Users\siddh\Desktop\Projects\GenderClassification\datafiles\Gender-Detection\gender_dataset_face' + '/**/*' , recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)
# %%
women,men= 1,1
for img in image_files:
    image=cv2.imread(img)
    image=cv2.resize(image,(96,96))
    image=img_to_array(image)
    data.append(image)
    
    lab=img.split(os.path.sep)[-2]
    if lab=='woman':
        lab=1
        women=women+1
    else:
        lab=0
        men=men+1
    labels.append(lab)
# %%
data=np.array(data,dtype=np.float64)/255.0
labels=np.array(labels,dtype=np.float64)
print(data[0].shape)
print(labels.shape)
# sns.countplot(labels)
# %%
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=100)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# %%
train_y=to_categorical(y_train,num_classes=2)
test_y=to_categorical(y_test,num_classes=2)

# %%
aug_data=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# %%
def build(height, width, depth , classes):
    model=keras.models.Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if keras.backend.image_data_format=="channel_first":
        imputShape=(depth,height,width)
        chanDim = 1

    
    
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model
    
# %%
model=build(width=img_dim[0], height=img_dim[1], depth=img_dim[2],classes=2)
model.summary()
# %%
opt=keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
# %%
# # history_model = model.fit_generator(aug_data.flow(x_train, train_y, batch_size=64),
#                         validation_data=(x_test,test_y),
#                         epochs=3, verbose=1)
# %%
model.fit(x_train, train_y, batch_size=64,
                        validation_data=(x_test,test_y),
                        epochs=50, verbose=1)
# %%
model.save('gender_detection_new.model')
# %%
model.evaluate(x_test,test_y)
# %%
keras.backend.clear_session()
# %%




























