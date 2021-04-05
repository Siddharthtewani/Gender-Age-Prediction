#%%
import cv2 , datetime , os , warnings ,glob , random , re
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
print("Done")
#%%
image_files=[f for f in glob.glob(r'C:\Users\siddh\Desktop\Projects\Gender&AgeClassification\datafiles\Image-Dataset-for-Age-Prediction\images' + '/*' , recursive=True)]
random.shuffle(image_files)
print("Done")
# %%
data=[]
for img in image_files:

   
    # cv2.imshow("image",image) 
    # cv2.destroyAllWindows()
    pattern =r'([^/]+)_\d_\d_\d+.jpg$'
    p=re.compile(pattern)
    r=p.search(img)
    print(r.group(1))

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
