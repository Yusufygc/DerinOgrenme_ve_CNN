from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D ,Activation, Dropout , Flatten , Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  , img_to_array , load_img
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
from  glob import glob # kaç clasın olduğunu öğrenmeyi sağlar

train_path ="C:\\fruits-360\\Training\\"
test_path ="C:\\fruits-360\\Test\\"


