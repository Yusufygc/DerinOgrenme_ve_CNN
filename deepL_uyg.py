from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D ,Activation, Dropout , Flatten , Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  , img_to_array , load_img
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
from  glob import glob # kaç clasın olduğunu öğrenmeyi sağlar

train_path ="C:\\fruits-360\\Training\\"
test_path ="C:\\fruits-360\\Test\\"

img = load_img(train_path + "Apple Braeburn\\0_100.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)

#print(x.shape)


img = load_img(train_path + "Apple Braeburn\\0_100.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)

#print(x.shape)
className = glob(train_path +'/*')
numberOfClass = len(className)

print("number Of Class : ",numberOfClass)

model =Sequential()

model.add(Conv2D(32,(3,3),input_shape= x.shape)) #32=filtre 3,3 filtre boyutu
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak 2,2 

model.add(Conv2D(32,(3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D()) 

model.add(Conv2D(64,(3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D()) 

model.add(Flatten())
model.add(Dense(1024))#nöron sayısı
model.add(Activation("relu"))
model.add(Dropout(0.5)) # 1024 ün yarısını kapatıp kapatıp deneycek her seferinde 512 tanesi aktif olur
model.add(Dense(numberOfClass)) #output layer
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer ="rmsprop",
              metrics =["accuracy"])
batch_size =32 # her iterasyonda kaç resim incelenecek o miktarı ayarlamak için