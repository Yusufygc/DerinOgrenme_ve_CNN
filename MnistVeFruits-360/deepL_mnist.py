from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D ,Activation, Dropout , Flatten , Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from  glob import glob 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#VERİ ÇEKME VE ÖN İŞLEMLER İLE VERİYİ KULLANILABİLR HALE GETİRME

def load_and_prepocess(data_path):
    data = pd.read_csv(data_path) # csv dosyasını okur
    data =data.values # veriyi matrise çevirir
    np.random.shuffle(data) # veriyi karıştırır
    x = data[:,1:].reshape(-1,28,28,1)/255.0   # resim
    y= data[:,0].astype(np.int32) # label
    y= to_categorical(y,num_classes=len(set(y))) # one hot encoding

    return x,y

train_path ="C:\\Mnist-dataset\\mnist_train.csv"
test_path ="C:\\Mnist-dataset\\mnist_test.csv"


x_train,y_train = load_and_prepocess(train_path)
x_test,y_test = load_and_prepocess(test_path)


index = 3
vis = x_train.reshape(60000,28,28) # 60000 resim 28,28 boyutunda
plt.imshow(vis[index,:,:])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index])) # hangi sayı olduğunu gösterir
 
# CNN MODELİ OLUŞTURMA

numberOfClass = y_train.shape[1] # kaç tane class olduğunu gösterir

model = Sequential()

model.add(Conv2D(input_shape = (28,28,1),filters =16,kernel_size=(3,3))) # 16 tane filtre 3,3 boyutunda
model.add(BatchNormalization()) # normalizasyon işlemi
model.add(Activation("relu")) # aktivasyon fonksiyonu
model.add(MaxPooling2D()) # max pooling işlemi

model.add(Conv2D(filters =64,kernel_size=(3,3))) # 64 tane filtre 3,3 boyutunda
model.add(BatchNormalization()) # normalizasyon işlemi
model.add(Activation("relu")) # aktivasyon fonksiyonu
model.add(MaxPooling2D()) # max pooling işlemi

model.add(Conv2D(filters =128,kernel_size=(3,3))) # 128 tane filtre 3,3 boyutunda
model.add(BatchNormalization()) # normalizasyon işlemi
model.add(Activation("relu")) # aktivasyon fonksiyonu
model.add(MaxPooling2D()) # max pooling işlemi

model.add(Flatten()) # düzleştirme işlemi
model.add(Dense(units =256)) # nöron sayısı
model.add(Activation("relu")) # aktivasyon fonksiyonu
model.add(Dropout(0.5)) # %50 drop out
model.add(Dense(units =numberOfClass)) # nöron sayısı
model.add(Activation("softmax")) # aktivasyon fonksiyonu

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]) # modeli derleme
#TRAIN
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=25,batch_size=4000) # modeli eğitme

#SAVE MODEL
model.save_weights("mnist_model.h5") # modeli kaydetme

print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()

plt.figure()# ikinci grafik için

plt.plot(hist.history["Acc"],label="Train acc")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()

# SAVE HIStory

import json
with open("mnist_history.json","w") as f:
    json.dump(hist.history,f) # history kaydetme

# LOAD HISTORY
import codecs
with codecs.open("mnist_history.json","r",encoding="utf-8") as f:
    h = json.loads(f.read()) # history yükleme

plt.figure()
plt.plot(h["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()
plt.figure()# ikinci grafik için
plt.plot(h["Acc"],label="Train acc")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()

