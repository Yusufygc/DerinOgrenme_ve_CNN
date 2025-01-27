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


# Veri çoğaltma işlemleri -train-test

#rescale 0-1 arasına çekiyor normalize etmek için
#shear_range = kesme açısı resmi belirli bir açıyla şekillendirir.
#zoom_range = resmi yakınlaştırır
train_dataGen=ImageDataGenerator(rescale=1./255, 
                   shear_range=0.3,
                   horizontal_flip=True,
                   zoom_range=0.3)

# modelimi rescale olan veriler ile eğittimiz için test verilerini de rescale etmemiz gerekiyor.
test_dataGen =ImageDataGenerator(rescale=1./255)

#train_path içindeki klasörleri ve içlerindeki resimleri bulur ↓↓↓ calss_mode =categorical birden fazla clası var demek
train_generator = train_dataGen.flow_from_directory(
    train_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical") 

test_generator = test_dataGen.flow_from_directory(
    test_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical") 

#steps_per_epoch = 1600 / batch_size = 1600 adet resim var 32 lik batch_size ile 50 tane iterasyon yapar
hist = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1600 // batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=800 // batch_size)

#Model save

model.save_weights("fruits_uyg.h5")

#Veri görselleştirme Model Değerlendirmesi/Model Evaluation 

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

# save history
import json
with open("fruits_uyg.json","w") as f:
    json.dump(hist.history,f)

# load history
import codecs
with codecs.open("fruits_uyg.json","r",encoding="utf-8") as f:
    h = json.loads(f.read())

plt.plot(h["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()

plt.figure()# ikinci grafik için

plt.plot(h["Acc"],label="Train acc")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()#label gösterir
plt.show()