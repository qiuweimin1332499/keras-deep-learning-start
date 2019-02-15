from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.applications import VGG16


conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
#调用卷积基
base_dir='C:\\Users\\lenovo\\.keras\\datasets\\cats_and_dogs_small'
#保存较小数据集的目录
train_dir=os.path.join(base_dir,'train')
#训练集目录
validation_dir=os.path.join(base_dir,'validation')
#验证集目录
test_dir=os.path.join(base_dir,'test')
#测试集目录
datagen=ImageDataGenerator(rescale=1./255)
batch_size=20
def extract_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=sample_count)
    generator=datagen.flow_from_directory(directory,
                                          target_size=(150,150),
                                          batch_size=batch_size,
                                          class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels
train_features,train_labels=extract_features(train_dir,2000)
validation_featurs,validation_labels=extract_features(validation_dir,1000)
test_features,test_labels=extract_features(test_dir,1000)
#图像特征批量化生成
train_features=np.reshape(train_features,(2000,4*4*512))
validation_featurs=np.reshape(validation_featurs,(1000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))
#图像特征展平


from keras import layers
from keras import models
from keras import optimizers
model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history=model.fit(train_features,train_labels,
                  epochs=30,
                  batch_size=batch_size,
                  validation_data=(validation_featurs,validation_labels))
#带有dropout的神经网络


import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()