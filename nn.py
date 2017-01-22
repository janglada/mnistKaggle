from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.utils import np_utils
from keras.models import load_model


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# dimensions of our images.
img_width, img_height = 28, 28


batch_size = 256
nb_classes = 10
nb_epoch = 100

df_ori = pd.read_csv("./data/train.csv", delimiter = ",", dtype=np.float32)
df_tr  = pd.read_csv("train_transformation.csv", delimiter = ",", dtype=np.float32)

df = pd.concat([df_ori, df_tr])
df.sample(frac=1)

train = df.as_matrix()

X = train[:, 1:785]
Y = train[:, 0]
seed = 5

X = X.reshape(X.shape[0], img_width, img_height, 1)

input_shape = (img_width, img_height, 1)


X_train = X.astype('float32')
X_train /= 255



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y, nb_classes)



model = load_model('model_2_extra_data.h5')

# model = Sequential()
#
# model.add(Convolution2D(64, 3,3, border_mode='valid',   input_shape=input_shape))
# model.add(Activation('relu'))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
#
# model.add(Convolution2D(128, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
#
#
# model.add(Convolution2D(256, 2, 2))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
#
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(4096))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

model.fit(X_train, Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=1,
    validation_split=0.2
)


model.save('model_2_extra_data.h5')
