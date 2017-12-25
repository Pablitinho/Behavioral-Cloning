import pickle

import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.regularizers import l2


# -----------------------------------------------------------------------
def generator(sub_name, num_batches):
    while 1: # Loop forever so the generator never terminates
        for pick_id in range(num_batches):
            with open('./augmented_data/images_' + sub_name + str(pick_id) + '.pickle', 'rb') as handle:
                X_train = np.array(pickle.load(handle))
            with open('./augmented_data/stearing_' + sub_name + str(pick_id) + '.pickle', 'rb') as handle:
                Y_train = np.array(pickle.load(handle))
            Y_train = np.expand_dims(Y_train, axis=1)
            #print(" pick id:",pick_id)
            yield sklearn.utils.shuffle(X_train, Y_train)
# -----------------------------------------------------------------------
import generate_data as gd
# -----------------------------------------------------------------------
gd.create_pickle('./data/driving_log.csv', './data/IMG/')

model = Sequential()
# Convert to grayscale
#model.add(Lambda(lambda x:(0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:]),input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))

# # # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Conv2D(32, (5, 5),strides = (2, 2), border_mode='valid', W_regularizer=l2(0.00001)))
# # #model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# #
model.add(Conv2D(64, (5, 5),strides = (2, 2), border_mode='valid', W_regularizer=l2(0.00001)))
# # #model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# #
model.add(Conv2D(128, (5, 5),strides = (2, 2), border_mode='valid', W_regularizer=l2(0.00001)))
# # #model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),border_mode='valid', W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),border_mode='valid', W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
# # # model.add(Conv2D(64, (3,3),border_mode='valid', W_regularizer=l2(0.00001)))
# # # model.add(Activation('elu'))
#
model.add(Flatten())
model.add(Dense(500, W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=64)

num_sub_samples_train = 378
num_sub_samples_test = 94
sub_name_train = 'train_'
sub_name_validation = 'validation_'

train_generator = generator(sub_name_train, num_sub_samples_train)
validation_generator = generator(sub_name_validation, num_sub_samples_test)

model.fit_generator(train_generator, samples_per_epoch=num_sub_samples_train, validation_data=validation_generator, nb_val_samples=num_sub_samples_test, nb_epoch=3)
model.save('model.h5')
exit()


























