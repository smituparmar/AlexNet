import tensorflow.keras as keras

from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Flatten,ZeroPadding2D,BatchNormalization,Dropout
from keras.models import Sequential 


model=Sequential([
        #first Convolutional Layer which has input image as (227,227) in RGB mode.
        Conv2D(filters=96, strides=(4,4),input_shape=(227,227,3),kernel_size=(11,11),activation='relu'),
        MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        BatchNormalization(),
        
        #2nd Convolutional Layer
        Conv2D(filters=256, strides=(1,1),kernel_size=(5,5),activation='relu'),
        ZeroPadding2D(padding=2),
        MaxPooling2D(pool_size=(3,3),strides=(2,2)), 
        BatchNormalization(),
        
        #3-4-5 Convolutional Layer without any pooling layer
        Conv2D(filters=384, strides=(1,1),kernel_size=(3,3),activation='relu'), 
        ZeroPadding2D(padding=1),
        
        Conv2D(filters=384, strides=(1,1),kernel_size=(3,3),activation='relu'),
        ZeroPadding2D(padding=(1)),

        Conv2D(filters=256, strides=(1,1),kernel_size=(3,3),activation='relu'),
        ZeroPadding2D(padding=(1)),
        
        #seperate Pooling layer
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization(),
        
        #flatter this output
        Flatten(),
        
        #dense network with 4096 units followed by 50% droput
        Dense(input_shape=(224*224,3),units=4096,activation='relu'),
        Dropout(0.5),
        
        #same process as above
        Dense(units=4096,activation='relu'),
        Dropout(0.5),
        
        #number of classes or output.
        Dense(1000,activation='relu'),
        
        ])
model.summary()