from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model


def basic_cnn(params):

    input_layer = Input(params['img_size'])
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(params['num_classes'])(x)
    output_layer = Activation('softmax')(x)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
