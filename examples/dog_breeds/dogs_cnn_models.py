from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import (AveragePooling2D, Convolution2D,
                                        MaxPooling2D, ZeroPadding2D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

learning_rate = 0.0001
optimizer = Adam(lr=learning_rate, decay=1e-3)


# ResNet 50
def resnet_dense(params):
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                            input_tensor=None, input_shape=params['img_size'])
    output = resnet_notop.get_layer(index=-1).output
    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(params['num_classes'],
                   activation='softmax', name='predictions')(output)
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=['accuracy'])
    return Resnet_model

# Xception


def xception_globalavgpool(params):
    Xception_notop = Xception(include_top=False, weights='imagenet',
                              input_tensor=None, input_shape=params['img_size'])
    output = Xception_notop.get_layer(index=-1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(params['num_classes'],
                   activation='softmax', name='predictions')(output)
    Xception_model = Model(Xception_notop.input, output)
    Xception_model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])
    return Xception_model


# Inception V3
def inception_globalavgpool(params):
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                    input_tensor=None, input_shape=params['img_size'])
    output = InceptionV3_notop.get_layer(index=-1).output
    output = GlobalAveragePooling2D()(output)
    output = Dense(params['num_classes'],
                   activation='softmax', name='predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    InceptionV3_model.compile(
        loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return InceptionV3_model
