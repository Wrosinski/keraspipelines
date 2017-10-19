# [Keras Pipelines](https://github.com/Wrosinski/keraspipelines) - Rapid Experimentation & Easy Usage

## Overview

During my adventure with Machine Learning and Deep Learning in particular, I spent a lot of time working with Convolutional Neural Networks. I used them a lot in Kaggle competitions and later, in research projects I have been doing. Over the months I tried to establish a workflow which would enable quick change of model definition and parameters, of dataset on which it is being trained and of many more parameters.

So far, the library of my choice was Keras, as for 95% examples, especially when you are using either CNN's or MLP's, it will provide sufficient capabilities while delivering an intuitive, high-level API to define and manipulate your neural networks.

I decided to clean my code up and create a more reusable framework for Keras models. During this process, I thought that some of people who are already working with Keras but did not have time to establish a pipeline for themselves may benefit from publishing those pipelines. Some may find them ready to use as they are, others will adjust them to their liking.

- [**Official Detailed Example**](https://github.com/Wrosinski/keraspipelines/tree/master/examples)
- [**Documentation**](https://github.com/Wrosinski/keraspipelines/tree/master/docs)

## Example usage

I will provide an example of usage based on Kaggle's [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) playground challenge.

Start with downloading the data, extract it and put in a chosen folder. I put my scripts in `/scripts` and data in `/input`.

To begin with, I'll define my models in `dogs_cnn_models.py` in `/scripts` directory in order to be able to call them for the pipeline.

An example, ResNet50 pretrained on ImageNet:

```python
from keras.applications.resnet50 import ResNet50
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Flatten
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
```

Now, let's start with the proper script, in which we will create a `KerasPipeline` object and use it to train our ResNet defined in `dogs_cnn_models.py`. We will call the script `dogs_training_script.py`.

1. First step, load the needed libraries:

```python
import os
import cv2
import dogs_cnn_models
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keraspipelines import KerasPipeline
from scipy import misc
from tqdm import tqdm
```

2. Let's define functions to load data:

```python
def load_image(path, img_size):
    img = misc.imread(path, mode='RGB')
    img = misc.imresize(img, img_size)
    return img


def load_data(src, df, img_size, labels=None):

    X = np.zeros((df.shape[0], img_size[0], img_size[1], 3), dtype='float32')
    if labels is not None:
        y_train = []

    for i in tqdm(range(df.shape[0])):
        X[i] = load_image('{}{}.jpg'.format(
            src, df.iloc[i, :]['id']), img_size)
        if labels is not None:
            y_train.append(labels[i])

    if labels is not None:
        return X, np.array(y_train, np.uint8)
    else:
        return X
```

3. Now we will specify basic parameters, directories to load data from and desired size of images to be provided into our model.

```python
src_dir = '../input/'
src_train = src_dir + 'train/'
src_test = src_dir + 'test/'
image_size = (224, 224)
```

4. Let's read source files to get ID's and labels, which will enable us to load the images.

Labels should be One-Hot encoded in order to be fed into the model.

```python
df_train = pd.read_csv(src_dir + 'labels.csv')
df_test = pd.read_csv(src_dir + 'sample_submission.csv')


targets_series = pd.Series(df_train['breed'])
one_hot_df = pd.get_dummies(targets_series, sparse=True)
one_hot = one_hot_df.values

number_classes = y_train.shape[1]

X_train, y_train = load_data(src_train, df_train, image_size, one_hot)
X_test = load_data(src_test, df_test, image_size)

print('Training data shape:', X_train.shape)
print('Test data shape:', X_test.shape)
```

5. Now, we will prepare model parameters dictionary, callbacks and parameters for data augmentation. All of those will enable us to create a `KerasFlowPipeline` object forming a pipeline with all our defined parameters based on a dictionary.

```python
model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                     patience=3, min_lr=1e-5)]

model_parameters = {
    'img_size': (image_size[0], image_size[1], 3),
    'num_classes': number_classes,
}


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.25,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    channel_shift_range=0.07)

valid_datagen = ImageDataGenerator(rescale=1. / 255,)


pipeline_parameters = {
    'model_name': getattr(dogs_cnn_models, 'resnet_dense'),
    'predict_test': True,
    'model_callbacks': model_callbacks,
    'number_epochs': 1,
    'batch_size': 16,
    'seed': 1337,
    'shuffle': True,
    'verbose': True,

    'run_save_name': 'resnet_dense_5fold_SKF_run1',
    'load_keras_model': False,
    'save_model': True,
    'save_history': True,
    'save_statistics': True,
    'output_statistics': True,

    'src_dir': os.getcwd(),

    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_test_augmentations': 5,
}
```

6. Let's now feed the object with defined parameters and run it!

```python
pipeline = KerasPipeline(model_name=pipeline_parameters['model_name'],
                         predict_test=pipeline_parameters['predict_test'],
                         model_callbacks=pipeline_parameters['model_callbacks'],
                         number_epochs=pipeline_parameters['number_epochs'],
                         batch_size=pipeline_parameters['batch_size'],
                         seed=pipeline_parameters['seed'],
                         shuffle=pipeline_parameters['shuffle'],
                         verbose=pipeline_parameters['verbose'],

                         run_save_name=pipeline_parameters['run_save_name'],
                         load_keras_model=pipeline_parameters['load_keras_model'],
                         save_model=pipeline_parameters['save_model'],
                         save_history=pipeline_parameters['save_history'],
                         save_statistics=pipeline_parameters['save_statistics'],
                         output_statistics=pipeline_parameters['output_statistics'],

                         src_dir=pipeline_parameters['src_dir'],

                         train_datagen=pipeline_parameters['train_datagen'],
                         valid_datagen=pipeline_parameters['valid_datagen'],
                         test_datagen=pipeline_parameters['test_datagen'],
                         number_test_augmentations=pipeline_parameters['number_test_augmentations'],
                         )


kf_model, oof_train, oof_test = pipeline.kfold_run(
   X_train=X_train,
   y_train=y_train,
   X_test=X_test,
   model_params=model_parameters,
   n_folds=5,
   stratify=True,
   index_number=1,
   flow_augment=True
)
```

This will output model from the last fold and out-of-fold predictions for train & test.

Additionally we can save the predictions into pickle files, so we can later load them and combine, if we would like to stack or blend predictions from our models. Let's also output a submission:

```python
pd.to_pickle(oof_train, 'OOF_train_resnet_dense_5fold_SKF_run1.pkl')
pd.to_pickle(oof_test, 'OOF_test_resnet_dense_5fold_SKF_run1.pkl')

submission = pd.DataFrame(oof_test.mean(axis=-1))
submission.columns = one_hot_df.columns.values
submission.insert(0, 'id', df_test['id'])
submission.to_csv(
    'SUB_resnet_dense_5fold_SKF_run1.csv', index=False)
```
