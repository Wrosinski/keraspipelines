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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


# Set directory structure:
src_dir = '/home/w/Projects/Dog_Breeds/input/'
src_train = src_dir + 'train/'
src_test = src_dir + 'test/'
image_size = (224, 224)


df_train = pd.read_csv(src_dir + 'labels.csv')
df_test = pd.read_csv(src_dir + 'sample_submission.csv')


targets_series = pd.Series(df_train['breed'])
one_hot_df = pd.get_dummies(targets_series, sparse=True)
one_hot = one_hot_df.values


X_train, y_train = load_data(src_train, df_train, image_size, one_hot)
X_test = load_data(src_test, df_test, image_size)

print('Training data shape:', X_train.shape)
print('Test data shape:', X_test.shape)


number_classes = y_train.shape[1]

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
    'number_epochs': 10,
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


pd.to_pickle(oof_train, 'OOF_train_resnet_dense_5fold_SKF_run1.pkl')
pd.to_pickle(oof_test, 'OOF_test_resnet_dense_5fold_SKF_run1.pkl')


submission = pd.DataFrame(oof_test.mean(axis=-1))
submission.columns = one_hot_df.columns.values
submission.insert(0, 'id', df_test['id'])
submission.to_csv(
    'SUB_resnet_dense_5fold_SKF_run1.csv', index=False)
