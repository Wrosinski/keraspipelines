import os

import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.utils import to_categorical
from keraspipelines import KerasPipeline

number_classes = 10


# Get example data - CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transform target into binary matrix
y_train = to_categorical(y_train, number_classes)
y_test = to_categorical(y_test, number_classes)

# Transform data into floats & divide by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Define model callbacks and parameters passed directly to the model definition
# as specified in cnn_models.py
model_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

model_parameters = {
    'img_size': (32, 32, 3),
    'num_classes': number_classes,
}


# Run parameters for bagging & for KFold
bag_parameters = {
    'model_name': getattr(cnn_models, 'basic_cnn'),
    'model_params': model_parameters,
    'predict_test': True,
    'n_bags': 2,
    'split_size': 0.2,
    'seed': 1337,
    'user_split': False,
    'verbose': True,
    'number_epochs': 1,
    'batch_size': 256,
    #'callbacks': model_callbacks,

    'src_dir': os.getcwd(),

    'run_save_name': 'basic_cnn_cifar_bagging',
    'save_history': True,
    #'save_model': True,
    'output_statistics': True,

    'X_train': x_train,
    'y_train': y_train,
    'X_test': x_test,
}

kf_parameters = {
    'model_name': getattr(cnn_models, 'basic_cnn'),
    'model_params': model_parameters,
    'predict_test': True,
    'n_folds': 2,
    'stratify': True,
    'split_size': 0.2,
    'seed': 1337,
    'user_split': False,
    'verbose': True,
    'number_epochs': 1,
    'batch_size': 256,
    'callbacks': model_callbacks,

    'src_dir': os.getcwd(),

    'run_save_name': 'basic_cnn_cifar_kfold',
    'save_history': True,
    'save_model': True,
    'output_statistics': True,

    'X_train': x_train,
    'y_train': y_train,
    'X_test': x_test,
}


# Pipelines definition with parameters defined by above dictionaries
bag_pipeline = KerasPipeline(model_name=bag_parameters['model_name'],
                             model_params=bag_parameters['model_params'],
                             predict_test=bag_parameters['predict_test'],
                             n_bags=bag_parameters['n_bags'],
                             split_size=bag_parameters['split_size'],
                             number_epochs=bag_parameters['number_epochs'],
                             batch_size=bag_parameters['batch_size'],
                             seed=bag_parameters['seed'],
                             user_split=bag_parameters['user_split'],
                             # callbacks=bag_parameters['callbacks'],
                             run_save_name=bag_parameters['run_save_name'],
                             save_history=bag_parameters['save_history'],
                             # save_model=bag_parameters['save_model'],
                             output_statistics=bag_parameters['output_statistics'])

kfold_pipeline = KerasPipeline(model_name=kf_parameters['model_name'],
                               model_params=kf_parameters['model_params'],
                               predict_test=kf_parameters['predict_test'],
                               n_folds=kf_parameters['n_folds'],
                               stratify=kf_parameters['stratify'],
                               number_epochs=kf_parameters['number_epochs'],
                               batch_size=kf_parameters['batch_size'],
                               callbacks=kf_parameters['callbacks'],
                               run_save_name=kf_parameters['run_save_name'],
                               save_history=kf_parameters['save_history'],
                               save_model=kf_parameters['save_model'],
                               output_statistics=kf_parameters['output_statistics'])


# Run bagging & KFold
bagging_model, bagging_preds_valid, bagging_preds_test = bag_pipeline.bag_run(
    X_train=bag_parameters['X_train'],
    y_train=bag_parameters['y_train'],
    X_test=bag_parameters['X_test'])

kf_model, oof_train, oof_test = kfold_pipeline.kfold_run(
    X_train=kf_parameters['X_train'],
    y_train=kf_parameters['y_train'],
    X_test=kf_parameters['X_test'])
