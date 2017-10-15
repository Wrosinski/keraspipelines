import os

import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keraspipelines import KerasFlowPipeline

number_classes = 10


# Get example data - CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transform target into binary matrix
y_train = to_categorical(y_train, number_classes)
y_test = to_categorical(y_test, number_classes)

# Transform data into floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Define model callbacks and parameters passed directly to the model definition
# as specified in cnn_models.py
model_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

model_parameters = {
    'img_size': (32, 32, 3),
    'num_classes': number_classes,
}

# Define ImageDataGenerators with image augmentation parameters
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


# Run parameters for bagging & for KFold using .flow method
# to augment training & test data for improved model performance
flow_bag_parameters = {
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
    'callbacks': model_callbacks,

    'src_dir': os.getcwd(),

    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_test_augmentations': 5,

    'run_save_name': 'basic_cnn_flow_cifar_bagging',
    'save_history': True,
    'save_model': True,
    'output_statistics': True,

    'X_train': x_train,
    'y_train': y_train,
    'X_test': x_test,
}

flow_kf_parameters = {
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
    'batch_size': 16,
    'callbacks': model_callbacks,

    'src_dir': os.getcwd(),

    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_test_augmentations': 5,

    'run_save_name': 'basic_cnn_flow_cifar_kfold',
    'save_history': True,
    'save_model': True,
    'output_statistics': True,

    'X_train': x_train,
    'y_train': y_train,
    'X_test': x_test,
}


# Pipelines definition with parameters defined by above dictionaries
flow_bag_pipeline = KerasFlowPipeline(model_name=flow_bag_parameters['model_name'],
                                      model_params=flow_bag_parameters['model_params'],
                                      predict_test=flow_bag_parameters['predict_test'],
                                      n_bags=flow_bag_parameters['n_bags'],
                                      split_size=flow_bag_parameters['split_size'],
                                      seed=flow_bag_parameters['seed'],
                                      verbose=flow_bag_parameters['verbose'],
                                      number_epochs=flow_bag_parameters['number_epochs'],
                                      batch_size=flow_bag_parameters['batch_size'],
                                      callbacks=flow_bag_parameters['callbacks'],

                                      train_datagen=flow_bag_parameters['train_datagen'],
                                      valid_datagen=flow_bag_parameters['valid_datagen'],
                                      test_datagen=flow_bag_parameters['test_datagen'],
                                      number_test_augmentations=flow_bag_parameters[
                                          'number_test_augmentations'],

                                      run_save_name=flow_bag_parameters['run_save_name'],
                                      save_history=flow_bag_parameters['save_history'],
                                      save_model=flow_bag_parameters['save_model'],
                                      output_statistics=flow_bag_parameters['output_statistics'])

flow_kfold_pipeline = KerasFlowPipeline(model_name=flow_kf_parameters['model_name'],
                                        model_params=flow_kf_parameters['model_params'],
                                        predict_test=flow_kf_parameters['predict_test'],
                                        n_folds=flow_kf_parameters['n_folds'],
                                        stratify=flow_kf_parameters['stratify'],
                                        split_size=flow_kf_parameters['split_size'],
                                        seed=flow_kf_parameters['seed'],
                                        verbose=flow_kf_parameters['verbose'],
                                        number_epochs=flow_kf_parameters['number_epochs'],
                                        batch_size=flow_kf_parameters['batch_size'],
                                        callbacks=flow_kf_parameters['callbacks'],

                                        train_datagen=flow_kf_parameters['train_datagen'],
                                        valid_datagen=flow_kf_parameters['valid_datagen'],
                                        test_datagen=flow_kf_parameters['test_datagen'],
                                        number_test_augmentations=flow_kf_parameters[
                                            'number_test_augmentations'],

                                        run_save_name=flow_kf_parameters['run_save_name'],
                                        save_history=flow_kf_parameters['save_history'],
                                        save_model=flow_kf_parameters['save_model'],
                                        output_statistics=flow_kf_parameters['output_statistics'])


# Run bagging & KFold
bagging_model, bagging_preds_valid, bagging_preds_test = flow_bag_pipeline.bag_flow_run(
    X_train=flow_bag_parameters['X_train'],
    y_train=flow_bag_parameters['y_train'],
    X_test=flow_bag_parameters['X_test'])

kf_model, oof_train, oof_test = flow_kfold_pipeline.kf_flow_run(
    X_train=flow_kf_parameters['X_train'],
    y_train=flow_kf_parameters['y_train'],
    X_test=flow_kf_parameters['X_test'])
