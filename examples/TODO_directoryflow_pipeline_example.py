import glob
import os

import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keraspipelines import KerasDirectoryFlowPipeline

# Define data sources
src = '/images_data/'
src_full_train = src + 'train_full/'
src_train = src + 'train_split/'
src_val = src + 'val_split/'
src_test = src + 'test/'


# Provide list with classes names
classes = ['A', 'B', 'C']

# Outputs number of samples in each split of data - needed for Keras generators
nb_train_samples = len(glob.glob(src_train + '*/*.*'))
nb_validation_samples = len(glob.glob(src_val + '*/*.*'))
nb_test_samples = len(glob.glob(src_test + '*/*.*'))

print('Number of training samples:', nb_train_samples)
print('Number of validation samples:', nb_validation_samples)
print('Number of test samples:', nb_test_samples)


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
directoryflow_bag_parameters = {
    'model_name': getattr(cnn_models, 'basic_cnn'),
    'model_params': model_parameters,
    'predict_test': True,
    'n_bags': 2,
    'split_size': 0.2,
    'seed': 1337,
    'verbose': True,
    'number_epochs': 1,
    'batch_size': 16,
    'callbacks': model_callbacks,

    'src_dir': os.getcwd(),
    'full_train_dir': src_total,
    'train_dir': src_train,
    'valid_dir': src_val,
    'test_dir': src_test,

    'image_size': (32, 32),
    'classes': classes,
    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_train_samples': nb_train_samples,
    'number_validation_samples': nb_validation_samples,
    'number_test_samples': nb_test_samples,
    'number_test_augmentations': 5,

    'run_save_name': 'basic_cnn_bag_directoryflow',
    'save_statistics': True,
    'save_model': True,
    'output_statistics': True,
}

# Pipeline definition for bagging using .flow_from_directory
bag_pipeline = KerasDirectoryFlowPipeline(model_name=directoryflow_bag_parameters['model_name'],
                                          model_params=directoryflow_bag_parameters['model_params'],
                                          predict_test=directoryflow_bag_parameters['predict_test'],
                                          n_bags=directoryflow_bag_parameters['n_bags'],
                                          split_size=directoryflow_bag_parameters['split_size'],
                                          seed=directoryflow_bag_parameters['seed'],
                                          verbose=directoryflow_bag_parameters['verbose'],
                                          number_epochs=directoryflow_bag_parameters['number_epochs'],
                                          batch_size=directoryflow_bag_parameters['batch_size'],
                                          callbacks=directoryflow_bag_parameters['callbacks'],

                                          src_dir=directoryflow_bag_parameters['src_dir'],
                                          full_train_dir=directoryflow_bag_parameters['full_train_dir'],
                                          train_dir=directoryflow_bag_parameters['train_dir'],
                                          valid_dir=directoryflow_bag_parameters['valid_dir'],
                                          test_dir=directoryflow_bag_parameters['test_dir'],

                                          image_size=directoryflow_bag_parameters['image_size'],
                                          classes=directoryflow_bag_parameters['classes'],
                                          train_datagen=directoryflow_bag_parameters['train_datagen'],
                                          valid_datagen=directoryflow_bag_parameters['valid_datagen'],
                                          test_datagen=directoryflow_bag_parameters['test_datagen'],
                                          number_train_samples=directoryflow_bag_parameters[
                                              'number_train_samples'],
                                          number_validation_samples=directoryflow_bag_parameters[
                                              'number_validation_samples'],
                                          number_test_samples=directoryflow_bag_parameters[
                                              'number_test_samples'],
                                          number_test_augmentations=directoryflow_bag_parameters[
                                              'number_test_augmentations'],

                                          run_save_name=directoryflow_bag_parameters['run_save_name'],
                                          save_statistics=directoryflow_bag_parameters['save_statistics'],
                                          save_model=directoryflow_bag_parameters['save_model'],
                                          output_statistics=directoryflow_bag_parameters['output_statistics'])


# Run bagged model
bagging_model, bagging_preds_test, test_filenames = bag_pipeline.bag_flow_run(
    split_every_bag=True)
