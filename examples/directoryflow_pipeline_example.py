import os

import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keraspipelines import KerasDirectoryFlowPipeline

# Define data sources
src = '/home/w/Projects/Keras_Pipelines/input/ncfm/'
src_full_train = src + 'train_full/'
src_train = src + 'train_split/'
src_val = src + 'val_split/'
src_test = src + 'test/'


# Provide list with classes names
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# Provide number of samples in each split of data - needed for Keras generators
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
    'size': (224, 224, 3),
    'num_classes': 10,
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

    'image_size': (299, 299),
    'classes': classes,
    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_train_samples': nb_train_samples,
    'number_validation_samples': nb_validation_samples,
    'number_test_samples': 1000,
    'number_test_augmentations': 5,

    'run_save_name': 'resnet50_bag_directoryflow',
    'save_history': True,
    'save_model': True,
    'output_statistics': True,
}

# Pipeline definition for bagging using .flow_from_directory
bag_pipeline = KerasDirectoryFlowPipeline(model_name=run_parameters['model_name'],
                                          model_params=run_parameters['model_params'],
                                          predict_test=run_parameters['predict_test'],
                                          n_bags=run_parameters['n_bags'],
                                          split_size=run_parameters['split_size'],
                                          seed=run_parameters['seed'],
                                          verbose=run_parameters['verbose'],
                                          number_epochs=run_parameters['number_epochs'],
                                          batch_size=run_parameters['batch_size'],
                                          callbacks=run_parameters['callbacks'],

                                          src_dir=run_parameters['src_dir'],
                                          full_train_dir=run_parameters['full_train_dir'],
                                          train_dir=run_parameters['train_dir'],
                                          valid_dir=run_parameters['valid_dir'],
                                          test_dir=run_parameters['test_dir'],

                                          image_size=run_parameters['image_size'],
                                          classes=run_parameters['classes'],
                                          train_datagen=run_parameters['train_datagen'],
                                          valid_datagen=run_parameters['valid_datagen'],
                                          test_datagen=run_parameters['test_datagen'],
                                          number_train_samples=run_parameters['number_train_samples'],
                                          number_validation_samples=run_parameters['number_validation_samples'],
                                          number_test_samples=run_parameters['number_test_samples'],
                                          number_test_augmentations=run_parameters['number_test_augmentations'],

                                          run_save_name=run_parameters['run_save_name'],
                                          save_history=run_parameters['save_history'],
                                          save_model=run_parameters['save_model'],
                                          output_statistics=run_parameters['output_statistics'])

# Run bagged model
bagging_model, bagging_preds_test, test_filenames = bag_pipeline.bag_flow_run(
    split_every_bag=True)
