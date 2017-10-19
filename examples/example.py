import glob
import os

import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keraspipelines import KerasPipeline

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Parameters specifying run type:
run_bagging = False
run_kfold = False
flow_augmentation = False  # whether to use real-time data augmentation

# whether to run directory_bag_flow_run using .flow_from_directory method
from_directory = True


number_classes = 10
use_trained_model = True  # whether to load already trained model and predict with it
index_number = 2  # from which bag/fold training or prediction should be started
n_runs = 2  # numer of runs - bags/folds to train/predict for

# name, under which checkpoints and logs will be saved
current_run_name = 'check_run1'


# Following are needed just for directory_bag_flow_run:

# directory structure for directory_bag_flow_run method
src = '/home/w/Projects/Keras_Pipelines/input/ncfm/'
src_full_train = src + 'train/'
src_train = src + 'train_split/'
src_val = src + 'val_split/'
src_test = src + 'test_stg1/'

# Provide list with classes names
classes = [x for x in os.listdir(src_train) if '.' not in x]
print(classes)

# Outputs number of samples in each split of data - needed for Keras generators
nb_train_samples = len(glob.glob(src_train + '*/*.*'))
nb_validation_samples = len(glob.glob(src_val + '*/*.*'))
nb_test_samples = len(glob.glob(src_test + '*/*.*'))

print('Number of training samples:', nb_train_samples)
print('Number of validation samples:', nb_validation_samples)
print('Number of test samples:', nb_test_samples)


# Get example data - CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transform target into binary matrix
y_train = to_categorical(y_train, number_classes)
y_test = to_categorical(y_test, number_classes)

# Transform data into floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


model_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

model_parameters = {
    'img_size': (32, 32, 3),
    'num_classes': number_classes,
}

model_parameters_dir = {
    'img_size': (32, 32, 3),
    'num_classes': len(classes),
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
    'model_name': getattr(cnn_models, 'basic_cnn'),
    'predict_test': True,
    'model_callbacks': model_callbacks,
    'number_epochs': 1,
    'batch_size': 16,
    'seed': 1337,
    'shuffle': True,
    'verbose': True,

    'run_save_name': current_run_name,
    'load_keras_model': use_trained_model,
    'save_model': True,
    'save_history': True,
    'save_statistics': True,
    'output_statistics': True,

    'src_dir': os.getcwd(),
    'full_train_dir': src_full_train,
    'train_dir': src_train,
    'valid_dir': src_val,
    'test_dir': src_test,

    'train_datagen': train_datagen,
    'valid_datagen': valid_datagen,
    'test_datagen': train_datagen,
    'number_test_augmentations': 5,

    'image_size': (32, 32),
    'classes': classes,
    'number_train_samples': nb_train_samples,
    'number_validation_samples': nb_validation_samples,
    'number_test_samples': nb_test_samples,
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
                         full_train_dir=pipeline_parameters['full_train_dir'],
                         train_dir=pipeline_parameters['train_dir'],
                         valid_dir=pipeline_parameters['valid_dir'],
                         test_dir=pipeline_parameters['test_dir'],

                         train_datagen=pipeline_parameters['train_datagen'],
                         valid_datagen=pipeline_parameters['valid_datagen'],
                         test_datagen=pipeline_parameters['test_datagen'],
                         number_test_augmentations=pipeline_parameters['number_test_augmentations'],

                         image_size=pipeline_parameters['image_size'],
                         classes=pipeline_parameters['classes'],
                         number_train_samples=pipeline_parameters['number_train_samples'],
                         number_validation_samples=pipeline_parameters['number_validation_samples'],
                         number_test_samples=pipeline_parameters['number_test_samples'],)


if run_bagging:
    if flow_augmentation:
        pipeline.bag_run(x_train, y_train,
                         X_test=x_test,
                         model_params=model_parameters,
                         n_bags=n_runs,
                         split_size=0.2,
                         user_split=False,
                         index_number=index_number,
                         flow_augment=True)
    else:
        pipeline.bag_run(x_train, y_train,
                         X_test=x_test,
                         model_params=model_parameters,
                         n_bags=n_runs,
                         split_size=0.2,
                         user_split=False,
                         index_number=index_number,
                         flow_augment=False)

if run_kfold:
    if flow_augmentation:
        pipeline.kfold_run(x_train, y_train,
                           X_test=x_test,
                           model_params=model_parameters,
                           n_folds=n_runs,
                           stratify=True,
                           index_number=index_number,
                           flow_augment=True)
    else:
        pipeline.kfold_run(x_train, y_train,
                           X_test=x_test,
                           model_params=model_parameters,
                           n_folds=n_runs,
                           stratify=True,
                           index_number=index_number,
                           flow_augment=False)

if from_directory:
    pipeline.directory_bag_flow_run(
        model_params=model_parameters_dir,
        n_bags=n_runs,
        split_size=0.2,
        split_every_bag=True,
        index_number=index_number,
    )
