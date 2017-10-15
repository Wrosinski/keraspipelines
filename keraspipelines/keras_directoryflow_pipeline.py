import gc
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.models import load_model
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .utils import copytree


class KerasDirectoryFlowPipeline(object):

    """Creates Keras pipeline with .flow_from_directory method for
        out-of-memory training with data augmentation.

    # Arguments
        model_name: Name of model based on .py file with models definitions.
        model_params: Dict, parameters provided to the model according to it's
            definitions as specified in .py models file.
        predict_test: Boolean, whether to predict on test set.
        n_bags: Int, number of bags to use in bagging run.
        n_folds: Int, number of folds to use in KFold/SKFold run.
        split_size: Float, size of validation split, percent of training data size.
        stratify: Boolean, whether to stratify target classes in KFold run.
        shuffle: Boolean, whether to shuffle data during training & data split.
        user_split: Boolean, whether validation data (X and y) is provided by user.
        seed: Int, random seed number for splits.
        verbose: Boolean, whether to print information about the run.
        number_epochs: Int, number of epochs to train the model for.
        batch_size: Int, batch size for model training and prediction.
        callbacks: List, list of callbacks for the model.
        run_save_name: String, name of run used during checkpoint & run statistics
            saving.
        save_statistics: Boolean, whether to save run statistics.
        save_model: Boolean, whether to save model checkpoints, by default in src_dir + 'checkpoints/'.
        output_statistics: Boolean, whether to show run statistics.
        src_dir: String, working directory for model training & default checkpoints location.
        full_train_dir: String, directory containing full (original) training dataset.
        train_dir: String, directory containing training split data.
        valid_dir: String, directory containing validation split data.
        test_dir: String, directory containing test split data.
        image_size: Tuple, containing image width and height, e.g. (299, 299)
        classes: List, list of names of classes in the data,
            e.g. ['Shark', 'Tuna', 'Whale']
        train_datagen: ImageDataGenerator object specifying data augmentation
            parameters for training set.
        valid_datagen: ImageDataGenerator object specifying data augmentation
            parameters for validation set.
        test_datagen: ImageDataGenerator object specifying data augmentation
            parameters for training set.
        number_train_samples: Int, number of samples in training set,
            given to Keras generator
        number_validation_samples: Int, number of samples in validation set,
            given to Keras generator
        number_test_samples: Int, number of samples in test set,
            given to Keras generator
        number_test_augmentations: Int, number of data augmentations to perform
            during test data prediction.

    """

    def __init__(self, model_name, model_params=None,
                 predict_test=False,
                 n_bags=2, n_folds=5, split_size=0.2,
                 stratify=False, shuffle=True,
                 user_split=False,
                 seed=None, verbose=True,
                 number_epochs=1, batch_size=1, callbacks=None,
                 run_save_name=None, save_statistics=False, save_model=True,
                 output_statistics=True,
                 src_dir=None, full_train_dir=None,
                 train_dir=None, valid_dir=None, test_dir=None,
                 image_size=None,
                 classes=None,
                 train_datagen=None, valid_datagen=None, test_datagen=None,
                 number_train_samples=0, number_validation_samples=0, number_test_samples=0,
                 number_test_augmentations=0):

        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.predict_test = predict_test
        self.n_bags = n_bags
        self.n_folds = n_folds
        self.split_size = split_size
        self.stratify = stratify
        self.shuffle = shuffle
        self.user_split = user_split
        self.seed = seed
        self.verbose = verbose
        self.number_epochs = number_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks if callbacks is not None else []

        self.src_dir = src_dir if src_dir is not None else os.getcwd()
        self.full_train_dir = full_train_dir
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir

        self.image_size = image_size
        self.classes = classes if classes is not None else []
        self.train_datagen = train_datagen
        self.valid_datagen = valid_datagen
        self.test_datagen = test_datagen
        self.number_train_samples = number_train_samples
        self.number_validation_samples = number_validation_samples
        self.number_test_samples = number_test_samples
        self.number_test_augmentations = number_test_augmentations

        self.run_save_name = run_save_name
        self.save_statistics = save_statistics if run_save_name is not None else False
        self.save_model = save_model if run_save_name is not None else False
        self.output_statistics = output_statistics

        self.oof_train = None
        self.oof_test = None

        self.i = 1
        self.start_time = time.time()
        self.checkpoints_dst = self.src_dir + '/checkpoints/'

        self.predictions_valid = []
        self.predictions_test = []
        self.loss_history = []
        self.min_losses = []

    def bag_flow_run(self, split_every_bag=False):
        """Runs Keras bagged run with out-of-memory training with data augmentation.

        # Arguments
            split_every_bag: Boolean, whether to create random training/validation
                split every bag.

        # Returns
            A trained model.
        """

        for bag in range(self.n_bags):
            print('Training on bag:', self.i, '\n')
            model = self.model_name(self.model_params)
            if self.save_statistics:
                os.makedirs('{}{}'.format(
                    self.checkpoints_dst, self.run_save_name), exist_ok=True)

            if self.save_model:
                self.callbacks.append(ModelCheckpoint('{}{}/{}_bag{}.h5'.format(self.checkpoints_dst,
                                                                                self.run_save_name, self.run_save_name,
                                                                                self.i),
                                                      monitor='val_loss',
                                                      verbose=0, save_best_only=True))

            if split_every_bag:
                self.perform_random_validation_split()

            train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                seed=self.seed,
                shuffle=self.shuffle,
                classes=self.classes,
                class_mode='categorical')

            validation_generator = self.valid_datagen.flow_from_directory(
                self.valid_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                seed=self.seed,
                shuffle=self.shuffle,
                classes=self.classes,
                class_mode='categorical')

            history = model.fit_generator(
                train_generator,
                steps_per_epoch=self.number_train_samples / self.batch_size,
                epochs=self.number_epochs,
                validation_data=validation_generator,
                validation_steps=self.number_validation_samples / self.batch_size,
                callbacks=self.callbacks)

            validation_loss = history.history['val_loss']
            self.loss_history.append(validation_loss)
            self.min_losses.append(np.min(validation_loss))
            self.i += 1

        if self.output_statistics:
            self.output_run_statistics()

        if self.predict_test:
            self.predictions_test, test_image_names = self.predict_test_augment()
            return model, self.predictions_test, test_image_names
        return model

    def predict_test_augment(self):
        """Runs Keras bagged model test data prediction with data augmentation.

        # Returns
            2 objects: test data predictions, test filenames
        """

        print('Predicting set from directory: {}'.format(self.test_dir))
        predictions_test_bags = []

        for bag in range(self.n_bags):
            print('Predicting crops for bag: {}'.format(bag + 1))
            model = load_model('{}{}/{}_bag{}.h5'.format(self.checkpoints_dst,
                                                         self.run_save_name,
                                                         self.run_save_name, bag + 1))

            print('Model loaded.', '\n')
            for augment in range(self.number_test_augmentations):
                print('Augmentation number: {}'.format(augment + 1))

                test_generator = self.test_datagen.flow_from_directory(
                    self.test_dir,
                    target_size=self.image_size,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    shuffle=False,
                    classes=None,
                    class_mode='categorical')
                test_image_names = test_generator.filenames

                if augment == 0:
                    predictions_test = model.predict_generator(test_generator,
                                                               self.number_test_samples / self.batch_size)
                else:
                    predictions_test += model.predict_generator(test_generator,
                                                                self.number_test_samples / self.batch_size)
                predictions_test /= self.number_test_augmentations
                predictions_test_bags.append(predictions_test)

        self.predictions_test = np.array(predictions_test_bags).mean(axis=0)
        print('Predictions on test data with augmentation done.')
        return self.predictions_test, test_image_names

    def perform_random_validation_split(self):
        """Performs random split into training and validation sets.
        """

        print('Performing random split with split size: {}'.format(self.split_size))
        os.chdir(self.train_dir)
        os.chdir('../')
        shutil.rmtree(self.train_dir)
        shutil.rmtree(self.valid_dir)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.valid_dir, exist_ok=True)
        copytree(self.full_train_dir, self.train_dir)
        os.chdir(self.train_dir)

        for _class in glob.glob('*'):
            os.mkdir(self.valid_dir + _class)

        train_images_names, valid_images_names = train_test_split(glob.glob(self.train_dir + '*/*.*'),
                                                                  test_size=self.split_size, random_state=self.seed)

        print('Number of training set images: {}, validation set images: {}'.format(len(train_images_names),
                                                                                    len(valid_images_names)))

        for i in range(len(valid_images_names)):
            os.rename(valid_images_names[i], '{}/{}'.format(self.valid_dir,
                                                            '/'.join(valid_images_names[i].split('/')[-2:])))

        return

    def output_run_statistics(self):

        if self.verbose:
            print('Loss statistics for best epoch in current run: \n',
                  'Mean: {}'.format(np.mean(self.min_losses)), '\n',
                  'Minimum: {}'.format(np.min(self.min_losses)), '\n',
                  'Maximum: {}'.format(np.max(self.min_losses)), '\n',
                  'Standard Deviation: {}'.format(np.std(self.min_losses)), '\n')
        if self.save_statistics:
            with open('{}{}/{}_stats.txt'.format(self.checkpoints_dst,
                                                 self.run_save_name, self.run_save_name), 'w') as text_file:
                text_file.write(
                    'Loss statistics for best epoch in current run: \n')
                text_file.write('Minimum: {} \n'.format(
                    np.min(self.min_losses)))
                text_file.write('Maximum: {} \n'.format(
                    np.max(self.min_losses)))
                text_file.write('Mean: {} \n'.format(
                    np.mean(self.min_losses)))
                text_file.write('Standard Deviation: {} \n'.format(
                    np.std(self.min_losses)))
                text_file.write('Seconds it took to train the model: {} \n'.format(
                    time.time() - self.start_time))
        return
