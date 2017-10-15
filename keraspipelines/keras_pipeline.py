import gc
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class KerasPipeline(object):

    """Creates standard Keras pipeline.

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

    """

    def __init__(self, model_name, model_params=None,
                 predict_test=False,
                 n_bags=2, n_folds=5, split_size=0.2,
                 stratify=False, shuffle=True,
                 user_split=False,
                 seed=None, verbose=True,
                 number_epochs=1, batch_size=1, callbacks=None,
                 run_save_name=None, save_statistics=False, save_model=False,
                 output_statistics=True,
                 src_dir=None):

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

    def bag_run(self,
                X_train, y_train,
                X_valid=None, y_valid=None,
                X_test=None, y_test=None):
        """Runs bagging using standard Keras pipeline.

        # Arguments
            X_train: training set data.
            y_train: training set labels.
            X_valid: validation set data.
            y_valid: validation set labels.
            X_test: test set data.
            y_test: test set labels.

        # Returns
            When predict_set:
                3 objects: a trained model, validation predictions, test predictions
            When predict_set == False:
                2 objects: a trained model, validation predictions
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

            if X_valid is not None and y_valid is not None and self.user_split:

                print('Validating on subset of data specified by user.')
                history = model.fit(X_train, y_train, verbose=self.verbose,
                                    batch_size=self.batch_size, epochs=self.number_epochs,
                                    validation_data=(X_valid, y_valid),
                                    callbacks=self.callbacks)
            else:
                if self.seed:
                    print('Splitting data - validation split size: {}, split seed: {}'.format(
                        self.split_size, self.seed))
                else:
                    print('Splitting data - validation split size: {}, seed not set.'.format(
                        self.split_size))

                X_tr, X_valid, y_tr, y_valid = train_test_split(
                    X_train, y_train, test_size=self.split_size, random_state=self.seed)

                history = model.fit(X_tr, y_tr, verbose=self.verbose,
                                    batch_size=self.batch_size, epochs=self.number_epochs,
                                    validation_data=(X_valid, y_valid),
                                    callbacks=self.callbacks)

            print('Predicting on validation data.')
            self.predictions_valid.append(model.predict(
                X_valid, batch_size=self.batch_size))
            if self.verbose:
                print('Validation split - standard deviation for original target values: {} \n \
                 for predicted target values: {} \n \n'.format(
                    np.std(y_valid), np.std(self.predictions_valid)))

            if self.predict_test and X_test is not None:
                print('Predicting on test data.')
                self.predictions_test.append(model.predict(
                    X_test, batch_size=self.batch_size))

            validation_loss = history.history['val_loss']
            self.loss_history.append(validation_loss)
            self.min_losses.append(np.min(validation_loss))
            self.i += 1

        if self.output_statistics:
            self.output_run_statistics()

        if self.predict_test and X_test is not None:
            return model, np.array(self.predictions_valid), np.array(self.predictions_test)
        return model, np.array(self.predictions_valid)

    def kfold_run(self,
                  X_train, y_train,
                  X_test=None, y_test=None):
        """Runs KFold/StratifiedKFold using standard Keras pipeline.

        # Arguments
            X_train: training set data.
            y_train: training set labels.
            X_test: test set data.
            y_test: test set labels.

        # Returns
            When predict_set:
                3 objects: a trained model, out-of-fold training predictions,
                    out-of-fold test predictions
            When predict_set == False:
                2 objects: a trained model, out-of-fold training predictions
        """

        if len(y_train.shape) == 1:
            y_train = y_train.reshape((y_train.shape[0], 1))

        self.oof_train = np.zeros(y_train.shape + (1,))
        print('OOF train predictions shape: {}'.format(self.oof_train.shape))

        if X_test is not None:
            self.oof_test = np.zeros(
                (X_test.shape[0],) + y_train.shape[1:] + (self.n_folds,))
            print('OOF test predictions shape: {}'.format(self.oof_test.shape))

        if self.stratify and self.oof_train.shape[-2] != 1:
            print(
                'To use StratifiedKFold please provide categorically encoded labels, not One-Hot encoded. \
                \n Reversing OH encoding now.')
            y_train_split = pd.DataFrame(y_train).idxmax(axis=1).values
            print('Labels after reversed encoding:', y_train_split[:10])
            kf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
        else:
            kf = KFold(
                n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.seed)
            y_train_split = y_train

        for train_index, test_index in kf.split(X_train, y_train_split):
            print('Training on fold:', self.i, '\n')

            X_tr, X_val = X_train[train_index], X_train[test_index]
            y_tr, y_val = y_train[train_index], y_train[test_index]

            model = self.model_name(self.model_params)

            if self.save_statistics:
                os.makedirs('{}{}'.format(
                    self.checkpoints_dst, self.run_save_name), exist_ok=True)

            if self.save_model:
                self.callbacks.append(ModelCheckpoint('{}{}/{}_fold{}.h5'.format(self.checkpoints_dst,
                                                                                 self.run_save_name, self.run_save_name,
                                                                                 self.i),
                                                      monitor='val_loss',
                                                      verbose=0, save_best_only=True))

            history = model.fit(X_tr, y_tr, verbose=self.verbose,
                                batch_size=self.batch_size, epochs=self.number_epochs,
                                validation_data=(X_val, y_val),
                                callbacks=self.callbacks)

            print('Predicting on validation data.')
            self.oof_train[test_index, :, 0] = model.predict(
                X_val, batch_size=self.batch_size)
            if self.verbose:
                print('Validation split - standard deviation for original target values: {} \n \
                for predicted target values: {} \n \n'.format(
                    np.std(y_val), np.std(self.oof_train[test_index, :])))

            if self.predict_test and X_test is not None:
                print('Predicting on test data.')
                self.oof_test[:, :, self.i - 1] = model.predict(
                    X_test, batch_size=self.batch_size)

            validation_loss = history.history['val_loss']
            self.loss_history.append(validation_loss)
            self.min_losses.append(np.min(validation_loss))
            self.i += 1

        if self.output_statistics:
            self.output_run_statistics()

        if self.predict_test and X_test is not None:
            return model, np.array(self.oof_train), np.array(self.oof_test)
        return model, np.array(self.oof_train).mean(axis=-1)

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
