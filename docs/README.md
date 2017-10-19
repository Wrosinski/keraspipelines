# Documentation

## KerasPipeline

```python
keraspipelines.KerasPipeline(
    model_name,
    predict_test=False,
    model_callbacks=None,
    number_epochs=1,
    batch_size=1,
    seed=None,
    shuffle=True,
    verbose=True,
    run_save_name=None,
    load_keras_model=False,
    save_model=True,
    save_history=False,
    save_statistics=False,
    output_statistics=True,
    src_dir=None,
    full_train_dir=None,
    train_dir=None,
    valid_dir=None,
    test_dir=None,
    train_datagen=None,
    valid_datagen=None,
    test_datagen=None,
    number_test_augmentations=0,
    image_size=None,
    classes=None,
    number_train_samples=0,
    number_validation_samples=0,
    number_test_samples=0,
)
```

Creates and defines KerasPipeline object. Enables in-memory and out-of-memory model training using bagging and KFold/StratifiedKFold.

### Arguments

- model_name: (String) Name of model based on .py file with models definitions.
- predict_test: (Boolean), whether to predict on test set.
- model_callbacks: (List), list of callbacks for the model.
- number_epochs: (Int), number of epochs to train the model for.
- batch_size: (Int), batch size for model training and prediction.

- seed: (Int), random seed number for splits.

- shuffle: (Boolean), whether to shuffle data during training & data split.

- verbose: (Boolean,) whether to print information about the run.

- run_save_name: (String), name of run used during checkpoint & run statistics saving.

- load_keras_model: (Boolean), whether to load already trained model.

- save_model: (Boolean), whether to save model checkpoints, by default in src_dir + 'checkpoints/'.

- save_history: (Boolean), whether to save history of a model in CSV file.
- save_statistics: (Boolean), whether to save run statistics.
- output_statistics: (Boolean), whether to show run statistics.

- src_dir: (String), working directory for model training & default checkpoints location.

### Optional, use when running KerasPipeline.directory_bag_flow_run:

- full_train_dir: (String), directory containing full (original) training dataset.
- train_dir: (String), directory containing training split data.
- valid_dir: (String), directory containing validation split data.
- test_dir: (String), directory containing test split data.

### Optional, use when running with flow_augment:

- train_datagen: (ImageDataGenerator object), specifying data augmentation parameters for training set.
- valid_datagen: (ImageDataGenerator object), specifying data augmentation parameters for validation set.
- test_datagen: (ImageDataGenerator object), specifying data augmentation parameters for training set.
- number_test_augmentations: (Int), number of data augmentations to perform during test data prediction.

### Optional, use when running KerasPipeline.directory_bag_flow_run:

- image_size: (Tuple), containing image width and height, e.g. (299, 299)
- classes: (List), list of names of classes in the data, e.g. ['Shark', 'Tuna', 'Whale']
- number_train_samples: (Int), number of samples in training set, given to Keras generator
- number_validation_samples: (Int), number of samples in validation set, given to Keras generator
- number_test_samples: (Int), number of samples in test set, given to Keras generator

--------------------------------------------------------------------------------

### KerasPipeline.bag_run

```python
keraspipelines.KerasPipeline.bag_run(
    X_train, y_train,
    X_valid=None, y_valid=None,
    X_test=None, y_test=None,
    model_params=None,
    n_bags=2,
    split_size=0.2,
    user_split=False,
    index_number=None,
    flow_augment=False,
)
```

Bagging run.

#### Arguments

- X_train: (numpy array), training set.
- y_train: (numpy array), training set labels.
- X_valid: (numpy array), validation set.
- y_valid: (numpy array), validation set labels.
- X_test: (numpy array), test set.
- y_test: (numpy array), test set labels.
- model_params: (Dict), dictionary of model parameters.
- n_bags: (Int), number of bags used in training.
- split_size: (Float), size of validation split in percentage of training set size.
- user_split: (Boolean), whether validation set is provided by user, not created.
- index_number: (Int), index specifying from which bag should training or prediction be started.
- flow_augment: (Boolean), whether to use data augmentation during test and prediction.

#### Returns

- model: (Keras model), trained model for last bag.
- predictions_valid: (numpy array), array for validation set predictions.

if predict_test additionally:

- predictions_test: (numpy array), array for test set predictions.

--------------------------------------------------------------------------------

### KerasPipeline.kfold_run

```python
keraspipelines.KerasPipeline.kfold_run(
    X_train, y_train,
    X_test=None, y_test=None,
    model_params=None,
    n_folds=5,
    stratify=False,
    index_number=None,
    flow_augment=False,
)
```

KFold/StratifiedKFold run.

#### Arguments

- X_train: (numpy array), training set.
- y_train: (numpy array), training set labels.
- X_test: (numpy array), test set.
- y_test: (numpy array), test set labels.
- model_params: (Dict), dictionary of model parameters.
- n_folds: (Int), number of folds used in training.
- stratify: (Boolean), whether fold split should be stratified according to labels distribution.
- index_number: (Int), index specifying from which bag should training or prediction be started.
- flow_augment: (Boolean), whether to use data augmentation during test and prediction.

#### Returns

- model: (Keras model), trained model for last fold.
- oof_train: (numpy array), array with out-of-fold training set predictions.

if predict_test additionally:

- oof_test: (numpy array), array with out-of-fold test set predictions.

--------------------------------------------------------------------------------

### KerasPipeline.directory_bag_flow_run

```python
keraspipelines.KerasPipeline.directory_bag_flow_run(
    model_params=None,
    n_bags=2,
    split_size=0.2,
    split_every_bag=False,
    index_number=None)
```

Bagging run using .flow_from_directory for loading data from directly from disk.

#### Arguments

- model_params: (Dict), dictionary of model parameters.
- n_bags: (Int), number of bags used in training.
- split_size: (Float), size of validation split in percentage of training set size.
- split_every_bag: (Boolean), whether to create random validation split for every bag.
- index_number: (Int), index specifying from which bag should training or prediction be started.

#### Returns

- model: (Keras model), trained model for last bag.

if predict_test additionally:

- predictions_test: (numpy array), array for test set predictions.
- test_image_names: (List), list with test filenames.

--------------------------------------------------------------------------------

### KerasPipeline.flow_predict_test_augment

```python
keraspipelines.KerasPipeline.flow_predict_test_augment(
    X_test,
    model)
```

Runs Keras bagged model test data prediction with data augmentation.

#### Arguments

- X_test: (numpy array), test dataset
- model: (Keras model), trained model

#### Returns

- predictions_test: (numpy array), test data predictions

--------------------------------------------------------------------------------

### KerasPipeline.directory_predict_test_augment

```python
keraspipelines.KerasPipeline.directory_predict_test_augment(
    n_bags,
    index_number)
```

Runs Keras bagged model test data prediction with data augmentation using .flow_from_directory method.

#### Arguments

- n_bags: (int), number of bags to predict on
- index_number: (Int), index specifying from which bag should training or prediction be started.

#### Returns

- predictions_test: (numpy array), test data predictions
- test_image_names: (List), test filenames

--------------------------------------------------------------------------------

## Helper functions

### KerasPipeline.perform_random_validation_split

```python
keraspipelines.KerasPipeline.perform_random_validation_split(split_size)
```

Performs random split into training and validation sets when loading data from directories.

### Arguments

- split_size: (float), size of validation set in percents

--------------------------------------------------------------------------------

### KerasPipeline.output_run_statistics

```python
keraspipelines.KerasPipeline.output_run_statistics(prefix)
```

Saves statistics for each best epoch in bag/fold in current run.

### Arguments

- prefix: (String), specifies prefix for filename - 'bag', 'bag_dir', 'fold'

--------------------------------------------------------------------------------

### KerasPipeline.load_trained_model

```python
keraspipelines.KerasPipeline.load_trained_model(prefix)
```

Loads trained model based on it's checkpoint.

### Arguments

- prefix: (String), specifies prefix for filename - 'bag', 'bag_dir', 'fold'

### Returns

- model: (Keras model), loaded trained keras model

--------------------------------------------------------------------------------

### KerasPipeline.callbacks_append_checkpoint

```python
keraspipelines.KerasPipeline.callbacks_append_checkpoint(prefix)
```

Appends checkpoint saving to model callbacks.

### Arguments

- prefix: (String), specifies prefix for filename - 'bag', 'bag_dir', 'fold'

--------------------------------------------------------------------------------

### KerasPipeline.callbacks_append_logger

```python
keraspipelines.KerasPipeline.callbacks_append_logger(prefix)
```

Appends CSV logging to model callbacks.

### Arguments

- prefix: (String), specifies prefix for filename - 'bag', 'bag_dir', 'fold'

--------------------------------------------------------------------------------

### KerasPipeline.callbacks_append_tensorboard - Not working

```python
keraspipelines.KerasPipeline.callbacks_append_tensorboard(prefix)
```

Appends Tensorboard logging. Currently not working due to a Keras bug.

### Arguments

- prefix: (String), specifies prefix for filename - 'bag', 'bag_dir', 'fold'
