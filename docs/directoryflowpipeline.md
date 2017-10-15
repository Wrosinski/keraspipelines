# KerasFlowPipeline

```python
keraspipelines.keras_pipeline.KerasDirectoryFlowPipeline(
               model_name, model_params=None,
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
               number_test_augmentations=0)
```

Creates Keras pipeline with .flow_from_directory method for out-of-memory training with data augmentation.

## Arguments

- model_name: Name of model based on .py file with models definitions.
- model_params: Dict, parameters provided to the model according to it's definitions as specified in .py models file.
- predict_test: Boolean, whether to predict on test set.
- n_bags: Int, number of bags to use in bagging run.
- n_folds: Int, number of folds to use in KFold/SKFold run.
- split_size: Float, size of validation split, percent of training data size.
- stratify: Boolean, whether to stratify target classes in KFold run.
- shuffle: Boolean, whether to shuffle data during training & data split.
- user_split: Boolean, whether validation data (X and y) is provided by user.
- seed: Int, random seed number for splits.
- verbose: Boolean, whether to print information about the run.
- number_epochs: Int, number of epochs to train the model for.
- batch_size: Int, batch size for model training and prediction.
- callbacks: List, list of callbacks for the model.
- run_save_name: String, name of run used during checkpoint & run statistics saving.
- save_statistics: Boolean, whether to save run statistics.
- save_model: Boolean, whether to save model checkpoints, by default in src_dir + 'checkpoints/'.
- output_statistics: Boolean, whether to show run statistics.
- src_dir: String, working directory for model training & default checkpoints location.
- full_train_dir: String, directory containing full (original) training dataset.
- train_dir: String, directory containing training split data.
- valid_dir: String, directory containing validation split data.
- test_dir: String, directory containing test split data.
- image_size: Tuple, containing image width and height, e.g. (299, 299)
- classes: List, list of names of classes in the data, e.g. ['Shark', 'Tuna', 'Whale']
- train_datagen: ImageDataGenerator object specifying data augmentation parameters for training set.
- valid_datagen: ImageDataGenerator object specifying data augmentation parameters for validation set.
- test_datagen: ImageDataGenerator object specifying data augmentation parameters for training set.
- number_train_samples: Int, number of samples in training set, given to Keras generator
- number_validation_samples: Int, number of samples in validation set, given to Keras generator
- number_test_samples: Int, number of samples in test set, given to Keras generator
- number_test_augmentations: Int, number of data augmentations to perform during test data prediction.

--------------------------------------------------------------------------------

## bag_flow_run

```python
bag_flow_run(split_every_bag=False)
```

Runs Keras bagged run with out-of-memory training with data augmentation.

### Arguments

- split_every_bag: Boolean, whether to create random training/validation

  ```
      split every bag.
  ```

### Returns

A trained model.

--------------------------------------------------------------------------------

## predict_test_augment

```python
predict_test_augment()
```

Runs Keras bagged model test data prediction with data augmentation.

### Arguments

- X_test: test dataset
- model: trained model

### Returns

2 objects: test data predictions, test filenames
