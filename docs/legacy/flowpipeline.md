# KerasFlowPipeline

```python
keraspipelines.keras_pipeline.KerasFlowPipeline(
               model_name, model_params=None,
               predict_test=False,
               n_bags=2, n_folds=5, split_size=0.2,
               stratify=False, shuffle=True,
               user_split=False,
               seed=None, verbose=True,
               number_epochs=1, batch_size=1, callbacks=None,
               run_save_name=None, save_statistics=False, save_model=False,
               output_statistics=True,
               src_dir=None,
               train_datagen=None, valid_datagen=None, test_datagen=None,
               number_test_augmentations=0)
```

Creates Keras pipeline with .flow method for data augmentation.

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
- train_datagen: ImageDataGenerator object specifying data augmentation parameters for training set.
- valid_datagen: ImageDataGenerator object specifying data augmentation parameters for validation set.
- test_datagen: ImageDataGenerator object specifying data augmentation parameters for training set.
- number_test_augmentations: Int, number of data augmentations to perform during test data prediction.

--------------------------------------------------------------------------------

## bag_flow_run

```python
bag_flow_run(X_train, y_train,
             X_valid=None, y_valid=None,
             X_test=None, y_test=None)
```

Runs bagging using Keras pipeline with data augmentation.

### Arguments

- X_train: training set data.
- y_train: training set labels.
- X_valid: validation set data.
- y_valid: validation set labels.
- X_test: test set data.
- y_test: test set labels.

### Returns

- When predict_set: 3 objects: a trained model, validation predictions, test predictions
- When predict_set == False: 2 objects: a trained model, validation predictions

--------------------------------------------------------------------------------

## kf_flow_run

```python
kf_flow_run(X_train, y_train,
            X_test=None, y_test=None)
```

### Arguments

- X_train: training set data.
- y_train: training set labels.
- X_test: test set data.
- y_test: test set labels.

### Returns

- When predict_set: 3 objects: a trained model, out-of-fold training predictions, out-of-fold test predictions

- When predict_set == False: 2 objects: a trained model, out-of-fold training predictions

## predict_test_augment

```python
predict_test_augment(X_test, model)
```

Runs Keras bagged model test data prediction with data augmentation.

### Arguments

- X_test: test dataset
- model: trained model

### Returns

Test data predictions
