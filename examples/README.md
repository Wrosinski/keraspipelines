# Example Tutorial

## File structure

To enable easy change & parametrization of used models, it is assumed that model definitions are located in a different file, for example `cnn_models.py` and each model can be fed with a `dict` object specifying it's parameters.

In the same directory you should have your script, for example `pipeline_example.py`, into which your model is loaded. Afterwards run parameters can be specified in another dictionary.

### Workflow - KerasPipeline.bag_run & .kfold_run

If we are willing to train a model in-memory, we can either use bagging or KFold/StratifiedKFold. What we will need is just our training & test data. In addition to that, various parameters, which will indicate what will happen during the run, should be specified.

For out-of-memory training using `.flow_from_directory` look below, at **KerasPipeline.directory_bag_flow_run** part.

Let's start with loading needed libraries and load the KerasPipeline object among them:

```python
import glob
import os
import cnn_models
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keraspipelines import KerasPipeline
```

Now let's define our run parameters, whether we will use bagging or KFold, will data augmentation be used or maybe if we prefer to just predict with our trained model.

```python
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
```

Then we proceed to data processing - load/prepare your data and encode your labels into binary (One-Hot) matrix. Here I use most basic example from Keras datasets. Data should be provided as numpy arrays, exactly like for Keras itself. `to_categorical` is very handy Keras processing method, which enables to process labels from categorical encoding into One-Hot encoding.

```python
# Get example data - CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Transform target into binary matrix
y_train = to_categorical(y_train, number_classes)
y_test = to_categorical(y_test, number_classes)

# Transform data into floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
```

Next step consists of providing callbacks for your model. Here only training callbacks should be given as `ModelCheckpoint` and `CSVLogger` can be appended to the list if `save_model` or `save_history` pipeline parameters are set to True. Those callbacks are defined in the Pipeline object, as we would like to have each bag/fold saved and logs for it written.

Model parameters should also be given if needed, this depends on the model definition in `cnn_models.py`. Our model definition needs input image size and number of classes as parameters, so we give exactly those ones.

```python
model_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

model_parameters = {
    'img_size': (32, 32, 3),
    'num_classes': number_classes,
}
```

If we would like to randomly augment our data during training and during test set prediction, we will need to set `flow_augment` argument when starting our run to True. For that we need `ImageDataGenerator` objects which will set data augmentation parameters for all subsets of our data. Here, I will specify one for training/test (same one will be used) and one for validation set.

```python
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
```

Now, let's get to the sweet part, defining our Pipeline parameters and setting them for a run. The most comfortable way, at least for me, is to set all parameters in a dictionary and feed them into the object afterwards. This enables easy change of a specified parameter, even one defined at the beginning of a script. Not all parameters are needed for a basic one.

There is a subset, which is required for using augmentation:

```python
'train_datagen': train_datagen,
'valid_datagen': valid_datagen,
'test_datagen': train_datagen,
'number_test_augmentations': 5,
```

And another subset, needed for `directory_bag_flow_run`:

```python
'full_train_dir': src_full_train,
'train_dir': src_train,
'valid_dir': src_val,
'test_dir': src_test,

'image_size': (32, 32),
'classes': classes,
'number_train_samples': nb_train_samples,
'number_validation_samples': nb_validation_samples,
'number_test_samples': nb_test_samples,
```

These subsets can be ignored when setting up a most basic run. With all possible parameters set, our dictionary will look like this:

```python
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
```

Okay, great, we have our pipeline all set up now! Let's create a `KerasPipeline` object, which will be fed all those parameters:

```python
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
```

The only thing left is to start the run:

```python
# Bagging run
trained_model, val_pred, test_pred = pipeline.bag_run(x_train, y_train,
                                                      X_test=x_test,
                                                      model_params=model_parameters,
                                                      n_bags=n_runs,
                                                      split_size=0.2,
                                                      user_split=False,
                                                      index_number=index_number,
                                                      flow_augment=True)
# StratifiedKFold run
trained_model, oof_train, oof_test = pipeline.kfold_run(x_train, y_train,
                                                        X_test=x_test,
                                                        model_params=model_parameters,
                                                        n_folds=n_runs,
                                                        stratify=True,
                                                        index_number=index_number,
                                                        flow_augment=True)
```

### Workflow - KerasPipeline.directory_bag_flow_run

This method requires a specific folder structure. Currently it is not an example working as is because I had a problem with finding a dataset having proper folder structure. To use [`.flow_from_directory`](https://keras.io/preprocessing/image/) Keras method, each data split assumes a structure, where all of images belonging to a certain class are in a folder with it's class name, e.g.

```bash
├── train
│   ├── Shark
│   ├── Tuna
│   ├── Whale
```

Also, because test data usually has no classes specified, it's folder structure should be nested, like `test/test/*.jpg`.

Full folder structure should look like this:

```bash
├── test
│   └── test
├── train
│   ├── Shark
│   ├── Tuna
│   ├── Whale
├── train_split
│   ├── Shark
│   ├── Tuna
│   ├── Whale
└── val_split
    ├── Shark
    ├── Tuna
    ├── Whale
```

- `train` folder is where the full training data is. It will be split into training/validation sets.
- `train_split` is where the part of data designed for model training is.
- `val_split` is where the part of data designed for model validation is.
- `test` contains test data. It is nested, because `classes` argument is set to `None` when using [`.flow_from_directory`](https://keras.io/preprocessing/image/).

#### Differences

For this type of run we need to specify additional parameters, as needed for `.flow_from_directory` Keras method:

```python
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
```

We omit steps connected with data processing but still provide all needed parameters, such as `model_callbacks` list, `model_parameters` dictionary if needed, `ImageDataGenerator` objects for data augmentation and the `KerasPipeline` definition itself.

Then we define the run:

```python
model, test_pred, test_filenames = pipeline.directory_bag_flow_run(
                                    model_params=model_parameters_dir,
                                    n_bags=n_runs,
                                    split_size=0.2,
                                    split_every_bag=True,
                                    index_number=index_number,
                                    )
```
