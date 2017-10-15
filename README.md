# Keras Pipelines

## Idea

'Keras is a high-level neural networks API' according to it's description. It is a great tool which can make Deep Learning a really pleasant experience. It enables user to choose from a whole set of parameters in every step, ranging from model definition to model training and prediction.

For some it is a great merit but for others it may seem hard to be able to combine all the steps into a functioning whole.

There are also situations where we would like to check many models quickly or set up a schedule with varying parameters.

**Keras Pipelines** come to rescue in both situations! They allow experienced users to perform rapid experimentation and easy change of parameters in one place with just a few easy steps or to even adjust the pipelines themselves. For people who start their journey into Deep Learning, **Pipelines** should provide an easy interface to _define, train and predict_ with their models.

## Structure

Three pipelines are provided.

All of them enable user to specify run parameters such as _model definition_, _model parameters_, _number of bags/folds_, _validation split size_, _stratification_ in case of KFold. Validation split can be either created as a subset of training set or provided by user.

_Seed_ can be set to enable easy model stacking or performance comparison.

Statistics of best epochs for each run can be shown with _output_statistics_ and saved as a text file with _save_history_. Each model can also be saved for further prediction or retraining with _save_model_.

When running KFold/StratifiedKFold _out-of-fold predictions_ for both train and test sets are returned to allow stacking & blending.

_KerasPipeline_ and _KerasFlowPipeline_ allow to run either **bagged** run or **KFold/StratifiedKFold** run. _KerasDirectoryFlowPipeline_ allows only a **bagged** run as of now. It would be possible to integrate KF into it but that will require a different approach for the data splitting part.

### _KerasPipeline_

Basic pipeline, specify all the parameters, train a model with either bagging or KFold/StratifiedKFold, output model + predictions for validation & test datasets.

### _KerasFlowPipeline_

Similar to _KerasPipeline_ but with real-time data augmentation during model training and test data augmentation during predictions with specified _ImageDataGenerator_ parameters.

### _KerasDirectoryFlowPipeline_

A different pipeline, based on [_.flow_from_directory_](https://keras.io/preprocessing/image/) method which enables **out-of-memory** training, where data is loaded in batches from folders specified in the run parameters. Training & test data augmentation implemented.

Models should be saved into `checkpoints` in working directory during training in order to be loaded afterwards for test data prediction.

A method is provided in the class definition which enables creation of random splits for every bag (with _seed_ parameter either set or not.)

Currently only supports model bagging.

## Usage

### Important!

- When using _save_model_ pipeline parameter it is assumed, that there is a folder named `checkpoints` in the _src_dir_ directory.

- Models are defined in different file, for example: `cnn_models.py` from which they are being loaded and are being given parameters from _model_params_.

**Basic _KerasPipeline_ example:**

It consists of two basic steps:

- Create your model/model zoo in different file, e.g. `cnn_models.py` in directory, from which you will be able to load them into your script/notebook
- Define model & run parameters and run

More detailed examples with full workflow in `examples/` directory.

#### Step 1 - Define your model

Specify your basic CNN model and save it into `cnn_models.py` file in the working directory:

```python
def basic_cnn(params):

    input_layer = Input(params['img_size'])
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(params['num_classes'])(x)
    output_layer = Activation('softmax')(x)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
```

#### Step 2 - Define your parameters & run

Then, in your script:

Specify your model parameters:

```python
model_parameters = {
    'img_size': (32, 32, 3),
    'num_classes': number_classes,
}
```

Set your run parameters:

```python
bag_parameters = {
    'model_name': getattr(cnn_models, 'basic_cnn'), # <- this loads basic_cnn model definition from cnn_models.py based on it's function name
    'model_params': model_parameters,
    'predict_test': True,
    'n_bags': 2,
    'split_size': 0.2,
    'seed': 1337,
    'user_split': False,
    'verbose': True,
    'number_epochs': 1,
    'batch_size': 256,

    'src_dir': os.getcwd(),

    'run_save_name': 'basic_cnn_bagging',
    'save_history': True,
    'output_statistics': True,

    'X_train': x_train,
    'y_train': y_train,
    'X_test': x_test,
}
```

Create pipeline definition:

```python
bag_pipeline = KerasPipeline(model_name=bag_parameters['model_name'],
                             model_params=bag_parameters['model_params'],
                             predict_test=bag_parameters['predict_test'],
                             n_bags=bag_parameters['n_bags'],
                             split_size=bag_parameters['split_size'],
                             number_epochs=bag_parameters['number_epochs'],
                             batch_size=bag_parameters['batch_size'],
                             seed=bag_parameters['seed'],
                             user_split=bag_parameters['user_split'],
                             run_save_name=bag_parameters['run_save_name'],
                             save_history=bag_parameters['save_history'],
                             output_statistics=bag_parameters['output_statistics'])
```

Run your model with current pipeline definition:

```python
bagging_model, bagging_preds_valid, bagging_preds_test = bag_pipeline.bag_run(
    X_train=bag_parameters['X_train'],
    y_train=bag_parameters['y_train'],
    X_test=bag_parameters['X_test'])
```

This will output a trained model, predictions for validation & test set.

## Installation

`git clone https://github.com/Wrosinski/keraspipelines`

Go into `keraspipelines` directory with `cd keraspipelines` and run:

`python setup.py install`

Sometimes adding `sudo` prefix may be needed.
