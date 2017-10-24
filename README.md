# Keras Pipelines - 0.1.2

Information about release in Release pages.

Added full training run functionality & automatic OOF predictions saving.

## Idea

--------------------------------------------------------------------------------

'Keras is a high-level neural networks API' according to it's description. It is a great tool which can make Deep Learning a really pleasant experience. It enables user to choose from a whole set of parameters in every step, ranging from model definition to model training and prediction.

For some it is a great merit but for others it may seem hard to be able to combine all the steps into a functioning whole.

There are also situations where we would like to check many models quickly or set up a schedule with varying parameters.

**Keras Pipelines** come to rescue in both situations! They allow experienced users to perform rapid experimentation and easy change of parameters in one place with just a few easy steps or to even adjust the pipelines themselves. For people who start their journey into Deep Learning, **Pipelines** should provide an easy interface to _define, train and predict_ with their models.

### Limitations

When creating this kind of wrapper, already over a high-level API, there had to be many compromises made. Current form is far from perfect, I tried to find a compromise between ease of use and pipeline manipulation possibilities.

For example, due to OOF_train and OOF_test shapes assumptions, KFold currently works for regression and classification out-of-the-box but those shapes can easily be changed to work with different problems, when arrays should be output (like segmentation). In a similar way, you are not able to provide your own data generator, unless you change that in the source code. In most cases using built-in Keras generators should be fine and that was my aim, to cover most of the cases without sacrificing a significant part of simplicity.

### Disclaimer

The project evolved from an idea of simply structuring my own pipelines to achieve a quite clean and reusable form. Then I thought it should be easier for many of you to simply install it as a package, so this is the way I release it but if you simply copy the `.py` **Pipelines** scripts, they should work without a problem too. One helper function is specified in `utils.py`.

## Structure

--------------------------------------------------------------------------------

_KerasPipeline_ is a pipeline definition, where most of the run parameters are being set.

All of them enable user to specify run parameters such as _model definition_, _model parameters_, _number of bags/folds_. Validation split can be either created as a subset of training set or provided by user.

_Seed_ can be set to enable easy model stacking or performance comparison.

Statistics of best epochs for each run can be shown with _output_statistics_ and saved as a text file with _save_history_. Each model can also be saved for further prediction or retraining with _save_model_.

When running KFold/StratifiedKFold _out-of-fold predictions_ for both train and test sets are returned to allow stacking & blending.

## Usage

--------------------------------------------------------------------------------

### Important!

- When using _save_model_ parameter it is assumed, that a folder named `checkpoints` is located in the _src_dir_ directory.

- Models are defined in different file, for example: `cnn_models.py` from which they are being loaded and are being given parameters from _model_params_ if needed.

**Basic example:**

It consists of two basic steps:

- Create your model/model zoo in different file, e.g. `cnn_models.py` in directory, from which you will be able to load them into your script/notebook
- Define model & run parameters and run

**!** More detailed examples with full workflow in `examples/` directory.

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
pipeline_parameters = {
    'model_name': getattr(cnn_models, 'basic_cnn'),
    'predict_test': True,
    'model_callbacks': model_callbacks,
    'number_epochs': 1,
    'batch_size': 16,
    'seed': 1337,
    'shuffle': True,
    'verbose': True,

    'run_save_name': 'basic_cnn_bagging',
    'load_keras_model': False,
    'save_model': True,
    'save_history': True,
    'save_statistics': True,
    'output_statistics': True,

    'src_dir': os.getcwd(),
}
```

Create pipeline definition:

```python
bagging_pipeline = KerasPipeline(model_name=pipeline_parameters['model_name'],
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
                                 )
```

Run your model with current pipeline definition:

```python
bagging_model, bagging_preds_valid, bagging_preds_test = bagging_pipeline.bag_run(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test)
```

This will output a trained model, predictions for validation & test set.

## Installation

--------------------------------------------------------------------------------

`git clone https://github.com/Wrosinski/keraspipelines`

Go into **keraspipelines** directory with `cd keraspipelines`

and run:

`python setup.py install`

Sometimes adding `sudo` prefix may be needed.

### Reference

--------------------------------------------------------------------------------

Library based on [Keras](https://github.com/fchollet/keras) by Francois Chollet.
