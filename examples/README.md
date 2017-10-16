# Examples Overview

## File structure

To enable easy change & parametrization of used models, it is assumed that model definitions are located in a different file, for example `cnn_models.py` and each model can be fed with a `dict` specifying it's parameters.

In the same directory you should have your script, for example `pipeline_example.py`, into which your model is loaded. Afterwards run parameters can be specified in another dictionary.

### KerasPipeline example

Example showing a basic workflow with KerasPipeline.

Assumes that you have a full training set to be split into train & validation subsets and a test set to predict.

### KerasFlowPipeline example

Example showing a workflow with KerasFlowPipeline, where data augmentation for images is included in training and at test set prediction.

Assumes, in a way similar to the **KerasPipeline** example, that you have a full training set to be split into train & validation subsets and a test set to predict.

### KerasDirectoryFlowPipeline example

Currently this is not a working example because I had a problem with finding a dataset having proper folder structure. To use [`.flow_from_directory`](https://keras.io/preprocessing/image/) Keras method, each data split assumes a structure, where all of images belonging to a certain class are in a folder with it's class name, e.g.

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
