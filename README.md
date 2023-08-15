# Arrhenius Chemprop Model

This repository contains a modified version of the original chemprop model described in [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and available at https://github.com/chemprop/chemprop. The original model was modified to incorporate Arrhenius behavior in its predictions to improve its performance on electrolyte prediction tasks, although it can be applied to any system that follows Arrhenius behavior. All arguments necessary to operate the model with Arrhenius behavior are described under [Running Arrhenius Chemprop](#running-arrhenius-chemprop). Portions of the original chemprop readme are kept intact below for reference during implementation.



## Table of Contents

- [Running Arrhenius Chemprop](#running-arrhenius-chemprop)
  * [Data - Arrhenius](#data-arrhenius)
  * [Training and Predicting - Arrhenius](#training-and-predicting-arrhenius)
- [Original Chemprop Documentation](#original-chemprop-documentation)
  * [Requirements](#requirements)
  * [Installation](#installation)
    * [Installation Steps](#installation-steps)
  * [Data](#data)
  * [Training](#training)
    * [Train/Validation/Test Splits](#train-validation-test-splits)
    * [Cross validation](#cross-validation)
    * [Ensembling](#ensembling)
    * [Hyperparameter Optimization](#hyperparameter-optimization)
    * [Aggregation](#aggregation)
    * [Additional Features](#additional-features)
      * [RDKit 2D Features](#rdkit-2d-features)
      * [Custom Features](#custom-features)
      * [Atomic Features](#atomic-features)
  * [Predicting](#predicting)
  * [TensorBoard](#tensorboard)

## Running Arrhenius Chemprop

Before running an Arrhenius Chemprop model it is recommended that you learn to run the original Chemprop. The instructions under [Data](#data) and [Training](#training) in the [Original Chemprop Documentation](#original-chemprop-documentation) section will help you get started with the original model. The following sections describe the modifications needed to run an Arrhenius Chemprop model.

### Data - Arrhenius

When training or predicting on an Arrhenius Chemprop model, data should be input to the model as a CSV file with the following column headers: 
```
smiles, (conductivity or viscosity or other property), temperature
```
This filename should be be used after the `--data_path` flag. Any additional features can be included in a separate .csv with the filename written after the `--features_path` flag.

### Training and Predicting - Arrhenius

To train or predict with an Arrhenius Chemprop model, include the `--arr` flag when running `train.py` and `predict.py`. When predicting, the Arrhenius model will output a prediction for the property of interest at the temperature given in the data file, as well as predictions for the two Arrhenius parameters, <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=E_a">. All three output predictions will appear in the prediction output file.


## Original Chemprop Documentation

### Requirements

For small datasets (~1000 molecules), it is possible to train models within a few minutes on a standard laptop with CPUs only. However, for larger datasets and larger Chemprop models, we recommend using a GPU for significantly faster training.

To use `chemprop` with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

### Installation

Chemprop can either be installed from PyPi via pip or from source (i.e., directly from this git repo). The PyPi version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source.

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).



#### Installation Steps

1. `git clone https://github.com/chemprop/chemprop.git`
2. `cd chemprop`
3. `conda env create -f environment.yml`
4. `conda activate chemprop`
5. `pip install -e .`



### Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```

By default, it is assumed that the SMILES are in the first column (can be changed using `--number_of_molecules`) and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the `--smiles_columns <column_1> ...` and `--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from [MoleculeNet](http://moleculenet.ai/) and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

### Training

To train a model, run:
```
chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is either "classification" or "regression" depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

A full list of available command-line arguments can be found in [chemprop/args.py](https://github.com/chemprop/chemprop/blob/master/chemprop/args.py).

If installed from source, `chemprop_train` can be replaced with `python train.py`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

#### Train/Validation/Test Splits

Our code supports several methods of splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding `--split_type scaffold_balanced`.

**Separate val/test:** If you have separate data files you would like to use as the validation or test set, you can specify them with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`.

Note: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is `--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`.

#### Cross validation

k-fold cross-validation can be run by specifying `--num_folds <k>`. The default is `--num_folds 1`.

#### Ensembling

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>`. The default is `--ensemble_size 1`.

#### Hyperparameter Optimization

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package), which will find the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
chemprop_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>
```
where `<n>` is the number of hyperparameter settings to try and `<config_path>` is the path to a `.json` file where the optimal hyperparameters will be saved.

If installed from source, `chemprop_hyperopt` can be replaced with `python hyperparameter_optimization.py`.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:
```
chemprop_train --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run hyperparameter_optimization.py separately on each split's training and validation data with test held out.

#### Aggregation

By default, the atom-level representations from the message passing network are averaged over all atoms of a molecule to yield a molecule-level representation. Alternatively, the atomic vectors can be summed up (by specifying `--aggregation sum`) or summed up and divided by a constant number N (by specifying `--aggregation norm --aggregation_norm <N>`). A reasonable value for N is usually the average number of atoms per molecule in the dataset of interest. The default is `--aggregation_norm 100`.

#### Additional Features

While the model works very well on its own, especially after hyperparameter optimization, we have seen that adding computed molecule-level features can further improve performance on certain datasets. Features can be added to the model using the `--features_generator <generator>` flag for molecule-level features, or `--atom_descriptors <mode>` for atom-level features, or both.

##### RDKit 2D Features

As a starting point, we recommend using pre-normalized RDKit features by using the `--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the `--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of `rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

The full list of available features for `--features_generator` is as follows. 

`morgan` is binary Morgan fingerprints, radius 2 and 2048 bits.
`morgan_count` is count-based Morgan, radius 2 and 2048 bits.
`rdkit_2d` is an unnormalized version of 200 assorted rdkit descriptors. Full list can be found at the bottom of our paper: https://arxiv.org/pdf/1904.01561.pdf
`rdkit_2d_normalized` is the CDF-normalized version of the 200 rdkit descriptors.

##### Custom Features

If you install from source, you can modify the code to load custom features as follows:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in `chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy `.npy` file or as a `.csv` file, you can load the features by using `--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that `.csv` files must have a header row and the features should be comma-separated with one line per molecule.

##### Atomic Features

Similar to the additional molecular features described above, you can also provide additional atomic features via `--atom_descriptors_path /path/to/features` with valid file formats:
* `.npz` file, where descriptors are saved as 2D array for each molecule in the exact same order as the SMILES strings in your data file.
* `.pkl` / `.pckl` / `.pickle` containing a pandas dataframe with smiles as index and a numpy array of descriptors as columns.
* `.sdf` containing all mol blocks with descriptors as entries.

The order of the descriptors for each atom per molecule must match the ordering of atoms in the RDKit molecule object. Further information on supplying atomic descriptors can be found [here](https://github.com/chemprop/chemprop/releases/tag/v1.1.0). Users must select in which way atom descriptors are used, where the command line option `--atom_descriptors descriptor` concatenates the new features to the embedded atomic features after the D-MPNN, or the option `--atom_descriptors feature` concatenates the features to each atomic feature vector before the D-MPNN, so that they are used during message-passing.


### Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
chemprop_predict --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```
or
```
chemprop_predict --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv
```

If installed from source, `chemprop_predict` can be replaced with `python predict.py`.


### TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, first install TensorFlow with `pip install tensorflow`. Then run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).


