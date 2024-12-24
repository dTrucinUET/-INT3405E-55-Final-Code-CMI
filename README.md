## INT3405E 55 Final project code
Members: 
1. Tăng Vĩnh Hà (leader) - 22028129 - INT3405E 55
2. Đặng Đào Xuân Trúc 	- 22028179 - INT3405E 55
3. Lê Minh Đức 			- 22028267 - INT3405E 56

This is the repository for the project of the Kaggle competition: Child Mind Institute — Problematic Internet Use of group 23 - class INT3405E 55

## Final overall pipeline

### Data preprocessing

We decided to discard all samples with missing labels in training dataset. Justification for this is in our final slide. We then impute missing features with `SimpleImputer` from `sklearn`.

### Machine learning algorithm 
We first began by trying basic machine learning model. The function `TrainML` is designed to so we can easily exprience with many algorithms. 
- The input is: `model_class, X, y, test_data`. 
    - `model_class`: can be any algorithm from any libraries - as long as it has a `fit` function
    - `X`: for `model_class` that can handle null values like `XGBoost`, `CatBoost`, `LGBM`; `X` can contain null values, otherwise it must be preprocessed before passing into this function. Similarly for `test_data`. Tyep of `X` is `pandas` dataframe.
    -  `y`: label of training samples. Type of `y` is a single `pandas` column
- The output is: 
    - `submission`: submission (dataframe form with predictions and ID) of the test set. 
    - `oof_non_rounded`: the raw prediction of model for `test_data` (every algorithm is in regression setting. Justification for the choice of regression setting can be found in our final slide).
    - `oof_tuned`: final predictions after applying optimized thresholds
    - `y`: return input y again - this is added mainly for developing models and debug process.
    - `optimized_thresholds`: the threshold to convert regression predictions into 4 discrete classes (0,1,2,3). This is optimized by `minimize` function of `spicy` library. 

- We experimented different algorithm:  from tree-based algorithm (`XGBoost`, `CatBoost`, `LGBM`) from libraries with the same name  and  `SVR` from `sklearn` (support vector machine in regression setting). `VotingRegressor` for ensembling models was also experimented. Simply by defining a model and pass into `model_class` argument. 


### Model improvement

We improved our models for this competition as progress above. 
1. Hyperparameters tuning: hyperparameters of machine learning models in every pipeline/setting can be tuned easily using `Optuna` library - but due to the large number of experiments/setting we have conducted, we only an example of hyperparameters tuning of our source code in `pipeline_tuning`:
    - Please refer to the `pipeline_tuning` file for the def `objective_sub1` function. `trial.suggest_*` functions in the code (e.g., `trial.suggest_float, trial.suggest_int`) define the hyperparameter space. `Optuna` use Bayesian optimization to searches the hyperparameter space. This a probabilistic approach based on prior evaluations. You can customize this function for your goal but the return output must be a number, the direction of maximizing that score can be `minimize` or `maximize`. 
2. Feature engineering: 
    - Tabular data: 
        - First method is simply `SelectKBest` (choosing features with high correlated with the target label). It is defined in `def feature_engineering_v2`.
        - Second method is using new features created from `FENet` (Please notice that a small number like `1e-6` is added to the denominator to avoid dividing for zero). More details to explain about `FENet` can be found in the comments of the file. New features are created and defined in `def feature_engineering`. 
    - Time series data: 
        - First method is simply using `class AutoEncoder` as many other team do in this competition (we therefore do not explain more about this improvement). The function to perform autoencoder is: `def perform_autoencoder`. 
        - Second method is to create new features about sleep detection - inspired an old competition also held by Child Mind Institute. Justification for this idea can be found in the script (speaker note) of our slide (slide 27 - 28): [Google slides](https://docs.google.com/presentation/d/1TpkzeuvpTdi4k63G_HzPMaGe0ttvqEgNOmZzIlRWby4/edit?usp=sharing). We take the solution of top 4 of the competition [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
        - Both methods are implemented in the file `pipeline_FE_timeseries`. The comments to explain in code are provided 

### Deep learning approach.
