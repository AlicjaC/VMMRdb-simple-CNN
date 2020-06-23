# VMMRdb-simple-CNN
Simple CNN model for cars' make, model and year classification on VMMRdb

Requires TF 2.0 or higher and Optuna.

For full project description please visit https://deepdrive.pl/?p=1170

prepare_csv
 - downloads data
 - creates csv file containing labels and paths to photos
 
train_v1
 - just any CNN model for tests
 - single output
 - data as np.array

train_v2
 - same CNN model as in train_v1
 - multiple output
 
optuna_env.yml and requirements.txt - my environment files in case they're necessary
