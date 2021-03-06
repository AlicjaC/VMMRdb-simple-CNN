# VMMRdb-simple-CNN
Simple CNN model for cars' make, model and year classification on VMMRdb as in:
A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition F. Tafazzoli, K. Nishiyama and H. Frigui In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2017.

Requires TF 2.0 or higher and Optuna.

For full project description please visit https://deepdrive.pl/?p=1170

## Basic files
prepare_csv
 - downloads data
 - creates csv file containing labels and paths to images
 
cars_final
 - script to train model with augmentation
 
## Hyperparameter optimization files
4 phrases of hyperparameter optimization:
 - optuna_conv
 - optuna_dense_make (must be adapted for model and year layers)
 - optuna_reg_make (must be adapted for model and year layers)
 - lr_scheduler
 
## Additional files
train_v1
 - just any CNN model for tests
 - single output
 - data as np.array

train_v2
 - multiple output
 - data as np.array
 
train_v3
 - single output
 - data as tf.data
 
train_v4
 - multiple output
 - data as tf.data
 
optuna_env.yml and requirements.txt - my environment's files, in case they're necessary
