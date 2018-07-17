class Train_Args():
    is_training                 =       True
    model_save_path             =       "./data/0.5/model/model-6_class-dec_1/"
    model_filename              =       "dec14.py"
    args_filename               =       "args_dec14.py"
    data_name                   =       "0.5/y_train_2.npy"

    regularization              =       True
    reg_rate                    =       0.00001
    dropout                     =       True
    dropout_rate                =       0.5
    n_classes                   =       6

    learning_rate               =       0.00001

    batch_size                  =       100

    num_epochs                  =       1000

    decay_rate                  =       0.5
    decay_step                  =       10
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0

class Predict_Args():
    is_training                 =       False
    data_path                   =       "data/KLHdata/mic/"
    result_path                 =       "./data/KLHdata/result/"
    model_save_path             =       "./data/KLHdata/model/model-9-272/"
    boxsize                     =       272
    resize                      =       224
    start_mic_num               =       75
    end_mic_num                 =       81
    dim_x                       =       2048
    dim_y                       =       2048
    scan_step                   =       20
    accuracy                    =       0.96
    threhold                    =       0.65
    name_length                 =       2
    name_prefix                 =       ""

    batch_size                  =       50