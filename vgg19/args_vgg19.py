class Train_Args():
    is_training                 =       True
    model_save_path             =       "./data/0.05/model/model-4/"
    model_filename              =       "vgg19.py"
    args_filename               =       "args_vgg19.py"
    data_name                   =       "0.5/y_train_2.npy"

    regularization              =       True
    reg_rate                    =       1.0
    dropout                     =       True
    dropout_rate                =       0.5
    n_classes                   =       6

    learning_rate               =       0.00005

    batch_size                  =       100

    num_epochs                  =       200

    decay_rate                  =       0.96
    decay_step                  =       100

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