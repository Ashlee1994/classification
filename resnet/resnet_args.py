class Train_Args():
    is_training                 =       True
    model_save_path             =       "./output/0.5/model/model-6_class-dec_2/"
    checkpoint_dir              =       "./output/0.5/model/model-6_class-dec_2/checkpoint_dir"
    model_filename              =       "resnet_model.py"
    args_filename               =       "resnet_args.py"
    data_name                   =       "0.5/y_train_2.npy"

    regularization              =       True
    reg_rate                    =       0.00001
    dropout                     =       True
    dropout_rate                =       0.5
    num_classes                 =       6
    use_bottleneck              =       False

    lrn_rate                    =       0.1
    min_lrn_rate                =       0.0001
    num_residual_units          =       5
    optimizer                   =       "mom"    # "mom"  or "sgd"

    batch_size                  =       50

    num_epochs                  =       100

    weight_decay_rate           =       0.0002
    relu_leakiness              =       0.1


    decay_step                  =       50
    decay_rate                  =       0.5
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

