class Train_Args():
    is_training                 =       True
    model_save_path             =       "output/0.5/model/resnet-6_class-2/"
    model_filename              =       "densenet_model.py"
    args_filename               =       "densenet_args.py"
    data_name                   =       "0.5/y_train_2.npy"
    dataset_name                =       "005-6"

    regularization              =       True
    reg_rate                    =       0.00001
    dropout                     =       True
    dropout_rate                =       0.5
    num_classes                 =       6
    use_bottleneck              =       False

    learning_rate               =       0.01
    min_lrn_rate                =       0.0001
    num_residual_units          =       5

    batch_size                  =       5

    num_epochs                  =       100

    weight_decay_rate           =       0.0002
    relu_leakiness              =       0.1

    # depth：Depth of whole network, restricted to paper choices. optional: 40 , 100, 190, 250 
    depth                       =       100 

    # growth_rate Grows rate for every layer. optional: 12, 24, 40  
    growth_rate                 =       12  

    # total_blocks: Total blocks of layers stack
    total_blocks                =       3 

    # use_bottleneck: should we use bottleneck layers and features reduction or not
    use_bottleneck              =       True  

    # reduction: `float`, reduction Theta at transition layer for DenseNets with bottleneck layers. See paragraph 'Compression'
    reduction                   =       1.0  

    # keep_prob: `float`, keep probability for dropout. If keep_prob = 1 dropout will be disables
    keep_prob                   =       1 

    # weight_decay: Weight decay for optimizer
    weight_decay                =       1e-4

    # decay_rate:  decay rate for learning rate
    decay_rate                  =       0.9
    decay_step                  =       300

    # Nesterov momentum
    nesterov_momentum           =       0.9
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

