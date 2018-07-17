# encoding: utf-8

import os,random
import pandas as pd
import numpy as np
import mrcfile
import cv2 as cv
from PIL import Image

def load_train_2():
    '''
    This part include load training parameters and training data
    '''

    train_y = np.load('0.5/y_train_2.npy')
    test_y = np.load('0.5/y_test_2.npy')

    train_x_tmp1 = np.load('0.5/x_train_1000_a_new.npy')
    train_x_tmp2 = np.load('0.5/x_train_1000_b_new.npy')

    train_x = np.vstack([train_x_tmp1,train_x_tmp2])

   #  train_x = (train_x - train_x.min())/(train_x.max() - train_x.min())
    # print("shape of train_x[0] first: ", train_x[0].shape)

    # print("train_x[0]:" , train_x[100])
    # image = (train_x[0]).astype(np.uint8)
    # cv.imshow("image: ",image)
    # cv.waitKey(0) 




    train_x = train_x.reshape([1600,180,180,1])

    test_x_tmp1 = np.load('0.5/x_test_1000_a_new.npy')
    test_x_tmp2 = np.load('0.5/x_test_1000_b_new.npy')
    test_x = np.vstack([test_x_tmp1,test_x_tmp2])

    # test_x = (test_x - test_x.min())/(test_x.max() - test_x.min())
    test_x = test_x.reshape([400,180,180,1])

    
    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    

    print("shape of train_x: ", train_x.shape)
    print("shape of train_y: ", train_y.shape)
    print("shape of test_x: ", test_x.shape)
    print("shape of test_y: ", test_y.shape)

    # print("train_x:" , train_x)
    # print("train_y: ", train_y)
    print("test_x:" , test_x)
    # print("test_y: ", test_y)
    return train_x, train_y, test_x, test_y


def load_train_6():
    '''
    This part include load training parameters and training data
    '''

    train_y = np.load('0.5/y_train_6.npy')
    test_y = np.load('0.5/y_test_6.npy')

    train_x_tmp1 = np.load('0.5/x_train_1000_a_new.npy')
    train_x_tmp2 = np.load('0.5/x_train_1000_b_new.npy')
    train_x_tmp3 = np.load('0.5/x_train_1000_c_new.npy')
    train_x_tmp4 = np.load('0.5/x_train_1000_d_new.npy')
    train_x_tmp5 = np.load('0.5/x_train_1000_d2_new.npy')
    train_x_tmp6 = np.load('0.5/x_train_1000_d3_new.npy')

    train_x = np.vstack([train_x_tmp1,train_x_tmp2])
    train_x = np.vstack([train_x,train_x_tmp3])
    train_x = np.vstack([train_x,train_x_tmp4])
    train_x = np.vstack([train_x,train_x_tmp5])
    train_x = np.vstack([train_x,train_x_tmp6])

   #  train_x = (train_x - train_x.min())/(train_x.max() - train_x.min())
    # print("shape of train_x[0] first: ", train_x[0].shape)

    # print("train_x[0]:" , train_x[100])
    # image = (train_x[0]).astype(np.uint8)
    # cv.imshow("image: ",image)
    # cv.waitKey(0) 

    train_x = train_x.reshape([4800,180,180,1])

    test_x_tmp1 = np.load('0.5/x_test_1000_a_new.npy')
    test_x_tmp2 = np.load('0.5/x_test_1000_b_new.npy')
    test_x_tmp3 = np.load('0.5/x_test_1000_c_new.npy')
    test_x_tmp4 = np.load('0.5/x_test_1000_d_new.npy')
    test_x_tmp5 = np.load('0.5/x_test_1000_d2_new.npy')
    test_x_tmp6 = np.load('0.5/x_test_1000_d3_new.npy')

    test_x = np.vstack([test_x_tmp1,test_x_tmp2])
    test_x = np.vstack([test_x,test_x_tmp3])
    test_x = np.vstack([test_x,test_x_tmp4])
    test_x = np.vstack([test_x,test_x_tmp5])
    test_x = np.vstack([test_x,test_x_tmp6])

    # test_x = (test_x - test_x.min())/(test_x.max() - test_x.min())


    # total_x = np.vstack([train_x, test_x])
    # total_y = np.vstack([train_y, test_y])


    test_x = test_x.reshape([1200,180,180,1])

    print("shape of train_x: ", train_x.shape)
    print("length of train_x: ", len(train_x))
    print("shape of train_y: ", train_y.shape)

    
    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    

    print("shape of train_x: ", train_x.shape)
    print("shape of train_y: ", train_y.shape)
    print("shape of test_x: ", test_x.shape)
    print("shape of test_y: ", test_y.shape)

    # print("train_x:" , train_x)
    # print("train_y: ", train_y)
    # print("test_x:" , test_x)
    # print("test_y: ", test_y)
    return train_x, train_y, test_x, test_y

def load_train_6_shuffle():
    '''
    This part include load training parameters and training data
    '''

    train_y = np.load('0.5/y_train_6.npy')
    test_y = np.load('0.5/y_test_6.npy')

    train_x_tmp1 = np.load('0.5/x_train_1000_a_new.npy')
    train_x_tmp2 = np.load('0.5/x_train_1000_b_new.npy')
    train_x_tmp3 = np.load('0.5/x_train_1000_c_new.npy')
    train_x_tmp4 = np.load('0.5/x_train_1000_d_new.npy')
    train_x_tmp5 = np.load('0.5/x_train_1000_d2_new.npy')
    train_x_tmp6 = np.load('0.5/x_train_1000_d3_new.npy')

    train_x = np.vstack([train_x_tmp1,train_x_tmp2])
    train_x = np.vstack([train_x,train_x_tmp3])
    train_x = np.vstack([train_x,train_x_tmp4])
    train_x = np.vstack([train_x,train_x_tmp5])
    train_x = np.vstack([train_x,train_x_tmp6])

   #  train_x = (train_x - train_x.min())/(train_x.max() - train_x.min())
    # print("shape of train_x[0] first: ", train_x[0].shape)

    # print("train_x[0]:" , train_x[100])
    # image = (train_x[0]).astype(np.uint8)
    # cv.imshow("image: ",image)
    # cv.waitKey(0) 


    test_x_tmp1 = np.load('0.5/x_test_1000_a_new.npy')
    test_x_tmp2 = np.load('0.5/x_test_1000_b_new.npy')
    test_x_tmp3 = np.load('0.5/x_test_1000_c_new.npy')
    test_x_tmp4 = np.load('0.5/x_test_1000_d_new.npy')
    test_x_tmp5 = np.load('0.5/x_test_1000_d2_new.npy')
    test_x_tmp6 = np.load('0.5/x_test_1000_d3_new.npy')

    test_x = np.vstack([test_x_tmp1,test_x_tmp2])
    test_x = np.vstack([test_x,test_x_tmp3])
    test_x = np.vstack([test_x,test_x_tmp4])
    test_x = np.vstack([test_x,test_x_tmp5])
    test_x = np.vstack([test_x,test_x_tmp6])

    # test_x = (test_x - test_x.min())/(test_x.max() - test_x.min())

    total_x = np.vstack([train_x, test_x])
    total_y = np.vstack([train_y, test_y])

    index = [i for i in range(len(total_x))]
    np.random.shuffle(index)
    total_x = total_x[index]
    total_y = total_y[index]

    train_x = total_x[0:4800]
    train_y = total_y[0:4800]
    test_x = total_x[4800:]
    test_y = total_y[4800:]

    train_x = train_x.reshape([4800,180,180,1])
    test_x = test_x.reshape([1200,180,180,1])

    print("shape of train_x: ", train_x.shape)
    print("length of train_x: ", len(train_x))
    print("shape of train_y: ", train_y.shape)

    
    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    

    print("shape of train_x: ", train_x.shape)
    print("shape of train_y: ", train_y.shape)
    print("shape of test_x: ", test_x.shape)
    print("shape of test_y: ", test_y.shape)

    # print("train_x:" , train_x)
    # print("train_y: ", train_y)
    # print("test_x:" , test_x)
    # print("test_y: ", test_y)
    return train_x, train_y, test_x, test_y

def load_train_005_6_shuffle():
    '''
    This part include load training parameters and training data
    '''

    train_y = np.load('0.05/y_train_6.npy')
    test_y = np.load('0.05/y_test_6.npy')

    train_x_tmp1 = np.load('0.05/x_train_1000_a_new.npy')
    train_x_tmp2 = np.load('0.05/x_train_1000_b_new.npy')
    train_x_tmp3 = np.load('0.05/x_train_1000_c_new.npy')
    train_x_tmp4 = np.load('0.05/x_train_1000_d_new.npy')
    train_x_tmp5 = np.load('0.05/x_train_1000_d2_new.npy')
    train_x_tmp6 = np.load('0.05/x_train_1000_d3_new.npy')

    train_x = np.vstack([train_x_tmp1,train_x_tmp2])
    train_x = np.vstack([train_x,train_x_tmp3])
    train_x = np.vstack([train_x,train_x_tmp4])
    train_x = np.vstack([train_x,train_x_tmp5])
    train_x = np.vstack([train_x,train_x_tmp6])

   #  train_x = (train_x - train_x.min())/(train_x.max() - train_x.min())
    # print("shape of train_x[0] first: ", train_x[0].shape)

    # print("train_x[0]:" , train_x[100])
    # image = (train_x[0]).astype(np.uint8)
    # cv.imshow("image: ",image)
    # cv.waitKey(0) 


    test_x_tmp1 = np.load('0.05/x_test_1000_a_new.npy')
    test_x_tmp2 = np.load('0.05/x_test_1000_b_new.npy')
    test_x_tmp3 = np.load('0.05/x_test_1000_c_new.npy')
    test_x_tmp4 = np.load('0.05/x_test_1000_d_new.npy')
    test_x_tmp5 = np.load('0.05/x_test_1000_d2_new.npy')
    test_x_tmp6 = np.load('0.05/x_test_1000_d3_new.npy')

    test_x = np.vstack([test_x_tmp1,test_x_tmp2])
    test_x = np.vstack([test_x,test_x_tmp3])
    test_x = np.vstack([test_x,test_x_tmp4])
    test_x = np.vstack([test_x,test_x_tmp5])
    test_x = np.vstack([test_x,test_x_tmp6])

    # test_x = (test_x - test_x.min())/(test_x.max() - test_x.min())

    total_x = np.vstack([train_x, test_x])
    total_y = np.vstack([train_y, test_y])

    index = [i for i in range(len(total_x))]
    np.random.shuffle(index)
    total_x = total_x[index]
    total_y = total_y[index]

    train_x = total_x[0:4800]
    train_y = total_y[0:4800]
    test_x = total_x[4800:]
    test_y = total_y[4800:]

    train_x = train_x.reshape([4800,180,180,1])
    test_x = test_x.reshape([1200,180,180,1])

    print("shape of train_x: ", train_x.shape)
    print("length of train_x: ", len(train_x))
    print("shape of train_y: ", train_y.shape)

    
    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    

    print("shape of train_x: ", train_x.shape)
    print("shape of train_y: ", train_y.shape)
    print("shape of test_x: ", test_x.shape)
    print("shape of test_y: ", test_y.shape)

    # print("train_x:" , train_x)
    # print("train_y: ", train_y)
    # print("test_x:" , test_x)
    # print("test_y: ", test_y)
    return train_x, train_y, test_x, test_y


def load_train_5_shuffle():
    '''
    This part include load training parameters and training data
    '''

    train_y = np.load('0.5/y_train_6.npy')
    test_y = np.load('0.5/y_test_6.npy')

    train_x_tmp1 = np.load('0.5/x_train_1000_a_new.npy')
    train_x_tmp2 = np.load('0.5/x_train_1000_b_new.npy')
    train_x_tmp3 = np.load('0.5/x_train_1000_c_new.npy')
    train_x_tmp4 = np.load('0.5/x_train_1000_d_new.npy')
    train_x_tmp5 = np.load('0.5/x_train_1000_d2_new.npy')
    train_x_tmp6 = np.load('0.5/x_train_1000_d3_new.npy')

    train_x = np.vstack([train_x_tmp1,train_x_tmp2])
    train_x = np.vstack([train_x,train_x_tmp3])
    train_x = np.vstack([train_x,train_x_tmp4])
    train_x = np.vstack([train_x,train_x_tmp5])
    train_x = np.vstack([train_x,train_x_tmp6])

   #  train_x = (train_x - train_x.min())/(train_x.max() - train_x.min())
    # print("shape of train_x[0] first: ", train_x[0].shape)

    # print("train_x[0]:" , train_x[100])
    # image = (train_x[0]).astype(np.uint8)
    # cv.imshow("image: ",image)
    # cv.waitKey(0) 


    test_x_tmp1 = np.load('0.5/x_test_1000_a_new.npy')
    test_x_tmp2 = np.load('0.5/x_test_1000_b_new.npy')
    test_x_tmp3 = np.load('0.5/x_test_1000_c_new.npy')
    test_x_tmp4 = np.load('0.5/x_test_1000_d_new.npy')
    test_x_tmp5 = np.load('0.5/x_test_1000_d2_new.npy')
    test_x_tmp6 = np.load('0.5/x_test_1000_d3_new.npy')

    test_x = np.vstack([test_x_tmp1,test_x_tmp2])
    test_x = np.vstack([test_x,test_x_tmp3])
    test_x = np.vstack([test_x,test_x_tmp4])
    test_x = np.vstack([test_x,test_x_tmp5])
    test_x = np.vstack([test_x,test_x_tmp6])

    # test_x = (test_x - test_x.min())/(test_x.max() - test_x.min())

    total_x = np.vstack([train_x, test_x])
    total_y = np.vstack([train_y, test_y])

    index = [i for i in range(len(total_x))]
    np.random.shuffle(index)
    total_x = total_x[index]
    total_y = total_y[index]

    train_x = total_x[0:4800]
    train_y = total_y[0:4800]
    test_x = total_x[4800:]
    test_y = total_y[4800:]

    train_x = train_x.reshape([4800,180,180,1])
    test_x = test_x.reshape([1200,180,180,1])

    print("shape of train_x: ", train_x.shape)
    print("length of train_x: ", len(train_x))
    print("shape of train_y: ", train_y.shape)

    
    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    

    print("shape of train_x: ", train_x.shape)
    print("shape of train_y: ", train_y.shape)
    print("shape of test_x: ", test_x.shape)
    print("shape of test_y: ", test_y.shape)

    # print("train_x:" , train_x)
    # print("train_y: ", train_y)
    # print("test_x:" , test_x)
    # print("test_y: ", test_y)
    return train_x, train_y, test_x, test_y

















