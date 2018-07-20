import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys,shutil,time
import resnet_model
from utils import *
from resnet_args import Train_Args


def train():
    args = Train_Args()
    train_start = time.time()
    time_start = time.time()

    checkpoint_dir = args.model_save_path
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    srcfile = args.model_filename
    dstfile = args.model_save_path + srcfile
    shutil.copyfile(srcfile,dstfile)
    shutil.copyfile(args.args_filename,args.model_save_path + args.args_filename)

    train_x, train_y, test_x, test_y = load_train_05_6_shuffle()
    
    print("train size is %d " % len(train_x),flush=True)
    num_train_batch = len(train_x) // args.batch_size
    num_test_batch = len(test_x) // args.batch_size
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # the second GPU
    best_train_accuracy = 0
    best_test_accuracy = 0

    output_name = args.model_save_path + "accuracy.txt"
    output = open(output_name, 'w')

    with tf.Session() as sess:
        # deepem = dec14(args)
        model = resnet_model.ResNet(args)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
        sess.run(tf.global_variables_initializer())
        for e in range(args.num_epochs):
            print('\n=============== Epoch %d/%d ==============='% (e + 1,args.num_epochs),flush=True)
            print("num_train_batch is %d" % num_train_batch,flush=True)
            acc_train = []
            for i in range(num_train_batch):
                batch_x = train_x[args.batch_size*i:args.batch_size*(i+1)]
                batch_y = train_y[args.batch_size*i:args.batch_size*(i+1)]       
                #loss,accuracy,lr,_= sess.run([deepem.loss, deepem.accuracy,deepem.lr,deepem.optimizer], {deepem.X:batch_x, deepem.Y: batch_y})
                loss,accuracy,lr,_= sess.run([model.cost, model.accuracy,model.lrn_rate,model.train_op], {model._images:batch_x, model.labels: batch_y})
                #model.build_graph()

                acc_train.append(accuracy)
                if i % 10 == 0:
                    print('lr: %.8f loss: %.6f  acc: %.6f'% (lr,loss, accuracy),flush=True)
            
            train_acc = np.mean(acc_train)
            if train_acc > best_train_accuracy:
                best_train_accuracy = train_acc
                output.write(" Epoch " + str(e + 1) + " train accuracy: " + str(best_train_accuracy) + '\n')
                output.flush()

            print("avg acc: %.6f" % train_acc )
            print("best_train_accuracy: %.6f" % best_train_accuracy,flush=True)

            if e % 10 == 0 or e == args.num_epochs -1:
                print("\ntesting start.",flush=True)
                print("num_test_batch is %d" % num_test_batch,flush=True)
                acc_test = []
                for i in range(num_test_batch):
                    batch_x = test_x[args.batch_size*i:args.batch_size*(i+1)]
                    batch_y = test_y[args.batch_size*i:args.batch_size*(i+1)]
                    accuracy = sess.run(model.accuracy,feed_dict={model._images:batch_x, model.labels: batch_y})
                    print('acc: %.6f'% (accuracy),flush=True)
                    acc_test.append(accuracy)

                acc = np.mean(acc_test)
                if acc > best_test_accuracy:
                    best_test_accuracy = acc
                    ckpt_path = os.path.join(checkpoint_dir, 'model.ckpt')
                    saver.save(sess, ckpt_path, global_step = e)
                    print("model saved!")
                    output.write(" Epoch " + str(e + 1) + " best test accuracy: " + str(best_test_accuracy) + '\n')
                print("avg acc: %.6f" % np.mean(acc) )
                print("best_test_accuracy: %.6f" % best_test_accuracy,flush=True)

    output.write("train accuracy: " + str(best_train_accuracy) + '\n')
    output.write("test accuracy: " + str(best_test_accuracy) + '\n')
    output.close
    train_end = time.time()
    print("\ntrain done! totally cost: %.5f \n" %(train_end - train_start),flush=True)
    print("best_test_accuracy: %.6f" % best_test_accuracy,flush=True)

if __name__ == '__main__':
    train()

