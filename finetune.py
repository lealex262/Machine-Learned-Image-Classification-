import tensorflow as tf
import numpy as np

from model import Model
from network import *

def main():


    # Learning params
    learning_rate = 0.0005
    training_iters = 6000 # 10 epochs
    batch_size = 10
    display_step = 20
    test_step = 100 # 0.5 epoch
    save_step = 1000

    #Graph input
    

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Network params
    n_classes = 5

    # Init
    init = tf.initialize_all_variables()

    #Run Session
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        print('Start training')
        step = 1


        print("Finish!")


if __name__ == '__main__':
    main()