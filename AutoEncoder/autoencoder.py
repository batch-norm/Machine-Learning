import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale


def scale(df):
    return pd.DataFrame(minmax_scale(df))

def get_batch(df):
    df = shuffle(df)
    return df[:batch_size].values,df[:batch_size].values


def init_Variable():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 11])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 11])
    W0 = tf.Variable(tf.random_uniform(shape=(input_length, hidden_layer)))
    b0 = tf.Variable(tf.zeros(hidden_layer))
    h1 = tf.add(tf.matmul(X, W0),b0)
    h1_output = tf.nn.sigmoid(h1)
    W1 = tf.Variable(tf.random_uniform(shape=(hidden_layer, input_length)))
    b1 = tf.Variable(tf.zeros(input_length))
    output = tf.add(tf.matmul(h1_output, W1) , b1)
    loss = tf.reduce_sum(tf.square(output-y))
    tf.summary.scalar('l2 loss', loss)
    merged = tf.summary.merge_all()
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_writer1 = tf.summary.FileWriter('./plot1', sess.graph)
        train_writer2 = tf.summary.FileWriter('./plot2', sess.graph)
        for i in range(epochs):
            for j in range(batch_size):
                batch_x,batch_y = get_batch(data)
                valid_batch_x,valid_batch_y = get_batch(valid_data)
                _loss,_a,_b = sess.run([loss,optimizer,merged],feed_dict={X:batch_x,y:batch_y})
                __loss, __a, __b = sess.run([loss, optimizer, merged], feed_dict={X: valid_batch_x, y: valid_batch_y})
                train_writer1.add_summary(_b,batch_size*i+j)
                train_writer1.flush()
                train_writer2.add_summary(__b, batch_size * i + j)
                train_writer1.flush()
                print('l2 loss:',_loss,'valid_l2:',__loss)



if __name__ == '__main__':
    data = pd.read_csv('../A_dataset1.csv', index_col=0)
    data = data.drop(['user_id', 'label'], axis=1)
    valid_data = pd.read_csv('../A_dataset2.csv', index_col=0)
    valid_data = valid_data.drop(['user_id', 'label'], axis=1)
    input_length = 11
    hidden_layer = 6
    batch_size = 512
    epochs = 100
    data = scale(data)
    valid_data = scale(valid_data)
    init_Variable()
