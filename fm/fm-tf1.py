import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
raw_x_train, raw_x_val, y_train, y_val = train_test_split(boston['data'], boston['target'], test_size=0.1, random_state=1)

dense_index = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sparse_index = [3, ]

def get_decoder(x, col):
    categories = set(x[:, col])
    idx2cat = dict(enumerate(categories, start=1))
    idx2cat[0] = "UNK"
    return idx2cat

def get_encoder(x, col):
    idx2cat = get_decoder(x, col)
    return {v:k for k, v in idx2cat.items()}

scaler = StandardScaler()
label_encoders = {idx: get_encoder(raw_x_train, idx) for idx in sparse_index}
label_decoders = {idx: get_decoder(raw_x_val, idx) for idx in sparse_index}

def preproess(x, is_train=False):
    if is_train:
        x_dense = scaler.fit_transform(x[:, dense_index])
    else:
        x_dense = scaler.transform(x[:, dense_index])
    x_sparse = np.zeros((x.shape[0], len(sparse_index)))
    for i, idx in enumerate(sparse_index):
        x_sparse[:, i] = [label_encoders[idx][val] for val in x[:, idx]]
    return x_dense, x_sparse

x_dense_train, x_sparse_train = preproess(raw_x_train, is_train=True)
x_dense_val, x_sparse_val = preproess(raw_x_val, is_train=False)

n_dense = len(dense_index)
n_sparse = len(sparse_index)
n_total_sparse_categories = sum(len(encoder) for encoder in label_encoders.values())
embedding_size = 3  # embedding size

np.random.seed(123)

# init placeholder
x_dense = tf.placeholder(tf.float32, shape=(None, n_dense))
x_sparse = tf.placeholder(tf.int32, shape=(None, n_sparse))
y = tf.placeholder(tf.float32, shape=(None, ))

# init variables
w0 =  tf.Variable(0.0)
w1_sparse = tf.Variable(np.random.normal(size=(n_total_sparse_categories, 1)), dtype=tf.float32)
w1_dense = tf.Variable(np.random.normal(size=(n_dense, 1)), dtype=tf.float32)

v_dense = tf.Variable(np.random.normal(size=(n_dense, embedding_size)), dtype=tf.float32)
v_sparse = tf.Variable(np.random.normal(size=(n_total_sparse_categories, embedding_size)), dtype=tf.float32)

# model
first_order_sparse_part = tf.squeeze(tf.nn.embedding_lookup(w1_sparse, x_sparse))
first_order_dense_part  = tf.squeeze(tf.matmul(x_dense, w1_dense))
first_order = first_order_sparse_part + first_order_dense_part

square_sum_sparse = tf.reshape(tf.nn.embedding_lookup(v_sparse, x_sparse), (-1, embedding_size))
square_sum_dense = tf.matmul(x_dense, v_dense)
square_sum = tf.square(square_sum_dense + square_sum_sparse)

sum_square_sparse= tf.reshape(tf.nn.embedding_lookup(tf.square(v_sparse), x_sparse), (-1, embedding_size))
sum_square_dense = tf.matmul(tf.square(x_dense),tf.square(v_dense))
sum_square = sum_square_dense + sum_square_sparse
second_order_square = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1)

logits = w0 + first_order + second_order_square

learning_rate = 0.1
n_epochs = 100
batch_size = 32
loss = tf.reduce_mean(tf.square(logits - y))
optimzier = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimzier.minimize(loss)

history = {'train_loss': []}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batch = x_dense_train.shape[0] // batch_size
    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(x_dense_train.shape[0])
        x_dense_train = x_dense_train[shuffled_idx]
        x_sparse_train = x_sparse_train[shuffled_idx]
        y_train = y_train[shuffled_idx]
        for i in range(n_batch):
            loss_val, _ = sess.run([loss, train_step], feed_dict={x_dense: x_dense_train[i*batch_size: (i+1)*batch_size] ,x_sparse: x_sparse_train[i*batch_size: (i+1)*batch_size], y: y_train[i*batch_size: (i+1)*batch_size]})

            history['train_loss'].append(loss_val)
            if i % 50 == 0:
                print(f"Epoch: {epoch}/{n_epochs} loss_val: {loss_val:>6.3} val loss: {sess.run([loss], feed_dict={x_dense: x_dense_val ,x_sparse: x_sparse_val, y: y_val})[0]:>6.3}")
