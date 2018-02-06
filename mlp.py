import tensorflow as tf
from sklearn import cross_validation, preprocessing
import numpy as np

keep_prob = tf.placeholder(tf.float64)

def build_graph(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    batch_mean1, batch_var1 = tf.nn.moments(layer_1, [0])
    scale1 = tf.Variable(tf.ones([n_hidden_1], dtype=tf.float64))
    beta1 = tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float64))
    layer_1 = tf.nn.batch_normalization(layer_1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    batch_mean2, batch_var2 = tf.nn.moments(layer_2, [0])
    scale2 = tf.Variable(tf.ones([n_hidden_2], dtype=tf.float64))
    beta2 = tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float64))
    layer_2 = tf.nn.batch_normalization(layer_2,batch_mean2,batch_var2,beta2,scale2,epsilon)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    batch_mean3, batch_var3 = tf.nn.moments(layer_3, [0])
    scale3 = tf.Variable(tf.ones([n_hidden_3], dtype=tf.float64))
    beta3 = tf.Variable(tf.zeros([n_hidden_3], dtype=tf.float64))
    layer_3 = tf.nn.batch_normalization(layer_3,batch_mean3,batch_var3,beta3,scale3,epsilon)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
                             
    return output_layer

def parse_file(fileName):
    file = open(fileName, 'r')
    arr = file.readlines()
    X = []
    Y = []
    for i in range(len(arr)):
        temp = arr[i].replace("\n", "").split(" ")
        temp = list(map(lambda x: float(x), temp))
        X.append(temp[0:-1])
        Y.append(temp[-1])
    return X, Y

def normalize(X_train, X_test, Y_train, Y_test):
	scaler_X = preprocessing.StandardScaler().fit(X_train)
	scaler_Y = preprocessing.StandardScaler().fit(np.array(Y_train).reshape(-1, 1))
	X_train = scaler_X.transform(X_train)
	X_test = scaler_X.transform(X_test)
	Y_train = scaler_Y.transform(np.array(Y_train).reshape(-1, 1))
	Y_test = scaler_Y.transform(np.array(Y_test).reshape(-1, 1))
	return X_train, X_test, np.array(Y_train).flatten(), np.array(Y_test).flatten()

x, y = parse_file("02.output")

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)

print(np.array(Y_test).shape)

X_train, X_test, Y_train, Y_test = normalize(X_train, X_test, Y_train, Y_test);
print(np.array(Y_test).shape)

learning_rate = 0.1
training_epoches = 15
batch_size = 200
display_step = 1
dropout_rate = 0.5
epsilon=1e-3

total_len = np.array(X_train).shape[0]
n_input = np.array(X_train).shape[1]
n_hidden_1 = 64
n_hidden_2 = 64
n_hidden_3 = 64
n_classes = 1

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None])

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], dtype=tf.float64)),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=tf.float64)),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], dtype=tf.float64)),
	'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], dtype=tf.float64))
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64)),
	'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=tf.float64)),
	'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=tf.float64)),
	'out': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))
}

pred = build_graph(X_train)
loss_op = tf.reduce_mean(tf.square(tf.transpose(pred) - Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

def main(_):
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for epoch in range(training_epoches):
	        avg_cost = 0.
	        total_batch = int(total_len/batch_size)
	        # Loop over all batches
	        for i in range(total_batch-1):
	            batch_x = np.array(X_train[i*batch_size:(i+1)*batch_size])
	            batch_y = np.array(Y_train[i*batch_size:(i+1)*batch_size])
	            # Run optimization op (backprop) and cost op (to get loss value)
	            _, c, p = sess.run([optimizer, loss_op, pred], feed_dict={X: batch_x,
	                                                          Y: batch_y, keep_prob: 0.5})
	            # Compute average loss
	            avg_cost += c / total_batch

	        # sample prediction
	        label_value = batch_y
	        estimate = p
	        err = label_value-estimate
	        print ("num batch:", total_batch)

	        # Display logs per epoch step
	        if epoch % display_step == 0:
	            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
	                "{:.9f}".format(avg_cost))
	            print ("[*]----------------------------")
	            for i in range(3):
	                print ("label value:", label_value[i], \
	                    "estimated value:", estimate[i])
	            print ("[*]============================")
	    print ("Optimization Finished!")

	    mse_final = sess.run(loss_op, feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0})
	    print("TEST LOSS: ")
	    print(mse_final)

if __name__ == '__main__':
	tf.app.run()
