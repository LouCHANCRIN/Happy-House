import tensorflow as tf

def conv_2d(X, W):
    return (tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME'))

def max_pool(data, k=2):
    return (tf.nn.max_pool(data, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME'))

def relu(data, bias):
    return (tf.nn.relu(tf.nn.bias_add(data, bias)))

def tanh(data, bias):
    return (tf.nn.tanh(tf.nn.bias_add(data, bias)))

def MyModel(data, weight, bias):
    shape = data.get_shape().as_list()
    data = tf.reshape(data, [-1, shape[1], shape[2], shape[3]])

    conv1 = conv_2d(data, weight['conv1'])
    conv1 = relu(conv1, bias['conv1'])
    conv1 = max_pool(conv1)

    conv2 = conv_2d(conv1, weight['conv2'])
    conv2 = relu(conv2, bias['conv2'])
    conv2 = max_pool(conv2)

    conv3 = conv_2d(conv2, weight['conv3'])
    conv3 = relu(conv3, bias['conv3'])
    conv3 = max_pool(conv3)

    shape = conv3.get_shape().as_list()
    fc = tf.reshape(conv3, [-1,shape[1]*shape[2]*shape[3]])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weight['fc1']), bias['fc1']))
    output = tf.add(tf.matmul(fc, weight['output']), bias['output'])

    return (output)
