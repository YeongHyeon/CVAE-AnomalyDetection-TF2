import tensorflow as tf

class CVAE(object):

    def __init__(self, height, width, channel, ksize, z_dim, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim = ksize, z_dim
        self.leaning_rate = leaning_rate
        self.initializer = tf.initializers.glorot_normal()
        self.name_bank, self.var_bank = [], []

        self.encoder(tf.zeros([1, self.height, self.width, self.channel]))

    def encoder(self, x):

        x = tf.cast(x, dtype=tf.float32)
        conv1_1 = self.conv2d(x, self.get_weight(vshape=[self.ksize, self.ksize, 1, 16], name="conv1_1"), \
            stride_size=1, padding='SAME')
        conv1_2 = self.conv2d(conv1_1, self.get_weight(vshape=[self.ksize, self.ksize, 16, 16], name="conv1_2"), \
            stride_size=1, padding='SAME')
        maxp1 = self.maxpool(conv1_2, pool_size=2, stride_size=2)

        conv2_1 = self.conv2d(maxp1, self.get_weight(vshape=[self.ksize, self.ksize, 16, 32], name="conv2_1"), \
            stride_size=1, padding='SAME')
        conv2_2 = self.conv2d(conv2_1, self.get_weight(vshape=[self.ksize, self.ksize, 32, 32], name="conv2_2"), \
            stride_size=1, padding='SAME')
        maxp2 = self.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.conv2d(maxp2, self.get_weight(vshape=[self.ksize, self.ksize, 32, 64], name="conv3_1"), \
            stride_size=1, padding='SAME')
        conv3_2 = self.conv2d(conv3_1, self.get_weight(vshape=[self.ksize, self.ksize, 64, 64], name="conv3_2"), \
            stride_size=1, padding='SAME')
        maxp3 = self.maxpool(conv3_2, pool_size=2, stride_size=2)

        dense
        

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def get_weight(self, vshape, name):

        try:
            var_idx = self.name_bank.index(name)
        except:
            w = tf.Variable(self.initializer(vshape), \
                name=name, trainable=True, dtype=tf.float32)
            b = tf.Variable(self.initializer([vshape[-1]]), \
                name=name, trainable=True, dtype=tf.float32)
            self.name_bank.append(name)
            self.var_bank.append([w, b])
        else:
            [w, b] = self.var_bank[var_idx]

        print(name, w.shape)
        return w, b

    def conv2d(self, inputs, variables, stride_size, padding):

        [weights, biasis] = variables
        out = tf.nn.conv2d(inputs, weights, strides=[1, stride_size, stride_size, 1], \
            padding=padding) + biasis
        return tf.nn.elu(out)

    def maxpool(self, inputs, pool_size, stride_size):

        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], \
            padding='VALID', strides=[1, stride_size, stride_size, 1])

    def dense(self, inputs, variables):

        [weights, biasis] = variables
        out = tf.matmul(inputs, weights) + biasis
        return tf.nn.elu(out)
