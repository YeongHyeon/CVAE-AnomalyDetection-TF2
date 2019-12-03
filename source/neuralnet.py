import tensorflow as tf

class CVAE(object):

    def __init__(self, height, width, channel, ksize, z_dim, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.ksize, self.z_dim = ksize, z_dim
        self.leaning_rate = leaning_rate
        self.initializer = tf.initializers.glorot_normal()
        self.name_bank, self.var_bank, self.params_trainable = [], [], []

        self.verbose = True
        self.encoder(tf.zeros([1, self.height, self.width, self.channel]))
        self.decoder(tf.zeros([1, self.z_dim]))
        self.verbose = False

        self.optimizer = tf.optimizers.Adam(self.leaning_rate)

    def loss(self, x, x_hat, z_mu, z_sigma):

        restore_error = -tf.reduce_sum(x * tf.math.log(x_hat + 1e-12) + (1 - x) * tf.math.log(1 - x_hat + 1e-12), axis=(1, 2, 3))
        kl_divergence = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.math.log(tf.square(z_sigma) + 1e-12) - 1, axis=(1))

        mean_restore = tf.reduce_mean(restore_error)
        mean_kld = tf.reduce_mean(kl_divergence)
        ELBO = tf.reduce_mean(restore_error + kl_divergence) # Evidence LowerBOund
        loss = ELBO

        return loss, mean_restore, mean_kld

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if(data_dim == 4):
            return tf.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.reduce_sum(tf.square(x1 - x2))

    def encoder(self, x):

        x = tf.cast(x, dtype=tf.float32)
        conv1_1 = self.conv2d(x, self.get_weight(vshape=[self.ksize, self.ksize, self.channel, 16], name="enc_conv1_1"), \
            stride_size=1, padding='SAME', activation='elu')
        conv1_2 = self.conv2d(conv1_1, self.get_weight(vshape=[self.ksize, self.ksize, 16, 16], name="enc_conv1_2"), \
            stride_size=1, padding='SAME', activation='elu')
        maxp1 = self.maxpool(conv1_2, pool_size=2, stride_size=2)

        conv2_1 = self.conv2d(maxp1, self.get_weight(vshape=[self.ksize, self.ksize, 16, 32], name="enc_conv2_1"), \
            stride_size=1, padding='SAME', activation='elu')
        conv2_2 = self.conv2d(conv2_1, self.get_weight(vshape=[self.ksize, self.ksize, 32, 32], name="enc_conv2_2"), \
            stride_size=1, padding='SAME', activation='elu')
        maxp2 = self.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.conv2d(maxp2, self.get_weight(vshape=[self.ksize, self.ksize, 32, 64], name="enc_conv3_1"), \
            stride_size=1, padding='SAME', activation='elu')
        conv3_2 = self.conv2d(conv3_1, self.get_weight(vshape=[self.ksize, self.ksize, 64, 64], name="enc_conv3_2"), \
            stride_size=1, padding='SAME', activation='elu')

        in_dense = (self.height//(2**2))*(self.width//(2**2))*64
        flatten = tf.reshape(conv3_2, shape=(-1, in_dense))
        fullcon1 = self.dense(flatten, self.get_weight(vshape=[in_dense, 512], name="enc_fullcon1"), \
            activation='elu')
        fullcon2 = self.dense(fullcon1, self.get_weight(vshape=[512, self.z_dim*2], name="enc_fullcon2"), \
            activation='none')

        z_mu, z_sigma = self.split_z(z=fullcon2)
        z = self.sample_z(mu=z_mu, sigma=z_sigma)

        return z, z_mu, z_sigma

    def decoder(self, z):

        z = tf.cast(z, dtype=tf.float32)
        out_dense = (self.height//(2**2))*(self.width//(2**2))*64
        fullcon1 = self.dense(z, self.get_weight(vshape=[self.z_dim, 512], name="dec_fullcon1"), \
            activation='elu')
        fullcon2 = self.dense(fullcon1, self.get_weight(vshape=[512, out_dense], name="dec_fullcon2"), \
            activation='elu')
        spatial = tf.reshape(fullcon2, shape=(-1, self.height//(2**2), self.width//(2**2), 64))

        conv1_1 = self.conv2d(spatial, self.get_weight(vshape=[self.ksize, self.ksize, 64, 64], name="dec_conv1_1"), \
            stride_size=1, padding='SAME', activation='elu')
        conv1_2 = self.conv2d(conv1_1, self.get_weight(vshape=[self.ksize, self.ksize, 64, 32], name="dec_conv1_2"), \
            stride_size=1, padding='SAME', activation='elu')

        conv2_1 = self.conv2d_tr(conv1_2, self.get_weight(vshape=[self.ksize, self.ksize, 32, 32], name="dec_conv2_1"), output_shape=[tf.shape(z)[0], self.height//(2**1), self.width//(2**1), 32], \
            stride_size=2, padding='SAME', activation='elu')
        conv2_2 = self.conv2d(conv2_1, self.get_weight(vshape=[self.ksize, self.ksize, 32, 16], name="dec_conv2_2"), \
            stride_size=1, padding='SAME', activation='elu')

        conv3_1 = self.conv2d_tr(conv2_2, self.get_weight(vshape=[self.ksize, self.ksize, 16, 16], name="dec_conv3_1"), output_shape=[tf.shape(z)[0], self.height//(2**0), self.width//(2**0), 16], \
            stride_size=2, padding='SAME', activation='elu')
        conv3_2 = self.conv2d(conv3_1, self.get_weight(vshape=[self.ksize, self.ksize, 16, self.channel], name="dec_conv3_2"), \
            stride_size=1, padding='SAME', activation='sigmoid')

        return conv3_2

    def split_z(self, z):

        z_mu = z[:, :self.z_dim]
        # z_mu = tf.clip_by_value(z_mu, -3+(1e-12), 3-(1e-12))
        z_sigma = z[:, self.z_dim:]
        # z_sigma = tf.clip_by_value(z_sigma, 1e-12, 1-(1e-12))

        return z_mu, z_sigma

    def sample_z(self, mu, sigma):

        # default of tf.random.normal: mean=0.0, stddev=1.0
        epsilon = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        sample = mu + (sigma * epsilon)

        return sample

    def get_weight(self, vshape, name):

        try:
            idx_w = self.name_bank.index("%s_w" %(name))
            idx_b = self.name_bank.index("%s_b" %(name))
        except:
            w = tf.Variable(self.initializer(vshape), \
                name=name, trainable=True, dtype=tf.float32)
            b = tf.Variable(self.initializer([vshape[-1]]), \
                name=name, trainable=True, dtype=tf.float32)

            self.name_bank.append("%s_w" %(name))
            self.params_trainable.append(w)
            self.name_bank.append("%s_b" %(name))
            self.params_trainable.append(b)
        else:
            w = self.params_trainable[idx_w]
            b = self.params_trainable[idx_b]

        if(self.verbose): print(name, w.shape)
        return w, b

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def conv2d(self, inputs, variables, stride_size, padding, activation):

        [weights, biasis] = variables
        out = tf.nn.conv2d(inputs, weights, \
            strides=[1, stride_size, stride_size, 1], padding=padding) + biasis
        return self.activation_fn(out, activation=activation)

    def conv2d_tr(self, inputs, variables, output_shape, stride_size, padding, activation):

        [weights, biasis] = variables
        out = tf.nn.conv2d_transpose(inputs, weights, output_shape, \
            strides=[1, stride_size, stride_size, 1], padding=padding) + biasis
        return self.activation_fn(out, activation=activation)

    def maxpool(self, inputs, pool_size, stride_size):

        if(self.verbose): print("max pool")
        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], \
            padding='VALID', strides=[1, stride_size, stride_size, 1])

    def dense(self, inputs, variables, activation):

        [weights, biasis] = variables
        out = tf.matmul(inputs, weights) + biasis
        return self.activation_fn(out, activation=activation)
