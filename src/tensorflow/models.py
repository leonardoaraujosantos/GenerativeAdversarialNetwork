import tensorflow as tf
import model_util as util


class VAE_CNN(object):
    def __init__(self, img_size=28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # Calculating the convolution output:
            # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
            # H_out = 1 + (H_in+(2*pad)-K)/S
            # W_out = 1 + (W_in+(2*pad)-K)/S
            # CONV1: Input 28x28x1 after CONV 5x5 P:2 S:2 H_out: 1 + (28+4-5)/2 = 14, W_out= 1 + (28+4-5)/2 = 14
            self.__conv1 = util.conv2d(self.__x_image, 5, 5, 1, 16, 2, "conv1", pad='SAME',
                                       viewWeights=True, do_summary=False)
            self.__conv1_act = util.relu(self.__conv1, do_summary=False)

            # CONV2: Input 14x14x16 after CONV 5x5 P:0 S:2 H_out: 1 + (14+4-5)/2 = 7, W_out= 1 + (14+4-5)/2 = 7
            self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 16, 32, 2, "conv2", do_summary=False, pad='SAME')
            self.__conv2_act = util.relu(self.__conv2, do_summary=False)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv2_act, [tf.shape(self.__x)[0], 7 * 7 * 32])

            # Add linear ops for mean and variance
            self.__w_mean = util.linear_layer(self.__enc_out, 7 * 7 * 32, latent_size, "w_mean")
            self.__w_stddev = util.linear_layer(self.__enc_out, 7 * 7 * 32, latent_size, "w_stddev")

            # Generate normal distribution with dimensions [B, latent_size]
            self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)

            self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)
            tf.summary.histogram("latent_sample", self.__guessed_z)

        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride

            # Embedding variable based on the latent value (Tensorboard stuff)
            #self.__embedding = tf.Variable(tf.zeros_like(self.__guessed_z), name="test_embedding")
            # self.__assignment = self.__embedding.assign(self.__guessed_z)
            self.__embedding = tf.Variable(tf.zeros([50, latent_size]), name="test_embedding")
            self.__assignment = self.__embedding.assign(
                tf.reshape(self.__guessed_z, [tf.shape(self.__x)[0], latent_size]))

            # Linear layer
            self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 7 * 7 * 32,
                                                 'z_matrix', do_summary=False)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]),
                                             do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(self.__z_develop_act, (5, 5), (14, 14), 32,16, 2,
                                                       name="dconv1",do_summary=False, pad='SAME')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out, do_summary=False)

            self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 16, 1, 2,
                                                       name="dconv2", do_summary=False, pad='SAME')

            # Model output
            self.__y = util.sigmoid(self.__conv_t1_out)
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], 28 * 28])


    @property
    def output(self):
        return self.__y

    @property
    def z_mean(self):
        return self.__w_mean

    @property
    def assignment(self):
        return self.__assignment

    @property
    def z_stddev(self):
        return self.__w_stddev

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input(self):
        return self.__x

    @property
    def image_in(self):
        return self.__x_image


class VAE_CNN_GEN(object):
    def __init__(self, img_size=28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, latent_size], name='LATENT_IN')

        with tf.name_scope('DECODER'):
            # Linear layer
            self.__z_develop = util.linear_layer(self.__x, latent_size, 7 * 7 * 32,
                                                 'z_matrix', do_summary=False)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]),
                                             do_summary=False)

            # Transpose API
            # Kernel, output size, in_volume, out_volume, stride
            self.__conv_t2_out = util.conv2d_transpose(self.__z_develop_act, (5, 5), (14, 14), 32,16, 2,
                                                       name="dconv1",do_summary=False, pad='SAME')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out, do_summary=False)

            self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 16, 1, 2,
                                                       name="dconv2", do_summary=False, pad='SAME')

            # Model output
            self.__y = util.sigmoid(self.__conv_t1_out)


    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        return self.__x


class CAE_CNN(object):
    def __init__(self, img_size = 28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # Calculating the convolution output:
            # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
            # H_out = 1 + (H_in+(2*pad)-K)/S
            # W_out = 1 + (W_in+(2*pad)-K)/S
            # CONV1: Input 28x28x1 after CONV 5x5 P:2 S:2 H_out: 1 + (28+4-5)/2 = 14, W_out= 1 + (28+4-5)/2 = 14
            self.__conv1 = util.conv2d(self.__x_image, 5, 5, 1, 16, 2, "conv1", pad='SAME',
                                       viewWeights=True, do_summary=False)
            self.__conv1_act = util.relu(self.__conv1, do_summary=False)

            # CONV2: Input 14x14x16 after CONV 5x5 P:0 S:2 H_out: 1 + (14+4-5)/2 = 7, W_out= 1 + (14+4-5)/2 = 7
            self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 16, 32, 2, "conv2", do_summary=False, pad='SAME')
            self.__conv2_act = util.relu(self.__conv2, do_summary=False)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv2_act, [tf.shape(self.__x)[0], 7 * 7 * 32])
            self.__guessed_z = util.linear_layer(self.__enc_out, 7 * 7 * 32, latent_size, "latent_var")
            tf.summary.histogram("latent", self.__guessed_z)


        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride

            # Embedding variable based on the latent value (Tensorboard stuff)
            #self.__embedding = tf.Variable(tf.zeros_like(self.__guessed_z), name="test_embedding")
            # self.__assignment = self.__embedding.assign(self.__guessed_z)
            self.__embedding = tf.Variable(tf.zeros([50, latent_size]), name="test_embedding")
            self.__assignment = self.__embedding.assign(
                tf.reshape(self.__guessed_z, [tf.shape(self.__x)[0], latent_size]))

            # Linear layer
            self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 7 * 7 * 32,
                                                 'z_matrix', do_summary=True)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]),
                                             do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(self.__z_develop_act, (5, 5), (14, 14), 32,16, 2,
                                                       name="dconv1",do_summary=False, pad='SAME')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out, do_summary=False)

            self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 16, 1, 2,
                                                       name="dconv2", do_summary=False, pad='SAME')

            # Model output
            self.__y = util.sigmoid(self.__conv_t1_out)
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], 28 * 28])


    @property
    def output(self):
        return self.__y


    @property
    def assignment(self):
        return self.__assignment


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input(self):
        return self.__x

    @property
    def image_in(self):
        return self.__x_image


class VAE_AutoEncoderSegnet(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, latent_size=40):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # Calculating the convolution output:
            # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
            # H_out = 1 + (H_in+(2*pad)-K)/S
            # W_out = 1 + (W_in+(2*pad)-K)/S
            # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
            if use_placeholder:
                # CONV 1 (Mark that want visualization)
                self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
            else:
                # CONV 1 (Mark that want visualization)
                self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

            self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
            self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

            # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
            self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
            self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
            self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

            # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
            self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 64, 2, "conv3", do_summary=False)
            self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
            self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

            # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
            self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 32, 1, "conv4", do_summary=False)
            self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
            self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

            # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
            self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 32, 32, 1, "conv5", do_summary=False)
            self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
            self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

            # CONV6: Input 21x21x32 after CONV 3x3 P:0 S:1 H_out: 1 + (21-3)/1 = 19, W_out=  1 + (21-3)/1 = 19
            self.__conv6 = util.conv2d(self.__conv5_act, 3, 3, 32, 32, 1, "conv6", do_summary=False)
            self.__conv6_bn = util.batch_norm(self.__conv6, training_mode, name='bn_c6')
            self.__conv6_act = util.relu(self.__conv6_bn, do_summary=False)

            # CONV7: Input 19x19x64 after CONV 3x3 P:0 S:1 H_out: 1 + (19-3)/2 = 9, W_out=  1 + (19-3)/2 = 9
            self.__conv7 = util.conv2d(self.__conv6_act, 3, 3, 32, 32, 2, "conv7", do_summary=False)
            self.__conv7_bn = util.batch_norm(self.__conv7, training_mode, name='bn_c6')
            self.__conv7_act = util.relu(self.__conv7_bn, do_summary=False)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv7_act, [tf.shape(self.__x)[0], 9 * 9 * 32])

            # Add linear ops for mean and variance
            self.__w_mean = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_mean")
            self.__w_stddev = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_stddev")

            # Generate normal distribution with dimensions [B, latent_size]
            self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)

            self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)
            tf.summary.histogram("latent_sample", self.__guessed_z)

            # Linear layer
            self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 9 * 9 * 32,
                                                 'z_matrix', do_summary=False)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 9, 9, 32]),
                                             do_summary=False)

        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride
            self.__conv_t7_out = util.conv2d_transpose(self.__z_develop_act,
                                                       (3, 3), (19, 19), 32, 32, 2, name="dconv1",do_summary=False)
            self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
            self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

            self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act,
                                                       (3, 3), (21, 21), 32, 32, 1, name="dconv2",do_summary=False)
            self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
            self.__conv_t6_out_act = util.relu(self.__conv_t6_out, do_summary=False)

            self.__conv_t5_out = util.conv2d_transpose(self.__conv_t6_out_act,
                                                       (3, 3), (23, 23), 32, 32, 1, name="dconv3",do_summary=False)
            self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
            self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

            self.__conv_t4_out = util.conv2d_transpose(
                self.__conv_t5_out_act,
                (3, 3), (25, 25), 32, 64, 1, name="dconv4",do_summary=False)
            self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
            self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

            self.__conv_t3_out = util.conv2d_transpose(
                self.__conv_t4_out_act,
                (5, 5), (53, 53), 64, 128, 2, name="dconv5",do_summary=False)
            self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
            self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(
                self.__conv_t3_out_act,
                (5, 5), (110, 110), 128, 64, 2, name="dconv6",do_summary=False)
            self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

            # Observe that the last deconv depth is the same as the number of classes
            self.__conv_t1_out = util.conv2d_transpose(
                self.__conv_t2_out_act,
                (5, 5), (img_size, img_size), 64, 3, 2, name="dconv7",do_summary=False)
            self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

            # Model output (It's not the segmentation yet...)
            self.__y = util.relu(self.__conv_t1_out_bn, do_summary = False)

            # Calculate flat tensor for Binary Cross entropy loss
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
            self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])


    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def z_mean(self):
        return self.__w_mean

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat

    @property
    def z_stddev(self):
        return self.__w_stddev

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class VAE_Segnet_Generator(object):
    def __init__(self, training_mode=False, img_size = 224, latent_size=40):
        self.__x = tf.placeholder(tf.float32, shape=[None, latent_size], name='LATENT_IN')

        with tf.name_scope('LATENT'):
            # Linear layer
            self.__z_develop = util.linear_layer(self.__x, latent_size, 9 * 9 * 32,
                                                 'z_matrix', do_summary=False)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 9, 9, 32]),
                                             do_summary=False)

        with tf.name_scope('DECODER'):

            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride
            self.__conv_t7_out = util.conv2d_transpose(self.__z_develop_act,
                                                       (3, 3), (19, 19), 32, 32, 2, name="dconv1",do_summary=False)
            self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
            self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

            self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act,
                                                       (3, 3), (21, 21), 32, 32, 1, name="dconv2",do_summary=False)
            self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
            self.__conv_t6_out_act = util.relu(self.__conv_t6_out, do_summary=False)

            self.__conv_t5_out = util.conv2d_transpose(self.__conv_t6_out_act,
                                                       (3, 3), (23, 23), 32, 32, 1, name="dconv3",do_summary=False)
            self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
            self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

            self.__conv_t4_out = util.conv2d_transpose(
                self.__conv_t5_out_act,
                (3, 3), (25, 25), 32, 64, 1, name="dconv4",do_summary=False)
            self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
            self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

            self.__conv_t3_out = util.conv2d_transpose(
                self.__conv_t4_out_act,
                (5, 5), (53, 53), 64, 128, 2, name="dconv5",do_summary=False)
            self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
            self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(
                self.__conv_t3_out_act,
                (5, 5), (110, 110), 128, 64, 2, name="dconv6",do_summary=False)
            self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

            # Observe that the last deconv depth is the same as the number of classes
            self.__conv_t1_out = util.conv2d_transpose(
                self.__conv_t2_out_act,
                (5, 5), (img_size, img_size), 64, 3, 2, name="dconv7",do_summary=False)
            self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

            # Model output (It's not the segmentation yet...)
            self.__y = util.relu(self.__conv_t1_out_bn, do_summary = False)

            # Calculate flat tensor for Binary Cross entropy loss
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
            self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])


    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        return self.__x

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


class CAE_AutoEncoderSegnet(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # Calculating the convolution output:
            # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
            # H_out = 1 + (H_in+(2*pad)-K)/S
            # W_out = 1 + (W_in+(2*pad)-K)/S
            # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
            if use_placeholder:
                # CONV 1 (Mark that want visualization)
                self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
            else:
                # CONV 1 (Mark that want visualization)
                self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

            self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
            self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

            # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
            self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
            self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
            self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

            # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
            self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 64, 2, "conv3", do_summary=False)
            self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
            self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

            # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
            self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 32, 1, "conv4", do_summary=False)
            self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
            self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

            # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
            self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 32, 32, 1, "conv5", do_summary=False)
            self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
            self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

            # CONV6: Input 21x21x32 after CONV 3x3 P:0 S:1 H_out: 1 + (21-3)/1 = 19, W_out=  1 + (21-3)/1 = 19
            self.__conv6 = util.conv2d(self.__conv5_act, 3, 3, 32, 32, 1, "conv6", do_summary=False)
            self.__conv6_bn = util.batch_norm(self.__conv6, training_mode, name='bn_c6')
            self.__conv6_act = util.relu(self.__conv6_bn, do_summary=False)

            # CONV7: Input 19x19x64 after CONV 3x3 P:0 S:1 H_out: 1 + (19-3)/2 = 9, W_out=  1 + (19-3)/2 = 9
            self.__conv7 = util.conv2d(self.__conv6_act, 3, 3, 32, 32, 2, "conv7", do_summary=False)
            self.__conv7_bn = util.batch_norm(self.__conv7, training_mode, name='bn_c6')
            self.__conv7_act = util.relu(self.__conv7_bn, do_summary=False)

        # with tf.name_scope('LATENT'):
        #     # Reshape: Input 7x7x32 after [7x7x32]
        #     self.__enc_out = tf.reshape(self.__conv7_act, [tf.shape(self.__x)[0], 9 * 9 * 32])
        #
        #     # Add linear ops for mean and variance
        #     self.__w_mean = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_mean")
        #     self.__w_stddev = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_stddev")
        #
        #     # Generate normal distribution with dimensions [B, latent_size]
        #     self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)
        #
        #     self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)
        #     tf.summary.histogram("latent_sample", self.__guessed_z)
        #
        #     # Linear layer
        #     self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 9 * 9 * 32,
        #                                          'z_matrix', do_summary=False)
        #     self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 9, 9, 32]),
        #                                      do_summary=False)

        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride
            self.__conv_t7_out = util.conv2d_transpose(self.__conv7_act,
                                                       (3, 3), (19, 19), 32, 32, 2, name="dconv1",do_summary=False)
            self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
            self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

            self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act,
                                                       (3, 3), (21, 21), 32, 32, 1, name="dconv2",do_summary=False)
            self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
            self.__conv_t6_out_act = util.relu(self.__conv_t6_out, do_summary=False)

            self.__conv_t5_out = util.conv2d_transpose(self.__conv_t6_out_act,
                                                       (3, 3), (23, 23), 32, 32, 1, name="dconv3",do_summary=False)
            self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
            self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

            self.__conv_t4_out = util.conv2d_transpose(
                self.__conv_t5_out_act,
                (3, 3), (25, 25), 32, 64, 1, name="dconv4",do_summary=False)
            self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
            self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

            self.__conv_t3_out = util.conv2d_transpose(
                self.__conv_t4_out_act,
                (5, 5), (53, 53), 64, 128, 2, name="dconv5",do_summary=False)
            self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
            self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(
                self.__conv_t3_out_act,
                (5, 5), (110, 110), 128, 64, 2, name="dconv6",do_summary=False)
            self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

            # Observe that the last deconv depth is the same as the number of classes
            self.__conv_t1_out = util.conv2d_transpose(
                self.__conv_t2_out_act,
                (5, 5), (img_size, img_size), 64, 3, 2, name="dconv7",do_summary=False)
            self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

            # Model output (It's not the segmentation yet...)
            self.__y = util.relu(self.__conv_t1_out_bn, do_summary = False)

            # Calculate flat tensor for Binary Cross entropy loss
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
            self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])


    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act

    class CAE_AutoEncoderSegnet(object):
        def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size=224):
            self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
            self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
            self.__use_placeholder = use_placeholder

            with tf.name_scope('ENCODER'):
                ##### ENCODER
                # Calculating the convolution output:
                # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
                # H_out = 1 + (H_in+(2*pad)-K)/S
                # W_out = 1 + (W_in+(2*pad)-K)/S
                # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
                if use_placeholder:
                    # CONV 1 (Mark that want visualization)
                    self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
                else:
                    # CONV 1 (Mark that want visualization)
                    self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

                self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
                self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

                # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
                self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
                self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
                self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

                # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
                self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 64, 2, "conv3", do_summary=False)
                self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
                self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

                # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
                self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 32, 1, "conv4", do_summary=False)
                self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
                self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

                # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
                self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 32, 32, 1, "conv5", do_summary=False)
                self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
                self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

                # CONV6: Input 21x21x32 after CONV 3x3 P:0 S:1 H_out: 1 + (21-3)/1 = 19, W_out=  1 + (21-3)/1 = 19
                self.__conv6 = util.conv2d(self.__conv5_act, 3, 3, 32, 32, 1, "conv6", do_summary=False)
                self.__conv6_bn = util.batch_norm(self.__conv6, training_mode, name='bn_c6')
                self.__conv6_act = util.relu(self.__conv6_bn, do_summary=False)

                # CONV7: Input 19x19x64 after CONV 3x3 P:0 S:1 H_out: 1 + (19-3)/2 = 9, W_out=  1 + (19-3)/2 = 9
                self.__conv7 = util.conv2d(self.__conv6_act, 3, 3, 32, 32, 2, "conv7", do_summary=False)
                self.__conv7_bn = util.batch_norm(self.__conv7, training_mode, name='bn_c6')
                self.__conv7_act = util.relu(self.__conv7_bn, do_summary=False)

            # with tf.name_scope('LATENT'):
            #     # Reshape: Input 7x7x32 after [7x7x32]
            #     self.__enc_out = tf.reshape(self.__conv7_act, [tf.shape(self.__x)[0], 9 * 9 * 32])
            #
            #     # Add linear ops for mean and variance
            #     self.__w_mean = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_mean")
            #     self.__w_stddev = util.linear_layer(self.__enc_out, 9 * 9 * 32, latent_size, "w_stddev")
            #
            #     # Generate normal distribution with dimensions [B, latent_size]
            #     self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)
            #
            #     self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)
            #     tf.summary.histogram("latent_sample", self.__guessed_z)
            #
            #     # Linear layer
            #     self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 9 * 9 * 32,
            #                                          'z_matrix', do_summary=False)
            #     self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 9, 9, 32]),
            #                                      do_summary=False)

            with tf.name_scope('DECODER'):
                ##### DECODER (At this point we have 1x18x64
                # Kernel, output size, in_volume, out_volume, stride
                self.__conv_t7_out = util.conv2d_transpose(self.__conv7_act,
                                                           (3, 3), (19, 19), 32, 32, 2, name="dconv1",do_summary=False)
                self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
                self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

                self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act, (3, 3), (21, 21), 32, 32, 1,
                                                           name="dconv2",
                                                           do_summary=False)
                self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
                self.__conv_t6_out_act = util.relu(self.__conv_t6_out, do_summary=False)

                self.__conv_t5_out = util.conv2d_transpose(self.__conv_t6_out_act, (3, 3), (23, 23), 32, 32, 1,
                                                           name="dconv3",
                                                           do_summary=False)
                self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
                self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

                self.__conv_t4_out = util.conv2d_transpose(
                    self.__conv_t5_out_act,
                    (3, 3), (25, 25), 32, 64, 1, name="dconv4", do_summary=False)
                self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
                self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

                self.__conv_t3_out = util.conv2d_transpose(
                    self.__conv_t4_out_act,
                    (5, 5), (53, 53), 64, 128, 2, name="dconv5", do_summary=False)
                self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
                self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

                self.__conv_t2_out = util.conv2d_transpose(
                    self.__conv_t3_out_act,
                    (5, 5), (110, 110), 128, 64, 2, name="dconv6", do_summary=False)
                self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
                self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

                # Observe that the last deconv depth is the same as the number of classes
                self.__conv_t1_out = util.conv2d_transpose(
                    self.__conv_t2_out_act,
                    (5, 5), (img_size, img_size), 64, 3, 2, name="dconv7", do_summary=False)
                self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

                # Model output (It's not the segmentation yet...)
                self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

                # Calculate flat tensor for Binary Cross entropy loss
                self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
                self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])

        @property
        def output(self):
            return self.__y

        @property
        def input(self):
            if self.__use_placeholder:
                return self.__x
            else:
                return None

        @property
        def label_in(self):
            if self.__use_placeholder:
                return self.__label
            else:
                return None

        @property
        def output_flat(self):
            return self.__y_flat

        @property
        def input_flat(self):
            return self.__x_flat

        @property
        def conv5(self):
            return self.__conv5_act

        @property
        def conv4(self):
            return self.__conv4_act

        @property
        def conv3(self):
            return self.__conv3_act

        @property
        def conv2(self):
            return self.__conv2_act

        @property
        def conv1(self):
            return self.__conv1_act


class CAE_AutoEncoderFE(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        self.__conv3 = util.conv2d(self.__conv2_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 2, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class CAE_AutoEncoderFE_MaxPool(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # Add Maxpool
        self.__conv2_mp_act = util.max_pool(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # Add Maxpool
        self.__conv4_mp_act = util.max_pool(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class CAE_AutoEncoderFE_MaxPool_MobileNet(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100, multiplier=1):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        #self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        #self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        #self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)
        self.__conv2_act = util.conv2d_separable(self.__conv1_act, 3, 3, 16, 16, 2, training_mode, "conv2",
                                                 do_summary=False, multiplier=multiplier)

        # Add Maxpool
        #self.__conv2_mp_act = util.max_pool(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        #self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        #self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        #self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)
        self.__conv3_act = util.conv2d_separable(self.__conv2_act, 3, 3, 16, 32, 1, training_mode, "conv3",
                                                 do_summary=False, multiplier=multiplier)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        #self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        #self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        #self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)
        self.__conv4_act = util.conv2d_separable(self.__conv3_act, 3, 3, 32, 64, 2, training_mode, "conv4",
                                                 do_summary=False, multiplier=multiplier)

        # Add Maxpool
        #self.__conv4_mp_act = util.max_pool(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        #self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        #self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        #self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)
        self.__conv5_act = util.conv2d_separable(self.__conv4_act, 3, 3, 64, 64, 1, training_mode, "conv5",
                                                 do_summary=False, multiplier=multiplier)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class CAE_AutoEncoderFE_MaxPoolUnpool(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # Add Maxpool
        self.__conv2_mp_act, self.__idx_mp2 = util.max_pool_argmax(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # Add Maxpool
        self.__conv4_mp_act, self.__idx_mp4 = util.max_pool_argmax(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 20x20x64
        # Now for upsampling we should have unpool-->conv-->BN-->Relu
        # Transposed Conv 20x20x64 ---> 22x22x64
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        # Unpool 22x22x64 ---> 44x44x64
        self.__conv_t5_out_act_unpool = util.unpool(self.__conv_t5_out_act, self.__idx_mp4, name="max_unpool1")

        # Transposed Conv 44x44x64 ---> 46x46x32
        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act_unpool,
            (3, 3), (46, 46), 64, 32, 1, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        # Transposed Conv 46x46x32 ---> 48x48x16
        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        # Unpool 48x48x16 ---> 96x96x16
        self.__conv_t3_out_act_unpool = util.unpool(self.__conv_t3_out_act, self.__idx_mp2, name="max_unpool2")

        # Transposed Conv 96x96x16 ---> 98x98x16
        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act_unpool,
            (3, 3), (98, 98), 16, 16, 1, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Transposed Conv 98x98x16 ---> 100x100x3
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class CAE_AutoEncoderFE_MaxPoolExpand(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # Add Maxpool
        self.__conv2_mp_act = util.max_pool(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # Add Maxpool
        self.__conv4_mp_act = util.max_pool(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        # Add Maxpool
        self.__conv5_mp_act, self.__conv5_mp_idx = util.max_pool_argmax(self.__conv5_act, 2, 2, 2, name="maxpool3")

        # CONV6: Input 10x10x64 after CONV 3x3 P:0 S:1 H_out: 1 + (10-3)/1 = 8, W_out=  1 + (10-3)/1 = 8
        self.__conv6 = util.conv2d(self.__conv5_mp_act, 3, 3, 64, 128, 1, "conv6", do_summary=False)
        self.__conv6_bn = util.batch_norm(self.__conv6, training_mode, name='bn_c6')
        self.__conv6_act = util.relu(self.__conv6_bn, do_summary=False)

        # CONV7: Input 8x8x64 after CONV 3x3 P:0 S:1 H_out: 1 + (8-3)/1 = 6, W_out=  1 + (8-3)/1 = 6
        self.__conv7 = util.conv2d(self.__conv6_act, 3, 3, 128, 256, 1, "conv7", do_summary=False)
        self.__conv7_bn = util.batch_norm(self.__conv7, training_mode, name='bn_c7')
        self.__conv7_act = util.relu(self.__conv7_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t7_out = util.conv2d_transpose(self.__conv7_act, (3, 3), (8, 8), 256, 128, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
        self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

        self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act, (3, 3), (10, 10), 128, 64, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
        self.__conv_t6_out_act = util.relu(self.__conv_t6_out_bn, do_summary=False)

        # Unpool 10x10x64 ---> 20x20x64
        self.__conv_5_unp = util.unpool(self.__conv_t6_out_act, self.__conv5_mp_idx, name="max_unpool1")

        self.__conv_t5_out = util.conv2d_transpose(self.__conv_5_unp, (3, 3), (22, 22), 64, 64, 1, name="dconv3",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv4", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv5", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv6", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv7", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act