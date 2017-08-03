import tensorflow as tf
import model_util as util

# Deep convolutional generative adversarial networks
# References
# https://medium.com/@liyin_27935/dcgan-79af14a1c247
# https://github.com/carpedm20/DCGAN-tensorflow
# https://github.com/pytorch/examples/tree/master/dcgan
# http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
# https://www.youtube.com/watch?v=0VPQHbMvGzg
# https://github.com/llSourcell/Generative_Adversarial_networks_LIVE/blob/master/EZGAN.ipynb
# https://www.youtube.com/watch?v=AJVyzd0rqdc
# https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
# https://github.com/osh/KerasGAN
# https://github.com/rajathkumarmp/DCGAN
# https://github.com/jonbruner/generative-adversarial-networks.git
class DCGAN(object):
    def __init__(self, img_size=28, latent_size=100, training_mode=True):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 1], name='IMAGE_IN')
        self.__z = tf.placeholder(tf.float32, shape=(None, latent_size), name='Z')

        with tf.variable_scope('GAN'):
            # Generator
            with tf.variable_scope("G") as scope:
                # G will have the generator output tensor
                self.__G = self.generator(self.__z, training_mode, latent_size)

            # Discriminator
            with tf.variable_scope("D") as scope:
                self.__D_real = self.discriminator(self.__x, training_mode)

                scope.reuse_variables()
                self.__D_fake = self.discriminator(self.__G, training_mode)

    # Create generator model
    def generator(self, z, training_mode, latent_size):
        g_w1 = tf.get_variable('g_w1', [latent_size, 3136], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features
        g_w2 = tf.get_variable('g_w2', [3, 3, 1, latent_size / 2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [latent_size / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])

        # Generate 25 features
        g_w3 = tf.get_variable('g_w3', [3, 3, latent_size / 2, latent_size / 4], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [latent_size / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])

        # Final convolution with one output channel
        g_w4 = tf.get_variable('g_w4', [1, 1, latent_size / 4, 1], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + g_b4
        g4 = tf.sigmoid(g4)

        # Dimensions of g4: batch_size x 28 x 28 x 1
        return g4


        # Crete discriminator model
    def discriminator(self, image, training_mode):
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4



    @property
    def output_generator(self):
        return self.__G

    @property
    def output_discriminator_real(self):
        return self.__D_real

    @property
    def output_discriminator_fake(self):
        return self.__D_fake

    @property
    def generator_input(self):
        return self.__z

    @property
    def discriminator_input_real(self):
        return self.__x

    @property
    def trainable_variables(self):
        return tf.trainable_variables()
