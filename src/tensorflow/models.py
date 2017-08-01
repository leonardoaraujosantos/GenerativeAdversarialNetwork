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
class DCGAN(object):
    def __init__(self, img_size=28, latent_size=100, hidden_size=1000, training_mode=True):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])
        self.__z = tf.placeholder(tf.float32, shape=(None, latent_size), name='Z')

        with tf.variable_scope('GAN'):
            # Generator
            with tf.variable_scope("G") as scope:
                # G will have the generator output tensor
                self.__G = self.generator(self.__z, hidden_size, training_mode)

            # Discriminator
            with tf.variable_scope("D") as scope:
                self.__D_real = self.discriminator(self.__x, training_mode)

                scope.reuse_variables()
                self.__D_fake = self.discriminator(self.__G, training_mode)

    # Create generator model
    def generator(self, z, hidden_size, training_mode):
        h0 = tf.nn.relu(util.linear_std(input, hidden_size, 'g0'))

        # Crete discriminator model
    def discriminator(self, image, training_mode):
        conv1 = util.conv2d(image, 5, 5, 1, 32, 1, "conv1", pad='SAME',viewWeights=False, do_summary=False)
        conv1_bn = util.batch_norm(conv1, training_mode, name='bn_c1')
        conv1_act = util.lrelu(conv1_bn, do_summary=False)

        avg_pool_1 = util.avg_pool(conv1_act, 2, 2, 2)

        conv2 = util.conv2d(avg_pool_1, 5, 5, 32, 64, 1, "conv2", pad='SAME', viewWeights=False, do_summary=False)
        conv2_bn = util.batch_norm(conv2, training_mode, name='bn_c1')
        conv2_act = util.lrelu(conv2_bn, do_summary=False)

        avg_pool_2 = util.avg_pool(conv2_act, 2, 2, 2)

        conv3 = util.conv2d(avg_pool_1, 7, 7, 64, 1024, 1, "conv3", pad='VALID', viewWeights=False, do_summary=False)
        conv3_bn = util.batch_norm(conv3, training_mode, name='bn_c1')
        conv3_act = util.lrelu(conv3_bn, do_summary=False)

        linear_1 = util.linear_layer(conv3_act, 1024, 1)
        return util.sigmoid(linear_1, do_summary=False)

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
