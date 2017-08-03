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
        # Takes a d-dimensional vector of noise and upsamples it to become a 28 x 28 image
        # Convert hidden vector (ex 1x100) to a matrix that could be reshaped and match the input of a strided
        # deconvolution
        #h0 = tf.nn.relu(util.linear_layer_std(z, latent_size, 6272, 'g0'))
        h0 = tf.nn.relu(util.linear_std(z, 6272, 'g0', stddev=0.02))
        # 7x7x200
        h0_reshape = tf.reshape(h0, [-1, 7, 7, 128])

        conv_t1_out = util.conv2d_transpose(h0_reshape, (3, 3), (14, 14), 128, 256, 2,
                                                  name="dconv1", do_summary=False, pad='SAME')
        conv1_bn = util.batch_norm(conv_t1_out, training_mode, name='bn_c1')
        conv_t1_out_act = util.relu(conv1_bn, do_summary=False)

        conv_t2_out = util.conv2d_transpose(conv_t1_out_act, (3, 3), (28, 28), 256, 1, 2,
                                            name="dconv2", do_summary=False, pad='SAME')
        #conv2_bn = util.batch_norm(conv_t2_out, training_mode, name='bn_c2')
        conv_t2_out_act = util.relu(conv_t2_out, do_summary=False)

        # Return values between -1..1
        return tf.nn.tanh(conv_t2_out_act)


        # Crete discriminator model
    def discriminator(self, image, training_mode):
        # Example: input 28x28x1
        conv1 = util.conv2d(image, 5, 5, 1, 32, 1, "conv1", pad='SAME',viewWeights=False, do_summary=False)
        conv1_bn = util.batch_norm(conv1, training_mode, name='bn_c1')
        conv1_act = util.lrelu(conv1_bn, do_summary=False)

        avg_pool_1 = util.avg_pool(conv1_act, 2, 2, 2)

        # input: 14x14x32
        conv2 = util.conv2d(avg_pool_1, 5, 5, 32, 64, 1, "conv2", pad='SAME', viewWeights=False, do_summary=False)
        conv2_bn = util.batch_norm(conv2, training_mode, name='bn_c2')
        conv2_act = util.lrelu(conv2_bn, do_summary=False)

        avg_pool_2 = util.avg_pool(conv2_act, 2, 2, 2)

        # input: 7x7x64, output:1x1x1024
        conv3 = util.conv2d(avg_pool_2, 7, 7, 64, 1024, 1, "conv3", pad='VALID', viewWeights=False, do_summary=False)
        conv3_bn = util.batch_norm(conv3, training_mode, name='bn_c3')
        conv3_act = util.lrelu(conv3_bn, do_summary=False)

        # Reshape 1x1x1024 to [-1,1024]
        conv3_act_reshape = tf.reshape(conv3_act, [-1, 1024])
        linear_1 = util.linear_layer_std(conv3_act_reshape, 1024, 1)

        # Return values between 0..1 (Probabilities)
        #return util.sigmoid(linear_1, do_summary=False)

        # Return unbounded to give enough gradient to the generator
        return linear_1

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
