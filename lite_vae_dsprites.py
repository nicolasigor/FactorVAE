import tensorflow as tf
import numpy as np

from vae_dsprites_v2 import VAE


class LiteVAE(VAE):
    """
    Class used for evaluation of a trained VAE. It doesn't have iterators, losses and optimization ops.
    FactorVAE and Our VAE checkpoints are also compatible since they share the encoder and decoder architecture.
    Note that we have disentanglement metric and traversals :)
    """
    def __init__(self, ckpt_path):
        # VAE parameters
        self.z_dim = 10

        # Model setup
        inputs_ph = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="input_ph")
        self.input_vae, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_logit, self.dec_sigm,\
            self.dec_mean_logit, self.dec_mean_sigm = self._vae_init(inputs=inputs_ph)
        self.vae_loss, self.recon_loss = self._loss_init()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Loading checkpoint at " + ckpt_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_path))

        self.data_path = 'database/'
        self.data_file = self.data_path + 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        _, self.data_test, self.all_imgs, self.all_factors, self.n_classes = self._data_init()

