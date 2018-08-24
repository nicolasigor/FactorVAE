import tensorflow as tf
import numpy as np
import time
import datetime

from vae_dsprites_v2 import VAE
from mine import MINE


class FactorVAE(VAE):
    def __init__(self, gamma):
        # VAE parameters
        self.z_dim = 10
        self.gamma = gamma

        # Iterations parameters
        self.max_it = 300000
        self.stat_every = 500
        self.saving_every = 1e8

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = 'factorvaemine_dsprites_gamma' + str(gamma)

        self.data_path = '../database/'
        self.model_path = 'results/' + model_name + '_' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'

        # Data
        self.data_file = self.data_path + 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.data_train, self.data_test, self.all_imgs, self.all_factors, self.n_classes = self._data_init()
        self.iterator, self.handle, self.train_img_ph, self.iterator_train, self.test_img_ph, self.iterator_test =\
            self._iterator_init(batch_size=64)

        # Model setup
        ##FactorVAE
        self.input_vae, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_logit, self.dec_sigm, \
            self.dec_mean_logit, self.dec_mean_sigm = self._vae_init(inputs=self.iterator.get_next())
        self.z, self.z_hat = self._get_encoded_samples()
        ##MINE
        self.mine_model = MINE(params={}, z=self.z, z_hat=self.z_hat) 
        self.vae_loss, self.recon_loss, self.tc_estimate, self.mine_loss = self._loss_init()
        self.vae_train_step, self.mine_train_step = self._optimizer_init()


        # Savers
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tb_path + 'test')
        tf.summary.scalar('vae_loss', self.vae_loss)
        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('mine_loss', self.mine_loss)
        tf.summary.scalar('tc_estimate', self.tc_estimate)

        # Initialization of variables
        self.sess.run(tf.global_variables_initializer())

        # Initialization of iterators
        self.sess.run([self.iterator_train.initializer, self.iterator_test.initializer],
                      feed_dict={self.train_img_ph: self.data_train, self.test_img_ph: self.data_test})

        # Initialization of handles
        self.train_handle = self.sess.run(self.iterator_train.string_handle())
        self.test_handle = self.sess.run(self.iterator_test.string_handle())

    def train(self, final_evaluation=False):
        start_time = time.time()
        merged = tf.summary.merge_all()
        print("Beginning training")
        print("Beginning training", file=open(self.model_path + 'train.log', 'w'))
        it = 0

        while it < self.max_it:
            it += 1
            #Train step
            self.sess.run([self.vae_train_step, self.mine_train_step], feed_dict={self.handle: self.train_handle})

            if it % self.stat_every == 0:

                # Train evaluation
                vae_loss, recon_loss, tc_term, mine_loss, summ = self.sess.run(
                    [self.vae_loss, self.recon_loss, self.tc_estimate, self.mine_loss, merged],
                    feed_dict={self.handle: self.train_handle})
                print("Iteration %i (train):\n VAE loss %f - Rec loss %f - TC est %f - MINE loss %f" % (
                    it, vae_loss, recon_loss, tc_term, mine_loss), flush=True)
                print("Iteration %i (train):\n VAE loss %f - Rec loss %f - TC est %f - MINE loss %f" % (
                    it, vae_loss, recon_loss, tc_term, mine_loss), flush=True,
                      file=open(self.model_path + 'train.log', 'a'))
                self.train_writer.add_summary(summ, it)

                # Test evaluation
                vae_loss, recon_loss, tc_term, mine_loss, summ = self.sess.run(
                    [self.vae_loss, self.recon_loss, self.tc_estimate, self.mine_loss, merged],
                    feed_dict={self.handle: self.test_handle})
                print("Iteration %i (test):\n VAE loss %f - Rec loss %f - TC est %f - MINE loss %f" % (
                    it, vae_loss, recon_loss, tc_term, mine_loss), flush=True)
                print("Iteration %i (test):\n VAE loss %f - Rec loss %f - TC est %f - MINE loss %f" % (
                    it, vae_loss, recon_loss, tc_term, mine_loss), flush=True, file=open(self.model_path + 'train.log', 'a'))
                self.test_writer.add_summary(summ, it)

                time_usage = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
                print("Time usage: " + time_usage)
                print("Time usage: " + time_usage, file=open(self.model_path + 'train.log', 'a'))

            if it % self.saving_every == 0:
                save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
        print("Model saved to: %s" % save_path)
        print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        # Closing savers
        self.train_writer.close()
        self.test_writer.close()
        # Total time
        time_usage = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
        print("Total training time: " + time_usage)
        print("Total training time: " + time_usage, file=open(self.model_path + 'train.log', 'a'))

        if final_evaluation:
            print("Evaluating final model...")
            mean_dis_metric = self.evaluate_mean_disentanglement()
            recon_loss_test = self.evaluate_test_recon_loss()
            print("Mean Disentanglement Metric: " + str(mean_dis_metric),
                  file=open(self.model_path + 'train.log', 'a'))
            print("Test Reconstruction Loss: " + str(recon_loss_test),
                  file=open(self.model_path + 'train.log', 'a'))

    """
    def _discriminator_init(self, inputs, reuse=False):
        with tf.variable_scope("discriminator"):
            n_units = 1000
            disc_1 = tf.layers.dense(inputs=inputs, units=n_units, activation=tf.nn.leaky_relu, name="disc_1", reuse=reuse)
            disc_2 = tf.layers.dense(inputs=disc_1, units=n_units, activation=tf.nn.leaky_relu, name="disc_2", reuse=reuse)
            disc_3 = tf.layers.dense(inputs=disc_2, units=n_units, activation=tf.nn.leaky_relu, name="disc_3", reuse=reuse)
            disc_4 = tf.layers.dense(inputs=disc_3, units=n_units, activation=tf.nn.leaky_relu, name="disc_4", reuse=reuse)
            disc_5 = tf.layers.dense(inputs=disc_4, units=n_units, activation=tf.nn.leaky_relu, name="disc_5", reuse=reuse)
            disc_6 = tf.layers.dense(inputs=disc_5, units=n_units, activation=tf.nn.leaky_relu, name="disc_6", reuse=reuse)
            logits = tf.layers.dense(inputs=disc_6, units=2, name="disc_logits", reuse=reuse)
            probabilities = tf.nn.softmax(logits, name="disc_prob")

        return logits, probabilities
    """
    
    def _get_encoded_samples(self):
        with tf.name_scope("encoded_samples"):
            z_hat_list = []
            for i in range(self.z_dim):
                input_aux = self.iterator.get_next()
                aux_mean, aux_logvar = self._encoder_init(inputs=input_aux, reuse=True)
                with tf.name_scope("sampling"):
                    # Reparameterisation trick
                    with tf.name_scope("noise"):
                        noise = tf.random_normal(shape=tf.shape(aux_mean))
                    with tf.name_scope("variance"):
                        variance = tf.exp(aux_logvar / 2)
                    with tf.name_scope("reparam_trick"):
                        aux_samples = tf.add(aux_mean, (variance * noise))
                z_single_dim = aux_samples[:, i]
                z_hat_list.append(z_single_dim)
            permuted_samples = tf.stack(z_hat_list, axis=1)
            print(permuted_samples)
            real_samples = self.z_sample
        return real_samples, permuted_samples


    '''   
    # Get samples from the second batch
    input_aux = self.iterator.get_next()
    aux_mean, aux_logvar = self._encoder_init(inputs=input_aux, reuse=True)
    with tf.name_scope("sampling"):
        # Reparameterisation trick
        with tf.name_scope("noise"):
            noise = tf.random_normal(shape=tf.shape(aux_mean))
        with tf.name_scope("variance"):
            variance = tf.exp(aux_logvar / 2)
        with tf.name_scope("reparam_trick"):
            aux_samples = tf.add(aux_mean, (variance * noise))
    real_samples = self.z_sample
    # Discrimination of joint samples
    # logits_real, probs_real = self._discriminator_init(self.real_samples)
    with tf.name_scope("permute_dims"):
        # Use second batch to produce independent samples
        permuted_rows = []
        for i in range(aux_samples.get_shape()[1]):
            permuted_rows.append(tf.random_shuffle(aux_samples[:, i]))
        permuted_samples = tf.stack(permuted_rows, axis=1, )
        # permuted_samples = aux_samples
    # Discrimination of independent samples
    # logits_permuted, probs_permuted = self._discriminator_init(self.permuted_samples, reuse=True)
    '''


    def _loss_init(self):
        with tf.name_scope("loss"):
            with tf.name_scope("reconstruction_loss"):
                # Reconstruction loss is bernoulli in each pixel
                im_flat = tf.reshape(self.input_vae, shape=[-1, 64 * 64 * 1])
                logits_flat = tf.reshape(self.dec_logit, shape=[-1, 64 * 64 * 1])
                by_pixel_recon = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat, labels=im_flat)
                by_example_recon = tf.reduce_sum(by_pixel_recon, axis=1)
                recon_loss = tf.reduce_mean(by_example_recon, name="recon_loss")

            with tf.name_scope("kl_loss"):
                # KL against N(0,1) is 0.5 * sum_j ( var_j - logvar_j + mean^2_j - 1 )
                with tf.name_scope("variance"):
                    variance = tf.exp(self.enc_logvar)
                with tf.name_scope("squared_mean"):
                    squared_mean = self.enc_mean ** 2
                with tf.name_scope("kl_divergence"):
                    sum_argument = variance - self.enc_logvar + squared_mean
                    by_example_kl = 0.5 * tf.reduce_sum(sum_argument, axis=1) - self.z_dim
                    kl_divergence = tf.reduce_mean(by_example_kl, name="kl_divergence")

            with tf.name_scope("tc_loss"):
                # FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algo 2, where D(z) is probability of being real
                # Let PT be probability of being true, PF be probability of being false. Then we want log(PT/PF)
                # Since PT = exp(logit_T) / [exp(logit_T) + exp(logit_F)]
                # and  PT = exp(logit_F) / [exp(logit_T) + exp(logit_F)], we have that
                # log(PT/PF) = logit_T - logit_F
                #logit_t = logits_real[:, 0]
                #logit_f = logits_real[:, 1]
                tc_estimate = self.mine_model.loss#tf.reduce_mean(logit_t - logit_f, axis=0)
                tc_term = self.gamma * tc_estimate

            with tf.name_scope("total_vae_loss"):
                vae_loss = recon_loss + kl_divergence + tc_term
                
            """    
            with tf.name_scope("disc_loss"):
                real_samples_loss = tf.reduce_mean(tf.log(probs_real[:, 0]))
                permuted_samples_loss = tf.reduce_mean(tf.log(probs_permuted[:, 1]))
                disc_loss = - tf.add(0.5 * real_samples_loss,
                                     0.5 * permuted_samples_loss,
                                     name="disc_loss")
            """
            with tf.name_scope("mine_loss"):
                mine_loss = self.mine_model.loss
            
        return vae_loss, recon_loss, tc_estimate, mine_loss

    def _optimizer_init(self):
        with tf.name_scope("optimizer"):
            with tf.name_scope("vae_optimizer"):
                enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
                vae_vars = enc_vars + dec_vars
                vae_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                       beta1=0.9,
                                                       beta2=0.999)
                vae_train_step = vae_optimizer.minimize(self.vae_loss, var_list=vae_vars)
            """
            with tf.name_scope("disc_optimizer"):
                disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                        beta1=0.5,
                                                        beta2=0.9)
                disc_train_step = disc_optimizer.minimize(self.disc_loss, var_list=disc_vars)
            """
            with tf.name_scope("mine_optimizer"):
                mine_train_step = self.mine_model.train_step
                
        return vae_train_step, mine_train_step


if __name__ == "__main__":
    vae = FactorVAE(gamma=35)
    vae.train(final_evaluation=True)
