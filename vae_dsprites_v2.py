import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class VAE(object):
    def __init__(self):
        # VAE parameters
        self.z_dim = 10

        # Iterations parameters
        self.max_it = 300000
        self.stat_every = 500
        self.saving_every = 1e8

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = 'vae_dsprites'
        self.data_path = 'database/'
        self.model_path = 'results/' + model_name + '_' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'

        # Data
        self.data_file = self.data_path + 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.data_train, self.data_test,  self.all_imgs, self.all_factors, self.n_classes = self._data_init()
        self.iterator, self.handle, self.train_img_ph, self.iterator_train, self.test_img_ph, self.iterator_test =\
            self._iterator_init(batch_size=64)

        # Model setup
        self.input_vae, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_logit, self.dec_sigm, \
            self.dec_mean_logit, self.dec_mean_sigm = self._vae_init(inputs=self.iterator.get_next())
        self.vae_loss, self.recon_loss = self._loss_init()
        self.vae_train_step = self._optimizer_init()

        # Savers
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tb_path + 'test')
        tf.summary.scalar('vae_loss', self.vae_loss)
        tf.summary.scalar('recon_loss', self.recon_loss)

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
            self.sess.run(self.vae_train_step, feed_dict={self.handle: self.train_handle})

            if it % self.stat_every == 0:

                # Train evaluation
                vae_loss, recon_loss, summ = self.sess.run([self.vae_loss, self.recon_loss, merged],
                                                           feed_dict={self.handle: self.train_handle})
                print("Iteration %i (train):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss), flush=True)
                print("Iteration %i (train):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))
                self.train_writer.add_summary(summ, it)

                # Test evaluation
                vae_loss, recon_loss, summ = self.sess.run([self.vae_loss, self.recon_loss, merged],
                                                           feed_dict={self.handle: self.test_handle})
                print("Iteration %i (test):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss), flush=True)
                print("Iteration %i (test):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))
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

    def load_latest_checkpoint(self, params_path):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(params_path))

    def _data_init(self):
        # Find dataset here: https://github.com/deepmind/dsprites-dataset
        with np.load(self.data_file, encoding='bytes') as data:
            all_imgs = data['imgs']
            all_imgs = all_imgs[:, :, :, None]  # make into 4d tensor
            all_factors = data['latents_classes']
            all_factors = all_factors[:, 1:]  # Remove color factor
            n_classes = np.array([3, 6, 40, 32, 32])

        # 90% random test/train split
        n_data = all_imgs.shape[0]
        idx_random = np.random.permutation(n_data)
        data_train = all_imgs[idx_random[0: (9 * n_data) // 10]]
        data_test = all_imgs[idx_random[(9 * n_data) // 10:]]

        return data_train, data_test, all_imgs, all_factors, n_classes

    def _iterator_init(self, batch_size=64):
        with tf.name_scope("iterators"):
            # Generate TF Dataset objects for each split
            train_img_ph = tf.placeholder(dtype=tf.float32, shape=self.data_train.shape)
            test_img_ph = tf.placeholder(dtype=tf.float32, shape=self.data_test.shape)
            dataset_train = tf.data.Dataset.from_tensor_slices(train_img_ph)
            dataset_test = tf.data.Dataset.from_tensor_slices(test_img_ph)
            dataset_train = dataset_train.repeat()
            dataset_test = dataset_test.repeat()

            # Random batching
            dataset_train = dataset_train.shuffle(buffer_size=5000)
            dataset_train = dataset_train.batch(batch_size=batch_size)
            dataset_test = dataset_test.shuffle(buffer_size=1000)
            dataset_test = dataset_test.batch(batch_size=batch_size)

            # Prefetch
            dataset_train = dataset_train.prefetch(buffer_size=4)
            dataset_test = dataset_test.prefetch(buffer_size=4)

            # Iterator for each split
            iterator_train = dataset_train.make_initializable_iterator()
            iterator_test = dataset_test.make_initializable_iterator()

            # Global iterator
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, dataset_train.output_types, dataset_train.output_shapes)

        return iterator, handle, train_img_ph, iterator_train, test_img_ph, iterator_test

    def _vae_init(self, inputs):
        with tf.name_scope("vae"):
            # Input
            input_vae = inputs
            # Encoder network
            enc_mean, enc_logvar = self._encoder_init(inputs=input_vae)
            with tf.name_scope("sampling"):
                # Reparameterisation trick
                with tf.name_scope("noise"):
                    noise = tf.random_normal(shape=tf.shape(enc_mean))
                with tf.name_scope("variance"):
                    variance = tf.exp(enc_logvar / 2)
                with tf.name_scope("reparam_trick"):
                    z_sample = tf.add(enc_mean, (variance * noise))
            # Decoder network
            dec_logit, dec_sigm = self._decoder_init(inputs=z_sample)
            # Non-random decoder
            dec_mean_logit, dec_mean_sigm = self._decoder_init(inputs=enc_mean, reuse=True)

        return input_vae, enc_mean, enc_logvar, z_sample, dec_logit, dec_sigm, dec_mean_logit, dec_mean_sigm

    def _encoder_init(self, inputs, reuse=False):
        with tf.variable_scope("encoder"):
            e_1 = tf.layers.conv2d(inputs=inputs,
                                   filters=32,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_1",
                                   reuse=reuse)
            e_2 = tf.layers.conv2d(inputs=e_1,
                                   filters=32,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_2",
                                   reuse=reuse)
            e_3 = tf.layers.conv2d(inputs=e_2,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_3",
                                   reuse=reuse)
            e_4 = tf.layers.conv2d(inputs=e_3,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_4",
                                   reuse=reuse)
            with tf.name_scope("enc_flatten"):
                dim = np.prod(e_4.get_shape().as_list()[1:])
                e_4_flat = tf.reshape(e_4, shape=(-1, dim))
            e_5 = tf.layers.dense(inputs=e_4_flat,
                                  units=128,
                                  activation=None,
                                  name="enc_fc_1",
                                  reuse=reuse)
            enc_mean = tf.layers.dense(inputs=e_5,
                                       units=self.z_dim,
                                       activation=None,
                                       name="enc_fc_2_mean",
                                       reuse=reuse)
            enc_logvar = tf.layers.dense(inputs=e_5,
                                         units=self.z_dim,
                                         activation=None,
                                         name="enc_fc_2_logvar",
                                         reuse=reuse)

        return enc_mean, enc_logvar

    def _decoder_init(self, inputs, reuse=False):
        with tf.variable_scope("decoder"):
            d_1 = tf.layers.dense(inputs=inputs,
                                  units=128,
                                  activation=tf.nn.relu,
                                  name="dec_fc_1",
                                  reuse=reuse)
            d_2 = tf.layers.dense(inputs=d_1,
                                  units=4*4*64,
                                  activation=tf.nn.relu,
                                  name="dec_fc_2",
                                  reuse=reuse)
            with tf.name_scope("dec_reshape"):
                d_2_reshape = tf.reshape(d_2, shape=[-1, 4, 4, 64])
            d_3 = tf.layers.conv2d_transpose(inputs=d_2_reshape,
                                             filters=64,
                                             kernel_size=4,
                                             strides=2,
                                             activation=tf.nn.relu,
                                             padding="same",
                                             name="dec_upconv_1",
                                             reuse=reuse)
            d_4 = tf.layers.conv2d_transpose(inputs=d_3,
                                             filters=32,
                                             kernel_size=4,
                                             strides=2,
                                             activation=tf.nn.relu,
                                             padding="same",
                                             name="dec_upconv_2",
                                             reuse=reuse)
            d_5 = tf.layers.conv2d_transpose(inputs=d_4,
                                             filters=32,
                                             kernel_size=4,
                                             strides=2,
                                             activation=tf.nn.relu,
                                             padding="same",
                                             name="dec_upconv_3",
                                             reuse=reuse)
            dec_logit = tf.layers.conv2d_transpose(inputs=d_5,
                                                   filters=1,
                                                   kernel_size=4,
                                                   strides=2,
                                                   activation=None,
                                                   padding="same",
                                                   name="dec_upconv_4",
                                                   reuse=reuse)
            dec_sigm = tf.sigmoid(dec_logit, name="dec_sigmoid_out")

        return dec_logit, dec_sigm

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

            with tf.name_scope("total_vae_loss"):
                vae_loss = recon_loss + kl_divergence

        return vae_loss, recon_loss

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

        return vae_train_step

    def evaluate_mean_disentanglement(self):
        n_tries = 5
        dis_metric = 0
        print("Evaluating disentanglement with "+str(n_tries)+" tries.")
        for i in range(n_tries):
            this_disen_metric = self.evaluate_disentanglement()
            print(str(i+1)+"/"+str(n_tries)+" Disentanglement Metric: "+str(this_disen_metric))
            dis_metric = dis_metric + this_disen_metric
        dis_metric = dis_metric / n_tries
        print("Mean Disentanglement Metric: "+str(dis_metric))
        return dis_metric

    def evaluate_test_recon_loss(self):
        print("Evaluating reconstruction loss in test set.")
        recon_loss = 0
        batch_size = 64
        n_data_test = self.data_test.shape[0]
        n_batches = int(n_data_test/batch_size)
        print("Total batches:", n_batches)
        for i in range(n_batches):
            start_img = i*batch_size
            end_img = (i+1)*batch_size
            batch_imgs = self.data_test[start_img:end_img, :, :, :]
            # Reconstruction without random sampling
            dec_mean_logit = self.sess.run(self.dec_mean_logit, feed_dict={self.input_vae: batch_imgs})
            # Error according to non-random reconstruction
            this_recon_loss = self.sess.run(self.recon_loss,
                                            feed_dict={self.dec_logit: dec_mean_logit, self.input_vae: batch_imgs})
            recon_loss = recon_loss + this_recon_loss
            if (i + 1) % 100 == 0:
                print(str(i+1)+"/"+str(n_batches)+" evaluated.")
        recon_loss = recon_loss / n_batches
        print("Reconstruction loss: "+str(recon_loss))
        return recon_loss

    def compute_mean_kl_dim_wise(self, batch_mu, batch_logvar):
        # Shape of batch_mu is [batch, z_dim], same for batch_logvar
        # KL against N(0,1) is 0.5 * ( var_j - logvar_j + mean^2_j - 1 )
        variance = np.exp(batch_logvar)
        squared_mean = np.square(batch_mu)
        batch_kl = 0.5 * (variance - batch_logvar + squared_mean - 1)
        mean_kl = np.mean(batch_kl, axis=0)
        return mean_kl

    def evaluate_disentanglement(self, verbose=False):
        n_examples_per_vote = 100  # Generated examples when we fix a factor (L in paper)
        n_votes = 800  # Total number of training pairs for the classifier
        n_factors = self.n_classes.shape[0]
        n_votes_per_factor = int(n_votes / n_factors)

        # First, we get all the necessary codes at once
        all_mus = []
        all_logvars = []

        code_list = []
        # Fix a factor k
        for k_fixed in range(n_factors):
            code_list_per_factor = []
            # Generate training examples for this factor
            for _ in range(n_votes_per_factor):
                # Fix a value for this factor
                fixed_value = np.random.choice(self.n_classes[k_fixed])
                # Generate data with this factor fixed but all other factors varying randomly. Sample L examples.
                useful_examples_idx = np.where(self.all_factors[:, k_fixed] == fixed_value)[0]
                sampled_examples_idx = np.random.choice(useful_examples_idx, n_examples_per_vote)
                sampled_imgs = self.all_imgs[sampled_examples_idx, :, :, :]
                # Obtain their representations with the encoder
                feed_dict = {self.input_vae: sampled_imgs}
                code_mu, code_logvar = self.sess.run([self.enc_mean, self.enc_logvar], feed_dict=feed_dict)
                all_mus.append(code_mu)
                all_logvars.append(code_logvar)
                code_list_per_factor.append((code_mu, code_logvar))
            code_list.append(code_list_per_factor)

        # Concatenate every code
        all_mus = np.concatenate(all_mus, axis=0)
        all_logvars = np.concatenate(all_logvars, axis=0)

        # Now, lets compute the KL divergence of each dimension wrt the prior
        emp_mean_kl = self.compute_mean_kl_dim_wise(all_mus, all_logvars)

        # Throw the dimensions that collapsed to the prior
        kl_tol = 1e-2
        useful_dims = np.where(emp_mean_kl > kl_tol)[0]

        # Compute scales for useful dims
        scales = np.std(all_mus[:, useful_dims], axis=0)

        if verbose:
            print("Empirical mean for kl dimension-wise:")
            print(np.reshape(emp_mean_kl, newshape=(-1, 1)))
            print("Useful dimensions:", useful_dims, " - Total:", useful_dims.shape[0])
            print("Empirical Scales:", scales)

        # Dataset for classifier:
        d_values = []
        k_values = []
        # Fix a factor k
        for k_fixed in range(n_factors):
            # Generate training examples for this factor
            for i in range(n_votes_per_factor):
                # Get previously generated codes
                codes = code_list[k_fixed][i][0]
                # Keep only useful dims
                codes = codes[:, useful_dims]
                # Normalise each dimension by its empirical standard deviation over the full data
                # (or a large enough random subset)
                norm_codes = codes / scales  # dimension (L, z_dim)
                # Take the empirical variance in each dimension of these normalised representations
                emp_variance = np.var(norm_codes, axis=0)  # dimension (z_dim,), variance for each dimension of code
                # Then the index of the dimension with the lowest variance...
                d_min_var = np.argmin(emp_variance)
                # ...and the target index k provide one training input/output example for the classifier majority vote
                d_values.append(d_min_var)
                k_values.append(k_fixed)
        d_values = np.array(d_values)
        k_values = np.array(k_values)

        # Since both inputs and outputs lie in a discrete space, the optimal classifier is the majority-vote classifier
        # and the metric is the error rate of the classifier (actually they show the accuracy in the paper lol)
        v_matrix = np.zeros((useful_dims.shape[0], n_factors))
        for j in range(useful_dims.shape[0]):
            for k in range(n_factors):
                v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))
        if verbose:
            print("Votes:")
            print(v_matrix)

        # Majority vote is C_j = argmax_k V_jk
        classifier = np.argmax(v_matrix, axis=1)
        predicted_k = classifier[d_values]
        accuracy = np.sum(predicted_k == k_values) / n_votes

        return accuracy

    def get_traversals(self, example_index, show_figure=False):
        # Return a list of arrays (n_travers, 64, 64), one per dimension.
        # Dimensions are sorted in descending order of KL divergence

        feed_dict = {self.input_vae: self.all_imgs[[example_index], :, :, :]}
        z_base, logvar_base = self.sess.run([self.enc_mean, self.enc_logvar], feed_dict=feed_dict)
        # Sort by KL (in descending order)
        mean_kl = self.compute_mean_kl_dim_wise(z_base, logvar_base)
        sorted_dims = np.argsort(-mean_kl)

        trav_values = np.arange(-2, 2.1, 0.5)
        n_trav = len(trav_values)
        traversals = []
        z_base_batch = np.concatenate([np.copy(z_base) for _ in range(n_trav)], axis=0)
        for j in sorted_dims:
            z_sample = np.copy(z_base_batch)
            z_sample[:, j] = trav_values
            generated_images = self.sess.run(self.dec_sigm, feed_dict={self.z_sample: z_sample})
            traversals.append(generated_images[:, :, :, 0])  # shape (n_trav, 64, 64)

        if show_figure:
            # Prepare plot
            print("Preparing plot...")
            plt.figure(figsize=(n_trav, self.z_dim))
            gs1 = gridspec.GridSpec(self.z_dim, n_trav)
            gs1.update(wspace=0.02, hspace=0.02)
            for j in range(self.z_dim):
                for i in range(n_trav):
                    # Plot traversals for this z_j
                    ax1 = plt.subplot(gs1[j, i])
                    ax1.set_aspect('equal')
                    plt.axis('off')
                    ax1.imshow(traversals[j][i, :, :], cmap='gray')
            plt.show()

        return traversals

    def get_recontructions(self, examples_index, show_figure=False):
        # Originals in first row, reconstructions in second row

        originals = self.all_imgs[examples_index, :, :, :]
        # Non-random reconstructions
        reconstructions = self.sess.run(self.dec_mean_sigm, feed_dict={self.input_vae: originals})

        originals = originals[:, :, :, 0]
        reconstructions = reconstructions[:, :, :, 0]

        if show_figure:
            # Prepare plot
            n_examples = len(examples_index)
            plt.figure(figsize=(n_examples, 2))
            gs1 = gridspec.GridSpec(2, n_examples)
            gs1.update(wspace=0.02, hspace=0.02)
            # Plot originals
            for i in range(n_examples):
                ax1 = plt.subplot(gs1[0, i])
                ax1.set_aspect('equal')
                plt.axis('off')
                ax1.imshow(originals[i, :, :], cmap='gray')
            # Plot reconstructions
            for i in range(n_examples):
                ax1 = plt.subplot(gs1[1, i])
                ax1.set_aspect('equal')
                plt.axis('off')
                ax1.imshow(reconstructions[i, :, :], cmap='gray')
            plt.show()

        return originals, reconstructions  # Shapes [batch, 64, 64]


if __name__ == "__main__":
    vae = VAE()
    vae.train()
