from lite_vae_dsprites import LiteVAE
import time

# Restore from checkpoint
ckpt_path = "results/factorvae_dsprites_gamma35_20180807-200012/checkpoints"
# ckpt_path = "results/vae_dsprites_20180807-124029/checkpoints"
vae = LiteVAE(ckpt_path)

# Show disentanglement metric
start_time = time.time()
disen_metric = vae.evaluate_mean_disentanglement()
print("Elapsed Time:", time.time()-start_time, "s")

# Show reconstruction loss on test set
start_time = time.time()
recon_loss = vae.evaluate_test_recon_loss()
print("Elapsed Time:", time.time()-start_time, "s")

# Show traversals
# start_time = time.time()
# vae.get_traversals(example_index=10, show_figure=True)
# print("Elapsed Time:", time.time()-start_time, "s")

# Show reconstructions
# start_time = time.time()
# vae.get_recontructions(examples_index=[10, 100000, 70000], show_figure=True)
# print("Elapsed Time:", time.time()-start_time, "s")
