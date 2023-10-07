# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
eval_interval = 1000  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100  # don't print too too often

tensorboard_log = True
tensorboard_project = "runs/shakespeare-char"
tensorboard_run_name = "NanoGPT"

dataset = "shakespeare"
dataset_suffix = "_char.bin"
meta = "meta.pkl"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
weight_decay = 0.1

learning_rate = 5e-5
max_iters = 100000
lr_decay_iters = 100000  # make equal to max_iters usually
min_lr = 5e-6  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 200  # not super necessary potentially

overwrite = True
out_dir = f"out/training/shakespeare-char-nanogpt"