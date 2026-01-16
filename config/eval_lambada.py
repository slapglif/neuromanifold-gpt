# evaluate on LAMBADA benchmark
# zero-shot perplexity evaluation
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2'
benchmark = 'lambada'
