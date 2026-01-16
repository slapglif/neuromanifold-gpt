"""
A much shorter version of train.py for benchmarking
"""
import os
import sys
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from neuromanifold_gpt.utils.logging import get_logger
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.config.training import BenchConfig

# Initialize logger
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Load configuration with CLI overrides
config = load_config(BenchConfig, sys.argv[1:])
# -----------------------------------------------------------------------------

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if config.real_data:
    data_dir = os.path.join('data', config.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
    y = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = config.block_size, # how far back does the model look? i.e. context size
    n_layer = config.n_layer, n_head = config.n_head, n_embd = config.n_embd, # size of the model
    dropout = config.dropout, # for determinism
    bias = config.bias,
)
model = GPT(gptconf)
model.to(config.device)

optimizer = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.learning_rate, betas=(config.beta1, config.beta2), device_type=device_type)

if config.compile:
    logger.info("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

if config.profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = config.profiler_wait, config.profiler_warmup, config.profiler_active
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            logger.progress("Profiling", k + 1, num_steps)
            logger.metric("loss", lossf)

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([config.burnin_steps, config.benchmark_steps]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            stage_name = "Burn-in" if stage == 0 else "Benchmark"
            logger.progress(stage_name, k + 1, num_steps)
            logger.metric("loss", lossf)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(config.batch_size * 1 * num_steps, dt)
        if stage == 1:
            logger.metric("time_per_iteration", dt/num_steps*1000, unit="ms")
            logger.metric("MFU", mfu*100, unit="%")
