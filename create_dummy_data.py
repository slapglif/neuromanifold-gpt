#!/usr/bin/env python3
import numpy as np

# Create dummy train and val data for benchmarking
data = np.random.randint(0, 50304, size=10000, dtype=np.uint16)
data.tofile('data/openwebtext/train.bin')

val_data = data[:1000]
val_data.tofile('data/openwebtext/val.bin')

print('Created dummy data files')
