"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

# Handle --help
if '--help' in sys.argv:
    print("""
Poor Man's Configurator - Simple config override system

Usage:
    python train.py [config_file.py] [--key=value ...]

Examples:
    python train.py config/my_config.py
    python train.py --batch_size=32 --learning_rate=1e-4
    python train.py config/my_config.py --batch_size=32

Common Configuration Options:
    --attention=<type>         Attention mechanism: standard, soliton, sdr, fast-spectral
    --batch_size=<int>         Training batch size
    --learning_rate=<float>    Learning rate
    --n_layer=<int>            Number of transformer layers
    --n_head=<int>             Number of attention heads
    --n_embd=<int>             Embedding dimension
    --block_size=<int>         Context window size
    --max_iters=<int>          Maximum training iterations
    --model_type=<str>         Model type: neuromanifold or gpt
    --use_sdr=<bool>           Use SDR memory
    --use_kan=<bool>           Use KAN networks
    --compile_model=<bool>     Compile with torch.compile

Note: Boolean values should be 'True' or 'False' (case-sensitive)
      Run with a config file to see all available options for that configuration
""")
    sys.exit(0)

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
