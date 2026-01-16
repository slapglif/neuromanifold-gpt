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
from neuromanifold_gpt.errors import ValidationError

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
        if arg.startswith('--'):
            raise ValidationError(
                problem="Invalid config file argument format",
                cause=f"Config file argument '{arg}' cannot start with '--'",
                recovery="Use either: python script.py config_file.py OR python script.py --key=value"
            )
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        if not arg.startswith('--'):
            raise ValidationError(
                problem="Invalid override argument format",
                cause=f"Override argument '{arg}' must start with '--'",
                recovery="Use format: --key=value (e.g., --batch_size=32)"
            )
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
            if type(attempt) != type(globals()[key]):
                raise ValidationError(
                    problem="Configuration type mismatch",
                    cause=f"Cannot override '{key}': expected {type(globals()[key]).__name__}, got {type(attempt).__name__}",
                    recovery=f"Provide a value of type {type(globals()[key]).__name__} (current value: {globals()[key]})"
                )
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
