"""
Export checkpoint metadata to JSON for experiment tracking.

This script demonstrates how to:
- Export checkpoint metadata (validation losses, timestamps, file sizes, training steps) to JSON
- Parse and display exported checkpoint information
- Track experiment history without loading full checkpoints
- Find best checkpoints by validation loss

Usage:
    # Export metadata from default 'out' directory
    python examples/export_checkpoint_metadata.py

    # Export from custom directory
    python examples/export_checkpoint_metadata.py --checkpoint_dir=out-ralph-1

    # Specify custom output file
    python examples/export_checkpoint_metadata.py --output_file=my_checkpoints.json

    # Export and display the results
    python examples/export_checkpoint_metadata.py --show_results=True

See also:
    - neuromanifold_gpt.utils.checkpoints: Checkpoint utilities
    - examples/evaluate_components.py: Loading and evaluating checkpoints
"""
import os
import sys
import json

# Handle --help before any imports that require dependencies
if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print("\nConfiguration parameters:")
    print("  --checkpoint_dir=<path>     Directory containing checkpoints (default: 'out')")
    print("  --output_file=<path>        Output JSON file path (default: 'checkpoints_metadata.json')")
    print("  --show_results=<bool>       Display exported metadata after export (default: True)")
    print("\nExamples:")
    print("  python examples/export_checkpoint_metadata.py")
    print("  python examples/export_checkpoint_metadata.py --checkpoint_dir=out-ralph-1")
    print("  python examples/export_checkpoint_metadata.py --output_file=my_checkpoints.json")
    sys.exit(0)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Import directly from file to avoid torch dependency in __init__.py
    import importlib.util
    checkpoints_path = os.path.join(
        os.path.dirname(__file__), '..',
        'neuromanifold_gpt', 'utils', 'checkpoints.py'
    )
    spec = importlib.util.spec_from_file_location(
        'neuromanifold_gpt.utils.checkpoints',
        checkpoints_path
    )
    checkpoints_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoints_module)
    export_checkpoints_metadata = checkpoints_module.export_checkpoints_metadata

except Exception as e:
    print(f"Error loading checkpoint utilities: {e}")
    print("\nMake sure you're running from the repository root:")
    print("  python examples/export_checkpoint_metadata.py")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration parameters
checkpoint_dir = 'out'  # directory containing checkpoint files
output_file = 'checkpoints_metadata.json'  # output JSON file
show_results = True  # display results after export

# Simple command-line argument parsing
for arg in sys.argv[1:]:
    if arg.startswith('--checkpoint_dir='):
        checkpoint_dir = arg.split('=', 1)[1]
    elif arg.startswith('--output_file='):
        output_file = arg.split('=', 1)[1]
    elif arg.startswith('--show_results='):
        show_results = arg.split('=', 1)[1].lower() in ('true', '1', 'yes')
# -----------------------------------------------------------------------------


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def display_checkpoint_metadata(json_file: str) -> None:
    """Display checkpoint metadata from exported JSON file.

    Args:
        json_file: Path to JSON file containing checkpoint metadata
    """
    if not os.path.exists(json_file):
        print(f"[Warning] Output file not found: {json_file}")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    checkpoints = data.get('checkpoints', [])

    print_section("Exported Metadata")

    print("\nExport Information:")
    print(f"  Exported at: {metadata.get('exported_at', 'N/A')}")
    print(f"  Directory: {metadata.get('directory', 'N/A')}")
    print(f"  Checkpoint count: {metadata.get('checkpoint_count', 0)}")

    if not checkpoints:
        print("\n[Info] No checkpoints found in directory")
        return

    print_section("Checkpoint Details")

    # Sort by val_loss (None values last)
    sorted_checkpoints = sorted(
        checkpoints,
        key=lambda x: (x['val_loss'] if x['val_loss'] is not None else float('inf'))
    )

    # Display checkpoint table
    print(f"\n{'#':<4} {'Filename':<40} {'Val Loss':<12} {'Step':<10} {'Age':<15} {'Size':<10}")
    print('-' * 90)

    for idx, ckpt in enumerate(sorted_checkpoints, 1):
        filename = ckpt['filename']
        val_loss = f"{ckpt['val_loss']:.4f}" if ckpt['val_loss'] is not None else "N/A"
        step = str(ckpt['training_step']) if ckpt['training_step'] is not None else "N/A"
        age = ckpt['age']
        size_mb = ckpt['size_mb']
        size = f"{size_mb:.1f}MB"

        # Truncate long filenames
        if len(filename) > 38:
            filename = filename[:35] + "..."

        print(f"{idx:<4} {filename:<40} {val_loss:<12} {step:<10} {age:<15} {size:<10}")

    # Find and highlight best checkpoint
    best_ckpt = None
    for ckpt in sorted_checkpoints:
        if ckpt['val_loss'] is not None:
            best_ckpt = ckpt
            break

    if best_ckpt:
        print("\n✓ Best checkpoint (lowest val_loss):")
        print(f"  Filename: {best_ckpt['filename']}")
        print(f"  Val Loss: {best_ckpt['val_loss']:.4f}")
        if best_ckpt['training_step'] is not None:
            print(f"  Training Step: {best_ckpt['training_step']}")
        print(f"  Age: {best_ckpt['age']}")
        print(f"  Size: {best_ckpt['size_mb']:.1f}MB")


def example_export_and_display() -> None:
    """Example: Export checkpoint metadata and display results."""
    print_section("Example: Export Checkpoint Metadata")

    print(f"\nExporting metadata from '{checkpoint_dir}' to '{output_file}'...")

    try:
        # Export checkpoint metadata to JSON
        export_checkpoints_metadata(checkpoint_dir, output_file)

        print(f"✓ Successfully exported checkpoint metadata to: {output_file}")

        # Display the results
        if show_results:
            display_checkpoint_metadata(output_file)

        print_section("Usage Examples")

        print("\n1. Load the JSON in Python:")
        print("   import json")
        print(f"   with open('{output_file}', 'r') as f:")
        print("       data = json.load(f)")
        print("   checkpoints = data['checkpoints']")

        print("\n2. Find best checkpoint:")
        print("   best = min([c for c in checkpoints if c['val_loss'] is not None],")
        print("              key=lambda x: x['val_loss'])")
        print("   print(f\"Best: {best['filename']} - Loss: {best['val_loss']}\")")

        print("\n3. Filter by training step:")
        print("   late_checkpoints = [c for c in checkpoints")
        print("                       if c['training_step'] and c['training_step'] > 1000]")

        print("\n4. Track checkpoint sizes:")
        print("   total_size_gb = sum(c['size_mb'] for c in checkpoints) / 1000")
        print("   print(f\"Total checkpoint storage: {total_size_gb:.2f}GB\")")

    except ValueError as e:
        print(f"✗ Error: {e}")
        print(f"\nDirectory '{checkpoint_dir}' does not exist or contains no checkpoints.")
        print("\nTo test this example:")
        print("  1. Train a model to create checkpoints:")
        print("     python train.py --out_dir=out")
        print("  2. Export the metadata:")
        print("     python examples/export_checkpoint_metadata.py")
        print("\nOr use an existing checkpoint directory:")
        print("     python examples/export_checkpoint_metadata.py --checkpoint_dir=path/to/checkpoints")

    except IOError as e:
        print(f"✗ Error writing output file: {e}")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run checkpoint metadata export example."""
    print("\n" + "="*70)
    print(" Checkpoint Metadata Export Example")
    print("="*70)
    print("\nThis script demonstrates exporting checkpoint metadata to JSON")
    print("for lightweight experiment tracking without loading full checkpoints.")

    example_export_and_display()

    print_section("Summary")

    print("\n✓ Example completed!")
    print("\nKey features of export_checkpoints_metadata():")
    print("  • Scans directory for checkpoint files (.pt, .ckpt, .pth)")
    print("  • Extracts validation loss from filenames")
    print("  • Extracts training step from filenames")
    print("  • Collects file metadata (timestamp, size)")
    print("  • Exports to structured JSON with metadata header")
    print("  • Enables tracking experiments without loading checkpoints")

    print("\nNext steps:")
    print("  1. Export metadata from your checkpoint directories")
    print("  2. Use JSON for experiment comparison and analysis")
    print("  3. Track checkpoint storage and cleanup old checkpoints")
    print("  4. Find best checkpoints by validation loss")
    print("\nFor more info:")
    print("  python examples/export_checkpoint_metadata.py --help")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
