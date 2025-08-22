"""
predict.py - Grand Challenge compatible inference script for Noise2Void denoising with NAFNet
"""

from pathlib import Path
import json
import tifffile
import numpy as np
import torch
import sys
import os

# Add N2V_Vanilla package to sys.path (adjust based on where the package is in the container)
sys.path.append("/opt/app/N2V_Vanilla")
from N2V_Vanilla.models import NAFNet, create_nafnet
from N2V_Vanilla.utils import denoise_full_image, load_checkpoint

# Constants for input and output paths
INPUT_PATH = Path("/input/images/image-stack-unstructured-noise")
OUTPUT_PATH = Path("/output/images/image-stack-denoised")

# Create output directory with error handling
try:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {OUTPUT_PATH}")
except PermissionError as e:
    print(f"Permission error creating output directory: {e}")
    # Try alternative path
    OUTPUT_PATH = Path("/output")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Using alternative output path: {OUTPUT_PATH}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    raise

def load_model():
    """Load the trained NAFNet model from checkpoint"""
    model_path = Path("/opt/app/model/best_model_NafNet_Noise2Void.pth")
    print(f"Loading model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model configuration from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create NAFNet model using config
    nafnet_config = config.get('nafnet_config', {
        'img_channel': 1,
        'width': 32,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 28],
        'dec_blk_nums': [1, 1, 1]
    })  # Fallback to medium model config if not in checkpoint
    model = create_nafnet(nafnet_config).to(device)

    # Load model weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, config.get('patch_size', 128)

def read_image(image_path: Path) -> np.ndarray:
    """Read and preprocess input image"""
    print(f"Reading image: {image_path}")
    input_array = tifffile.imread(image_path)
    
    input_array = input_array.astype(np.float32)
    print(f"Loaded image shape: {input_array.shape}")
    
    return input_array

def run_inference(model, input_array, device, patch_size):
    """Run inference on the input array using patch-based denoising"""
    print("Running inference...")
    print(f"Input shape: {input_array.shape}")
    
    # Check if input is 3D and needs slice-by-slice processing
    if input_array.ndim == 3 and not (input_array.shape[-1] in [1, 3, 4]):
        # Assume 3D image with shape (depth, height, width)
        print("Processing 3D volume slice-by-slice...")
        depth, height, width = input_array.shape
        
        # Initialize output stack
        output_stack = np.zeros((depth, height, width), dtype=np.float32)
        
        # Process each slice
        for z in range(depth):
            noisy_slice = input_array[z, :, :]  # Shape: (height, width)
            
            # Denoise using patch-based inference
            denoised_slice = denoise_full_image(
                model,
                noisy_slice,
                patch_size=patch_size,
                stride=patch_size // 2,  # 50% overlap
                device=device
            )
            
            if denoised_slice.shape != (height, width):
                raise ValueError(f"Denoised slice {z} has incorrect shape: {denoised_slice.shape}, expected: ({height}, {width})")
            
            output_stack[z, :, :] = denoised_slice
        
        print(f"3D Output shape: {output_stack.shape}")
        return output_stack
    else:
        # Standard 2D processing
        denoised_image = denoise_full_image(
            model,
            input_array,
            patch_size=patch_size,
            stride=patch_size // 2,  # 50% overlap
            device=device
        )
        
        print(f"2D Output shape: {denoised_image.shape}")
        return denoised_image

def save_output(array, output_path):
    """Save the processed array as 16-bit TIFF"""
    print(f"Saving output to: {output_path}")
    
    # Convert back to 16-bit
    if array.ndim == 3:
        # 3D volume: convert each slice
        output_img = np.clip(array, 0, 65535).round().astype(np.uint16)
    else:
        # 2D image
        output_img = np.clip(array, 0, 65535).round().astype(np.uint16)

    with tifffile.TiffWriter(output_path) as out:
        out.write(
            output_img,
            resolutionunit=2  # Important flag for Grand Challenge
        )

def inference_handler():
    """Main handler for processing images"""
    # Show torch cuda info
    _show_torch_cuda_info()

    # Load model
    model, device, patch_size = load_model()

    # Find input files
    input_files = sorted(INPUT_PATH.glob("*.tif")) + sorted(INPUT_PATH.glob("*.tiff"))
    print(f"Found input files: {input_files}")
    
    if not input_files:
        print(f"No .tif or .tiff files found in expected input folders under {INPUT_PATH}")
        return 1

    for input_file in input_files:
        # Read image
        input_array = read_image(input_file)

        # Run inference
        result = run_inference(model, input_array, device, patch_size)

        # Save output
        output_path = OUTPUT_PATH / input_file.name
        save_output(result, output_path)

    return 0

def run():
    """Entry point for Grand Challenge"""
    # Get interface key
    interface_key = get_interface_key()
    print(f"Interface key: {interface_key}")

    handler = inference_handler
    return handler()

def get_interface_key():
    """Get interface key from inputs.json"""
    try:
        inputs = load_json_file(INPUT_PATH.parent.parent / "inputs.json")
        socket_slugs = [sv["interface"]["slug"] for sv in inputs]
        return tuple(sorted(socket_slugs))
    except Exception as e:
        print(f"Warning: Could not load inputs.json: {e}")
        return ("stacked-images-subject-to-unstructured-noise",)

def load_json_file(location):
    """Load JSON file"""
    with open(location, "r") as f:
        return json.loads(f.read())

def _show_torch_cuda_info():
    """Show CUDA information"""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    raise SystemExit(run())