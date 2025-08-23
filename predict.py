"""
predict.py - Ultra-robust Grand Challenge compatible inference script for Noise2Void denoising with NAFNet
Handles 2D, 3D, and 4D images by processing them slice-by-slice with the 2D model
"""

from pathlib import Path
import json
import tifffile
import numpy as np
import torch
import sys
import os
import traceback
import warnings
from typing import Union, Tuple, Optional

# Add N2V_Vanilla package to sys.path
sys.path.append("/opt/app/N2V_Vanilla")
from N2V_Vanilla.models import NAFNet, create_nafnet
from N2V_Vanilla.utils import denoise_full_image, load_checkpoint

# Constants for input and output paths
INPUT_PATH = Path("/input/images/stacked-neuron-images-with-noise")
OUTPUT_PATH = Path("/output/images/stacked-neuron-images-with-reduced-noise")

# Processing constants
MIN_PATCH_SIZE = 32
MAX_PATCH_SIZE = 1024
DEFAULT_PATCH_SIZE = 128
MEMORY_THRESHOLD_GB = 2.0  # GPU memory threshold for adaptive processing

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
    """Load the trained NAFNet model from checkpoint with robust error handling"""
    model_path = Path("/opt/app/model/best_model_NafNet_Noise2Void.pth")
    print(f"Loading model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    try:
        # Load model configuration from checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})

        # Create NAFNet model using config with fallbacks
        nafnet_config = config.get('nafnet_config')
        
        if nafnet_config is None:
            print("Warning: No nafnet_config found, using fallback configuration")
            nafnet_config = {
                'img_channel': 1,
                'width': 64,
                'middle_blk_num': 2,
                'enc_blk_nums': [2, 2, 28],
                'dec_blk_nums': [2, 2, 2]
            }
        
        print(f"Model config: {nafnet_config}")
        model = create_nafnet(nafnet_config).to(device)

        # Load model weights with error handling
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed ({e}), trying non-strict loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        model.eval()
        
        # Test model with dummy input
        test_input = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            _ = model(test_input)
        print("Model test passed!")
        
        patch_size = config.get('patch_size', DEFAULT_PATCH_SIZE)
        patch_size = max(MIN_PATCH_SIZE, min(MAX_PATCH_SIZE, patch_size))
        
        return model, device, patch_size
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

def read_image_robust(image_path: Path) -> np.ndarray:
    """Robust image reading with multiple fallbacks and format detection"""
    print(f"Reading image: {image_path}")
    
    try:
        # Try tifffile first (most common)
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            input_array = tifffile.imread(str(image_path))
            print(f"Successfully read TIFF with tifffile")
        
        # Try reading .mha/.mhd files
        elif image_path.suffix.lower() in ['.mha', '.mhd']:
            try:
                import SimpleITK as sitk
                image_sitk = sitk.ReadImage(str(image_path))
                input_array = sitk.GetArrayFromImage(image_sitk)
                print(f"Successfully read {image_path.suffix} with SimpleITK")
            except ImportError:
                print("SimpleITK not available, trying alternative methods...")
                # Fallback to tifffile (sometimes works with .mha)
                input_array = tifffile.imread(str(image_path))
                print(f"Read {image_path.suffix} with tifffile fallback")
        
        # Try reading other formats
        else:
            # Try multiple readers
            readers = [
                lambda p: tifffile.imread(str(p)),
                lambda p: np.load(str(p)) if p.suffix == '.npy' else None,
            ]
            
            input_array = None
            for reader in readers:
                try:
                    result = reader(image_path)
                    if result is not None:
                        input_array = result
                        break
                except:
                    continue
            
            if input_array is None:
                raise ValueError(f"Could not read file with any available reader")
    
    except Exception as e:
        print(f"Error reading image: {e}")
        raise
    
    # Convert to float32 and validate
    input_array = np.array(input_array, dtype=np.float32)
    
    # Handle potential issues with the array
    if input_array.size == 0:
        raise ValueError("Empty image array")
    
    if np.any(np.isnan(input_array)):
        print("Warning: Image contains NaN values, replacing with zeros")
        input_array = np.nan_to_num(input_array, nan=0.0)
    
    if np.any(np.isinf(input_array)):
        print("Warning: Image contains infinite values, clipping to valid range")
        input_array = np.clip(input_array, 0, 65535)
    
    print(f"Loaded image shape: {input_array.shape}, dtype: {input_array.dtype}")
    print(f"Value range: [{input_array.min():.2f}, {input_array.max():.2f}]")
    
    return input_array

def analyze_image_dimensions(input_array: np.ndarray) -> dict:
    """Analyze image dimensions and determine processing strategy"""
    shape = input_array.shape
    ndim = input_array.ndim
    
    analysis = {
        'shape': shape,
        'ndim': ndim,
        'processing_strategy': None,
        'iteration_axes': [],
        'slice_shape': None,
        'total_slices': 1
    }
    
    print(f"Analyzing image dimensions: {shape}")
    
    if ndim == 2:
        # Standard 2D image: (height, width)
        analysis['processing_strategy'] = '2D'
        analysis['slice_shape'] = shape
        print("Strategy: Direct 2D processing")
        
    elif ndim == 3:
        # 3D image: could be (depth, height, width) or (height, width, channels)
        
        # Heuristic to determine if last dimension is channels
        if shape[-1] <= 4 and shape[-1] < min(shape[:-1]):
            # Likely (height, width, channels)
            analysis['processing_strategy'] = '2D_channels'
            analysis['slice_shape'] = shape[:-1]  # (height, width)
            analysis['channels'] = shape[-1]
            analysis['total_slices'] = shape[-1]
            print(f"Strategy: 2D with {shape[-1]} channels")
        else:
            # Likely (depth, height, width) - 3D volume
            analysis['processing_strategy'] = '3D_volume'
            analysis['iteration_axes'] = [0]  # Iterate over depth
            analysis['slice_shape'] = shape[1:]  # (height, width)
            analysis['total_slices'] = shape[0]
            print(f"Strategy: 3D volume with {shape[0]} slices of {shape[1:]}")
            
    elif ndim == 4:
        # 4D image: likely (time/batch, depth, height, width) or (depth, height, width, channels)
        
        if shape[-1] <= 4 and shape[-1] < min(shape[:-1]):
            # Likely (depth, height, width, channels)
            analysis['processing_strategy'] = '3D_volume_channels'
            analysis['iteration_axes'] = [0, -1]  # Iterate over depth and channels
            analysis['slice_shape'] = shape[1:-1]  # (height, width)
            analysis['total_slices'] = shape[0] * shape[-1]
            print(f"Strategy: 3D volume with {shape[0]} slices and {shape[-1]} channels")
        else:
            # Likely (time/batch, depth, height, width)
            analysis['processing_strategy'] = '4D_time_volume'
            analysis['iteration_axes'] = [0, 1]  # Iterate over time and depth
            analysis['slice_shape'] = shape[2:]  # (height, width)
            analysis['total_slices'] = shape[0] * shape[1]
            print(f"Strategy: 4D time-volume with {shape[0]} timepoints, {shape[1]} slices each")
    
    elif ndim == 5:
        # 5D image: likely (time, depth, height, width, channels)
        analysis['processing_strategy'] = '5D_time_volume_channels'
        analysis['iteration_axes'] = [0, 1, -1]  # Iterate over time, depth, and channels
        analysis['slice_shape'] = shape[2:-1]  # (height, width)
        analysis['total_slices'] = shape[0] * shape[1] * shape[-1]
        print(f"Strategy: 5D time-volume with channels")
    
    else:
        raise ValueError(f"Unsupported image dimensions: {ndim}D. Maximum supported is 5D.")
    
    return analysis

def get_memory_usage_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def adaptive_patch_size(image_shape: Tuple[int, int], base_patch_size: int, device: torch.device) -> int:
    """Adaptively determine patch size based on image size and available memory"""
    height, width = image_shape
    
    # Get available GPU memory
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        used_memory = torch.cuda.memory_allocated() / 1e9
        available_memory = total_memory - used_memory
        
        # Estimate memory needed for patch processing (rough estimate)
        memory_per_patch_gb = (base_patch_size ** 2) * 4 * 8 / 1e9  # 4 bytes per float, 8x overhead
        
        if available_memory < MEMORY_THRESHOLD_GB:
            # Reduce patch size for low memory
            new_patch_size = max(MIN_PATCH_SIZE, base_patch_size // 2)
            print(f"Low GPU memory ({available_memory:.1f}GB), reducing patch size to {new_patch_size}")
            return new_patch_size
    
    # Adjust patch size based on image dimensions
    min_dim = min(height, width)
    if min_dim < base_patch_size:
        new_patch_size = max(MIN_PATCH_SIZE, min_dim)
        print(f"Small image dimension ({min_dim}), adjusting patch size to {new_patch_size}")
        return new_patch_size
    
    return base_patch_size

def process_2d_slice(model, slice_2d: np.ndarray, device: torch.device, patch_size: int) -> np.ndarray:
    """Process a single 2D slice with the model"""
    if slice_2d.ndim != 2:
        raise ValueError(f"Expected 2D slice, got {slice_2d.ndim}D")
    
    # Adaptive patch size based on slice dimensions
    adaptive_size = adaptive_patch_size(slice_2d.shape, patch_size, device)
    
    try:
        # Use the denoise_full_image function with patch-based processing
        denoised = denoise_full_image(
            model,
            slice_2d,
            patch_size=adaptive_size,
            stride=adaptive_size // 2,  # 50% overlap
            device=device
        )
        
        # Ensure output shape matches input
        if denoised.shape != slice_2d.shape:
            print(f"Warning: Output shape {denoised.shape} != input shape {slice_2d.shape}")
            # Try to resize if needed (should not happen with proper patch processing)
            if denoised.size == slice_2d.size:
                denoised = denoised.reshape(slice_2d.shape)
            else:
                raise ValueError(f"Output size mismatch: {denoised.shape} vs {slice_2d.shape}")
        
        return denoised
        
    except Exception as e:
        print(f"Error processing 2D slice: {e}")
        print(f"Slice shape: {slice_2d.shape}, patch_size: {adaptive_size}")
        raise

def run_inference_robust(model, input_array: np.ndarray, device: torch.device, patch_size: int) -> np.ndarray:
    """Ultra-robust inference that handles any dimensional input by processing 2D slices"""
    
    print("Starting robust inference...")
    analysis = analyze_image_dimensions(input_array)
    
    strategy = analysis['processing_strategy']
    total_slices = analysis['total_slices']
    
    print(f"Processing {total_slices} total 2D slices...")
    
    if strategy == '2D':
        # Direct 2D processing
        return process_2d_slice(model, input_array, device, patch_size)
    
    elif strategy == '2D_channels':
        # Process each channel separately
        height, width, channels = input_array.shape
        result = np.zeros_like(input_array)
        
        for c in range(channels):
            print(f"Processing channel {c+1}/{channels}")
            slice_2d = input_array[:, :, c]
            denoised_slice = process_2d_slice(model, slice_2d, device, patch_size)
            result[:, :, c] = denoised_slice
            
            # Memory cleanup
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return result
    
    elif strategy == '3D_volume':
        # Process each depth slice
        depth, height, width = input_array.shape
        result = np.zeros_like(input_array)
        
        for z in range(depth):
            print(f"Processing slice {z+1}/{depth}")
            slice_2d = input_array[z, :, :]
            denoised_slice = process_2d_slice(model, slice_2d, device, patch_size)
            result[z, :, :] = denoised_slice
            
            # Memory cleanup every few slices
            if (z + 1) % 5 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
        
        return result
    
    elif strategy == '3D_volume_channels':
        # Process each (depth, channel) combination
        depth, height, width, channels = input_array.shape
        result = np.zeros_like(input_array)
        
        slice_count = 0
        for z in range(depth):
            for c in range(channels):
                slice_count += 1
                print(f"Processing slice {slice_count}/{total_slices} (depth={z+1}/{depth}, channel={c+1}/{channels})")
                slice_2d = input_array[z, :, :, c]
                denoised_slice = process_2d_slice(model, slice_2d, device, patch_size)
                result[z, :, :, c] = denoised_slice
                
                # Memory cleanup
                if slice_count % 10 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return result
    
    elif strategy == '4D_time_volume':
        # Process each (time, depth) combination
        time_points, depth, height, width = input_array.shape
        result = np.zeros_like(input_array)
        
        slice_count = 0
        for t in range(time_points):
            for z in range(depth):
                slice_count += 1
                print(f"Processing slice {slice_count}/{total_slices} (time={t+1}/{time_points}, depth={z+1}/{depth})")
                slice_2d = input_array[t, z, :, :]
                denoised_slice = process_2d_slice(model, slice_2d, device, patch_size)
                result[t, z, :, :] = denoised_slice
                
                # Memory cleanup
                if slice_count % 10 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return result
    
    elif strategy == '5D_time_volume_channels':
        # Process each (time, depth, channel) combination
        time_points, depth, height, width, channels = input_array.shape
        result = np.zeros_like(input_array)
        
        slice_count = 0
        for t in range(time_points):
            for z in range(depth):
                for c in range(channels):
                    slice_count += 1
                    print(f"Processing slice {slice_count}/{total_slices} (t={t+1}/{time_points}, z={z+1}/{depth}, c={c+1}/{channels})")
                    slice_2d = input_array[t, z, :, :, c]
                    denoised_slice = process_2d_slice(model, slice_2d, device, patch_size)
                    result[t, z, :, :, c] = denoised_slice
                    
                    # Memory cleanup
                    if slice_count % 10 == 0 and device.type == "cuda":
                        torch.cuda.empty_cache()
        
        return result
    
    else:
        raise ValueError(f"Unknown processing strategy: {strategy}")

def save_output_robust(array: np.ndarray, output_path: Path):
    """Robust output saving with format detection and error handling"""
    print(f"Saving output to: {output_path}")
    print(f"Output array shape: {array.shape}, dtype: {array.dtype}")
    
    try:
        # Determine output format based on file extension
        output_ext = output_path.suffix.lower()
        
        # Convert to appropriate data type for saving
        if array.dtype == np.float32 or array.dtype == np.float64:
            # Convert float to 16-bit integer (common for microscopy)
            # Assume input was originally in 16-bit range
            output_img = np.clip(array, 0, 65535).round().astype(np.uint16)
            print(f"Converted float array to uint16, range: [{output_img.min()}, {output_img.max()}]")
        else:
            output_img = array
        
        # Save based on format
        if output_ext in ['.tif', '.tiff']:
            with tifffile.TiffWriter(output_path) as out:
                out.write(output_img, resolutionunit=2)
            print("Saved as TIFF")
            
        elif output_ext in ['.mha', '.mhd']:
            try:
                import SimpleITK as sitk
                image_sitk = sitk.GetImageFromArray(output_img)
                sitk.WriteImage(image_sitk, str(output_path))
                print(f"Saved as {output_ext.upper()} using SimpleITK")
            except ImportError:
                print("SimpleITK not available, saving as TIFF instead")
                tiff_path = output_path.with_suffix('.tif')
                with tifffile.TiffWriter(tiff_path) as out:
                    out.write(output_img, resolutionunit=2)
                print(f"Saved as TIFF: {tiff_path}")
                
        elif output_ext == '.npy':
            np.save(output_path, output_img)
            print("Saved as NumPy array")
            
        else:
            # Default to TIFF
            print(f"Unknown extension {output_ext}, saving as TIFF")
            with tifffile.TiffWriter(output_path) as out:
                out.write(output_img, resolutionunit=2)
        
        # Verify file was created and has reasonable size
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"Output file saved successfully, size: {file_size_mb:.1f} MB")
        else:
            raise RuntimeError("Output file was not created")
            
    except Exception as e:
        print(f"Error saving output: {e}")
        traceback.print_exc()
        raise

def inference_handler():
    """Main handler for processing images with comprehensive error handling"""
    try:
        # Show torch cuda info
        _show_torch_cuda_info()

        # Load model
        model, device, patch_size = load_model()
        print(f"Model loaded successfully, patch_size: {patch_size}")

        # Find input files with multiple extensions
        supported_extensions = ['*.tif', '*.tiff', '*.mha', '*.mhd', '*.npy']
        input_files = []
        
        for ext in supported_extensions:
            input_files.extend(sorted(INPUT_PATH.glob(ext)))
        
        print(f"Found input files: {[f.name for f in input_files]}")
        
        if not input_files:
            print(f"No supported files found in {INPUT_PATH}")
            print(f"Supported extensions: {supported_extensions}")
            
            # Enhanced debugging
            if INPUT_PATH.exists():
                print(f"Directory contents:")
                for item in INPUT_PATH.iterdir():
                    if item.is_file():
                        print(f"  File: {item.name} (size: {item.stat().st_size} bytes)")
                    else:
                        print(f"  Directory: {item.name}/")
            else:
                print(f"Input directory does not exist: {INPUT_PATH}")
                # Try to find what directories do exist
                current = INPUT_PATH
                while current != current.parent:
                    if current.exists():
                        print(f"Existing parent directory: {current}")
                        if current.is_dir():
                            print(f"  Contents: {list(current.iterdir())}")
                        break
                    current = current.parent
            
            return 1

        # Process each file
        for i, input_file in enumerate(input_files):
            print(f"\n=== Processing file {i+1}/{len(input_files)}: {input_file.name} ===")
            
            try:
                # Read image
                input_array = read_image_robust(input_file)
                
                # Run inference
                print(f"Starting inference for {input_file.name}...")
                start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                
                if start_time:
                    start_time.record()
                
                result = run_inference_robust(model, input_array, device, patch_size)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_time.elapsed_time(end_time)
                    print(f"Inference completed in {elapsed_ms/1000:.2f} seconds")
                
                # Save output
                output_path = OUTPUT_PATH / input_file.name
                save_output_robust(result, output_path)
                
                print(f"✅ Successfully processed {input_file.name}")
                
                # Memory cleanup
                del result, input_array
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error processing {input_file.name}: {e}")
                traceback.print_exc()
                # Continue with next file instead of stopping
                continue

        print(f"\n=== Processing completed ===")
        return 0
        
    except Exception as e:
        print(f"Fatal error in inference handler: {e}")
        traceback.print_exc()
        return 1

def run():
    """Entry point for Grand Challenge"""
    try:
        # Get interface key
        interface_key = get_interface_key()
        print(f"Interface key: {interface_key}")

        # Run inference
        return inference_handler()
        
    except Exception as e:
        print(f"Fatal error in main run function: {e}")
        traceback.print_exc()
        return 1

def get_interface_key():
    """Get interface key from inputs.json with robust error handling"""
    try:
        json_path = INPUT_PATH.parent.parent.parent / "inputs.json"
        inputs = load_json_file(json_path)
        socket_slugs = [sv["interface"]["slug"] for sv in inputs]
        return tuple(sorted(socket_slugs))
    except Exception as e:
        print(f"Warning: Could not load inputs.json: {e}")
        return ("stacked-neuron-images-with-noise",)

def load_json_file(location):
    """Load JSON file with error handling"""
    try:
        with open(location, "r") as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"Error loading JSON from {location}: {e}")
        raise

def _show_torch_cuda_info():
    """Show comprehensive CUDA information"""
    print("=+=" * 15)
    print("PyTorch and CUDA Information")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
    
    print("=+=" * 15)

if __name__ == "__main__":
    raise SystemExit(run())