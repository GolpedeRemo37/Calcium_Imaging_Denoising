#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="denoising-calcium"

# Convert Windows paths to Docker-compatible paths for Git Bash
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    INPUT_DIR=$(cygpath -w "${SCRIPT_DIR}/test/input/interface_0" | sed 's/\\/\//g')
    OUTPUT_DIR=$(cygpath -w "${SCRIPT_DIR}/test/output/interface_0" | sed 's/\\/\//g')
else
    INPUT_DIR="${SCRIPT_DIR}/test/input/interface_0"
    OUTPUT_DIR="${SCRIPT_DIR}/test/output/interface_0"
fi

echo "=== Grand Challenge Local Test Script ==="
echo "Script directory: $SCRIPT_DIR"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

echo "Building docker image..."
docker build -t $DOCKER_IMAGE_TAG .

echo "Creating and setting up output directory..."
mkdir -p "${SCRIPT_DIR}/test/output/interface_0"
# Set permissions so Docker can write to it
chmod 777 "${SCRIPT_DIR}/test/output/interface_0"

echo "Checking input files before running container..."
ls -la "${SCRIPT_DIR}/test/input/interface_0/"

# Check for CORRECTED directory name
if [ -d "${SCRIPT_DIR}/test/input/interface_0/images/stacked-neuron-images-with-noise" ]; then
    echo "Images directory contents (CORRECTED PATH):"
    ls -la "${SCRIPT_DIR}/test/input/interface_0/images/stacked-neuron-images-with-noise/"
elif [ -d "${SCRIPT_DIR}/test/input/interface_0/images/image-stack-unstructured-noise" ]; then
    echo "❌ ERROR: Found OLD directory name 'image-stack-unstructured-noise'"
    echo "Please rename it to 'stacked-neuron-images-with-noise' to match Grand Challenge requirements!"
    echo ""
    echo "Run this command to fix:"
    echo "  cd test/input/interface_0/images/"
    echo "  mv image-stack-unstructured-noise stacked-neuron-images-with-noise"
    echo ""
    exit 1
else
    echo "❌ ERROR: No images directory found!"
    echo "Expected: ${SCRIPT_DIR}/test/input/interface_0/images/stacked-neuron-images-with-noise/"
    exit 1
fi

# Clean up previous outputs - CORRECTED OUTPUT PATH
echo "Cleaning up previous test outputs..."
rm -rf "${SCRIPT_DIR}/test/output/interface_0/images/stacked-neuron-images-with-reduced-noise"/*
mkdir -p "${SCRIPT_DIR}/test/output/interface_0/images/stacked-neuron-images-with-reduced-noise/"

echo "Running container..."
docker run --rm \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output \
    --gpus all \
    $DOCKER_IMAGE_TAG

echo ""
echo "=== Test Results ==="

# Check outputs - CORRECTED OUTPUT PATH
OUTPUT_IMAGES_DIR="${SCRIPT_DIR}/test/output/interface_0/images/stacked-neuron-images-with-reduced-noise"
if [ "$(ls -A "$OUTPUT_IMAGES_DIR" 2>/dev/null)" ]; then
    echo "✅ Success! Output files generated:"
    ls -la "$OUTPUT_IMAGES_DIR"
    
    # Show file sizes and types
    echo ""
    echo "File details:"
    for file in "$OUTPUT_IMAGES_DIR"/*.tif "$OUTPUT_IMAGES_DIR"/*.tiff; do
        if [ -f "$file" ]; then
            echo "  $(basename "$file"): $(du -h "$file" | cut -f1)"
        fi
    done
else
    echo "❌ Error: No output files generated in $OUTPUT_IMAGES_DIR"
    echo "Check the container logs above for errors."
    exit 1
fi

echo ""
echo "=== Test completed successfully! ==="
echo "Check ${SCRIPT_DIR}/test/output/interface_0 for results."
echo "Your container is ready for Grand Challenge submission."