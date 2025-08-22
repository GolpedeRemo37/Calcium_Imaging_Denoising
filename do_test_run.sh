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
if [ -d "${SCRIPT_DIR}/test/input/interface_0/images/image-stack-unstructured-noise" ]; then
    echo "Images directory contents:"
    ls -la "${SCRIPT_DIR}/test/input/interface_0/images/image-stack-unstructured-noise/"
fi

echo "Running container..."
docker run --rm \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output \
    $DOCKER_IMAGE_TAG

echo "Test run completed! Check ${SCRIPT_DIR}/test/output/interface_0 for results."