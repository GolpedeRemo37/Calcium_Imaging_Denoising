# 2D Denoising Submission (FMD)

This container performs 2D image denoising using a pretrained model.

## Usage

Run the container with input and output folders mounted:

```bash
docker run --rm \
  -v $(pwd)/test/input:/workspace/input \
  -v $(pwd)/test/output:/workspace/output \
  denoising-fmd \
  --input_dir ./input \
  --output_dir ./output
