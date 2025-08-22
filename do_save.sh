#!/usr/bin/env bash
set -e

IMAGE="denoising-calcium"
ARCHIVE="${IMAGE}.tar.gz"

echo "Rebuilding image..."
docker build -t "${IMAGE}" .

echo "Saving Docker image to ${ARCHIVE}…"
docker save "${IMAGE}" | gzip > "${ARCHIVE}"

echo "Done. Archive size:"
ls -lh "${ARCHIVE}"