#!/bin/bash
# Build the gdsdv2 PyTorch container and import it to squashfs on CSCS Alps.
# Run on a COMPUTE node (podman cannot build from a login node):
#   srun -A a0112 -p debug -t 00:30:00 bash cscs/build_image.sh
#
# The base (NGC PyTorch 25.06 + git) is identical to the sibling d2 project's
# image, so if $SCRATCH/ce-images/d2+25.06.sqsh already exists you may instead
# point cscs/gdsdv2-pt.toml at that file and skip this build entirely.
set -euo pipefail

REPO="${SCRATCH}/projects/gdsdv2"
TAG="gdsdv2-pt:25.06"
SQSH="${SCRATCH}/ce-images/gdsdv2-pt+25.06.sqsh"

cd "${REPO}/cscs"
echo "[build] host: $(hostname)  date: $(date)"
echo "[build] podman build -t ${TAG} ..."
podman build -t "${TAG}" .

echo "[build] enroot import -> ${SQSH}"
rm -f "${SQSH}"
enroot import -x mount -o "${SQSH}" "podman://${TAG}"

echo "[build] done: $(ls -lh "${SQSH}")"
