#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Building Stereo Pipeline (C++) ==="

mkdir -p build
cd build

echo "--- Running CMake ---"
cmake -DCMAKE_BUILD_TYPE=Release ..

echo ""
echo "--- Compiling ---"
NPROC=$(nproc 2>/dev/null || echo 4)
make -j"$NPROC"

echo ""
echo "--- Build complete ---"
ls -la stereo_pipeline
echo ""
echo "Usage:"
echo "  ./build/stereo_pipeline --left <left.png> --right <right.png> --save-all"
