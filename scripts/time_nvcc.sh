#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/Time.md

COMPILER_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
COMPILER_NAME="CUDA $COMPILER_VERSION"

cd ..

echo "## Basic Time Measurements" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu             "CUDA Global Memory (3D 8x8x8 Blocking)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 2 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_opt-gpu             "CUDA Global Memory (3D 8x8x8 Blocking)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 2 --warm-up $*"

#NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                 "CUDA Shared Memory on U"                                           nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 2 --warm-up $*"
#NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    time_run    cuda_matsu25d_32x32_opt-gpu         "CUDA Kazuaki Matsumuta (2.5D 32x32)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 2 --warm-up $*"
