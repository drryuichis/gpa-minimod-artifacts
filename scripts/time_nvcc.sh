#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/Time.md

COMPILER_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
COMPILER_NAME="CUDA $COMPILER_VERSION"

cd ..

echo "## Time Measurements" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory <100> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 100 $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory <100> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 100 $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory <100> [Single-Stream] (Fast Math + Reorder)"                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 100 $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory <600> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory <600> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory <600> [Single-Stream] (Fast Math + Reorder)"                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 $*"
