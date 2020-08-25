#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/Time.md

COMPILER_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
COMPILER_NAME="CUDA $COMPILER_VERSION"

cd ..

echo "## Basic Time Measurements" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_s_opt-gpu             "CUDA Global Memory [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_s_opt-gpu          "CUDA Global Memory [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_s_opt-gpu           "CUDA Global Memory [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_s_opt-gpu              "CUDA Global Memory [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu               "CUDA Global Memory [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_opt-gpu            "CUDA Global Memory [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_opt-gpu             "CUDA Global Memory [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_opt-gpu                "CUDA Global Memory [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_s_opt-gpu         "CUDA Shared Memory [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                   "CUDA Shared Memory [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_opt-gpu          "CUDA Shared Memory [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_opt-gpu           "CUDA Shared Memory [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_opt-gpu              "CUDA Shared Memory [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
