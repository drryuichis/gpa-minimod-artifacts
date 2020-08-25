#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/Time.md

COMPILER_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
COMPILER_NAME="CUDA $COMPILER_VERSION"

cd ..

echo "## Basic Time Measurements" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_s_opt-gpu             "CUDA Global Memory <400> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_s_opt-gpu          "CUDA Global Memory <400> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_s_opt-gpu           "CUDA Global Memory <400> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_s_opt-gpu              "CUDA Global Memory <400> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu               "CUDA Global Memory <400> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_opt-gpu            "CUDA Global Memory <400> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_opt-gpu             "CUDA Global Memory <400> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_opt-gpu                "CUDA Global Memory <400> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_s_opt-gpu             "CUDA Global Memory <600> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_s_opt-gpu          "CUDA Global Memory <600> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_s_opt-gpu           "CUDA Global Memory <600> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_s_opt-gpu              "CUDA Global Memory <600> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu               "CUDA Global Memory <600> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_opt-gpu            "CUDA Global Memory <600> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_opt-gpu             "CUDA Global Memory <600> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_opt-gpu                "CUDA Global Memory <600> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_s_opt-gpu             "CUDA Global Memory <893> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_s_opt-gpu          "CUDA Global Memory <893> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_s_opt-gpu           "CUDA Global Memory <893> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_s_opt-gpu              "CUDA Global Memory <893> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu               "CUDA Global Memory <893> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_fastmath_opt-gpu            "CUDA Global Memory <893> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_reorder_opt-gpu             "CUDA Global Memory <893> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_gmem_both_opt-gpu                "CUDA Global Memory <893> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory <400> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory <400> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_s_opt-gpu         "CUDA Shared Memory <400> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory <400> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                   "CUDA Shared Memory <400> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_opt-gpu          "CUDA Shared Memory <400> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_opt-gpu           "CUDA Shared Memory <400> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_opt-gpu              "CUDA Shared Memory <400> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 400 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory <600> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory <600> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_s_opt-gpu         "CUDA Shared Memory <600> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory <600> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                   "CUDA Shared Memory <600> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_opt-gpu          "CUDA Shared Memory <600> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_opt-gpu           "CUDA Shared Memory <600> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_opt-gpu              "CUDA Shared Memory <600> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_s_opt-gpu                 "CUDA Shared Memory <893> [Single-Stream] (Original)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_s_opt-gpu        "CUDA Shared Memory <893> [Single-Stream] (Fast Math)"                            nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_s_opt-gpu         "CUDA Shared Memory <893> [Single-Stream] (Reorder)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_s_opt-gpu            "CUDA Shared Memory <893> [Single-Stream] (Both)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                   "CUDA Shared Memory <893> [Multi-Stream] (Original)"                              nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_fastmath_opt-gpu          "CUDA Shared Memory <893> [Multi-Stream] (Fast Math)"                             nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_reorder_opt-gpu           "CUDA Shared Memory <893> [Multi-Stream] (Reorder)"                               nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    time_run    cuda_smem_u_both_opt-gpu              "CUDA Shared Memory <893> [Multi-Stream] (Both)"                                  nvcc    "$COMPILER_NAME"    $REPORT     "--grid 893 --niters 5 --warm-up $*"
