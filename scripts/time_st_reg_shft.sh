#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/Time.md

COMPILER_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
COMPILER_NAME="CUDA $COMPILER_VERSION"

cd ..

echo "## Time Measurements" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_80 -maxrregcount=64                "        time_run    cuda_st_reg_shft_opt-gpu                 "CUDA 2.5D Register Shifting <100> [Single-Stream] (Original)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 100 $*"
NVCCFLAGS="-arch=sm_80 -maxrregcount=64 -DENABLE_MEMCPY_ASYNC"  time_run    cuda_st_reg_shft_opt-gpu                 "CUDA 2.5D Register Shifting <100> [Single-Stream] (Double buffering using memcpy_async)"      nvcc    "$COMPILER_NAME"    $REPORT     "--grid 100 $*"

NVCCFLAGS="-arch=sm_80 -maxrregcount=64                "        time_run    cuda_st_reg_shft_opt-gpu                 "CUDA 2.5D Register Shifting <600> [Single-Stream] (Original)"                                 nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 $*"
NVCCFLAGS="-arch=sm_80 -maxrregcount=64 -DENABLE_MEMCPY_ASYNC"  time_run    cuda_st_reg_shft_opt-gpu                 "CUDA 2.5D Register Shifting <600> [Single-Stream] (Double buffering using memcpy_async)"      nvcc    "$COMPILER_NAME"    $REPORT     "--grid 600 $*"
