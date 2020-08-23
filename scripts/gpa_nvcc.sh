#!/bin/bash

source gpa_run.sh

cd ..

HPCTK_LM=-lineinfo

NVCCFLAGS="-arch=sm_70                 "    gpa_run  cuda_gmem_8x8x8_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "$*"
NVCCFLAGS="-arch=sm_70                 "    gpa_run  cuda_smem_u_opt-gpu                 nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "$*"
NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    gpa_run  cuda_matsu25d_32x32_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "$*"
