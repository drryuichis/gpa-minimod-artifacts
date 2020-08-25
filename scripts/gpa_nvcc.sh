#!/bin/bash

source gpa_run.sh

cd ..

HPCTK_LM=-lineinfo

NVCCFLAGS="-arch=sm_70                 "    gpa_run  cuda_smem_u_s_opt-gpu                 nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 600 $*"
NVCCFLAGS="-arch=sm_70 --use_fast_math "    gpa_run  cuda_smem_u_fastmath_s_opt-gpu        nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 600 $*"
