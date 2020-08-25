## Time Measurements

### CUDA , on Intel(R), CUDA Shared Memory <100> [Single-Stream] (Original)
```
==30357== NVPROF is profiling process 30357, command: ./main_cuda_smem_u_s_opt-gpu_nvcc --grid 100
ndamp = 27 27 27
grid = 100 100 100
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -0.205806, 0.140146
Time kernel: 0.117521 s
Time modeling: 0.540201 s
==30357== Profiling application: ./main_cuda_smem_u_s_opt-gpu_nvcc --grid 100
==30357== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.10%  89.254ms      6000  14.875us  6.5280us  30.560us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    6.29%  6.3766ms      1000  6.3760us  5.9200us  11.455us  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                    3.34%  3.3869ms         7  483.84us  1.4400us  649.69us  [CUDA memcpy HtoD]
                    1.75%  1.7690ms      1000  1.7690us  1.6950us  1.9520us  kernel_add_source_kernel(float*, __int64, float)
                    0.48%  487.90us         3  162.63us  1.7280us  484.32us  [CUDA memcpy DtoH]
                    0.03%  32.992us         1  32.992us  32.992us  32.992us  find_min_max_u_kernel(float const *, float*, float*)
      API calls:   64.94%  224.15ms         8  28.018ms  7.1450us  222.99ms  cudaMalloc
                   20.88%  72.076ms      1000  72.076us  42.693us  85.722us  cudaStreamSynchronize
                   12.29%  42.420ms      8001  5.3010us  4.7320us  512.80us  cudaLaunchKernel
                    1.40%  4.8305ms        10  483.05us  13.150us  778.17us  cudaMemcpy
                    0.22%  759.04us         8  94.879us  9.1350us  151.69us  cudaFree
                    0.18%  605.16us         1  605.16us  605.16us  605.16us  cuDeviceTotalMem
                    0.07%  252.96us       101  2.5040us     135ns  124.25us  cuDeviceGetAttribute
                    0.01%  37.928us         1  37.928us  37.928us  37.928us  cuDeviceGetName
                    0.01%  22.505us         1  22.505us  22.505us  22.505us  cudaStreamCreate
                    0.00%  12.864us         1  12.864us  12.864us  12.864us  cudaStreamDestroy
                    0.00%  6.7990us         1  6.7990us  6.7990us  6.7990us  cuDeviceGetPCIBusId
                    0.00%  1.4630us         3     487ns     197ns  1.0110us  cuDeviceGetCount
                    0.00%     736ns         2     368ns     176ns     560ns  cuDeviceGet
                    0.00%     291ns         1     291ns     291ns     291ns  cuDeviceGetUuid
```

### CUDA , on Intel(R), CUDA Shared Memory <100> [Single-Stream] (Fast Math)
```
==30457== NVPROF is profiling process 30457, command: ./main_cuda_smem_u_fastmath_s_opt-gpu_nvcc --grid 100
ndamp = 27 27 27
grid = 100 100 100
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -0.205792, 0.140187
Time kernel: 0.114503 s
Time modeling: 0.535976 s
==30457== Profiling application: ./main_cuda_smem_u_fastmath_s_opt-gpu_nvcc --grid 100
==30457== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.80%  86.686ms      6000  14.447us  6.6560us  29.088us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    6.48%  6.4013ms      1000  6.4010us  5.9520us  7.4240us  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                    3.39%  3.3508ms         7  478.69us  1.4400us  643.58us  [CUDA memcpy HtoD]
                    1.80%  1.7748ms      1000  1.7740us  1.6960us  1.9520us  kernel_add_source_kernel(float*, __int64, float)
                    0.49%  480.22us         3  160.07us  1.7280us  476.64us  [CUDA memcpy DtoH]
                    0.03%  33.119us         1  33.119us  33.119us  33.119us  find_min_max_u_kernel(float const *, float*, float*)
      API calls:   65.25%  222.49ms         8  27.812ms  6.9100us  221.37ms  cudaMalloc
                   21.44%  73.089ms      1000  73.088us  42.380us  83.701us  cudaStreamSynchronize
                   11.40%  38.855ms      8001  4.8560us  4.2930us  508.52us  cudaLaunchKernel
                    1.41%  4.7916ms        10  479.16us  13.396us  765.12us  cudaMemcpy
                    0.23%  774.28us         8  96.784us  8.6010us  150.40us  cudaFree
                    0.19%  655.48us         1  655.48us  655.48us  655.48us  cuDeviceTotalMem
                    0.07%  229.36us       101  2.2700us     133ns  96.166us  cuDeviceGetAttribute
                    0.01%  37.487us         1  37.487us  37.487us  37.487us  cuDeviceGetName
                    0.01%  20.799us         1  20.799us  20.799us  20.799us  cudaStreamCreate
                    0.00%  11.334us         1  11.334us  11.334us  11.334us  cudaStreamDestroy
                    0.00%  6.9030us         1  6.9030us  6.9030us  6.9030us  cuDeviceGetPCIBusId
                    0.00%  1.9660us         3     655ns     220ns  1.5160us  cuDeviceGetCount
                    0.00%     704ns         2     352ns     173ns     531ns  cuDeviceGet
                    0.00%     303ns         1     303ns     303ns     303ns  cuDeviceGetUuid
```

### CUDA , on Intel(R), CUDA Shared Memory <100> [Single-Stream] (Fast Math + Reorder)
```
==30560== NVPROF is profiling process 30560, command: ./main_cuda_smem_u_both_s_opt-gpu_nvcc --grid 100
ndamp = 27 27 27
grid = 100 100 100
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -0.205792, 0.140187
Time kernel: 0.110535 s
Time modeling: 0.533947 s
==30560== Profiling application: ./main_cuda_smem_u_both_s_opt-gpu_nvcc --grid 100
==30560== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.22%  82.072ms      6000  13.678us  6.1120us  27.104us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    6.78%  6.3779ms      1000  6.3770us  5.9200us  7.6160us  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                    3.58%  3.3706ms         7  481.52us  1.4400us  647.23us  [CUDA memcpy HtoD]
                    1.87%  1.7604ms      1000  1.7600us  1.6950us  1.9520us  kernel_add_source_kernel(float*, __int64, float)
                    0.51%  484.48us         3  161.49us  1.7280us  480.89us  [CUDA memcpy DtoH]
                    0.03%  32.672us         1  32.672us  32.672us  32.672us  find_min_max_u_kernel(float const *, float*, float*)
      API calls:   66.28%  224.21ms         8  28.026ms  6.6010us  223.05ms  cudaMalloc
                   19.49%  65.934ms      1000  65.934us  39.317us  70.055us  cudaStreamSynchronize
                   12.30%  41.593ms      8001  5.1980us  4.6050us  514.82us  cudaLaunchKernel
                    1.43%  4.8285ms        10  482.85us  14.722us  777.80us  cudaMemcpy
                    0.22%  751.79us         8  93.973us  9.5450us  146.92us  cudaFree
                    0.18%  622.40us         1  622.40us  622.40us  622.40us  cuDeviceTotalMem
                    0.07%  244.04us       101  2.4160us     137ns  113.12us  cuDeviceGetAttribute
                    0.01%  37.917us         1  37.917us  37.917us  37.917us  cuDeviceGetName
                    0.01%  22.970us         1  22.970us  22.970us  22.970us  cudaStreamCreate
                    0.00%  11.969us         1  11.969us  11.969us  11.969us  cudaStreamDestroy
                    0.00%  6.3390us         1  6.3390us  6.3390us  6.3390us  cuDeviceGetPCIBusId
                    0.00%  1.2260us         3     408ns     220ns     783ns  cuDeviceGetCount
                    0.00%     715ns         2     357ns     149ns     566ns  cuDeviceGet
                    0.00%     294ns         1     294ns     294ns     294ns  cuDeviceGetUuid
```

### CUDA , on Intel(R), CUDA Shared Memory <600> [Single-Stream] (Original)
```
==30762== NVPROF is profiling process 30762, command: ./main_cuda_smem_u_s_opt-gpu_nvcc --grid 600
==30762== ndamp = 27 27 27
grid = 600 600 600
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -71.111679, 37.468662
Time kernel: 10.9237 s
Time modeling: 12.267 s
Profiling application: ./main_cuda_smem_u_s_opt-gpu_nvcc --grid 600
==30762== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.76%  6.57595s      1000  6.5759ms  6.5438ms  6.6100ms  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                   35.76%  4.29410s      6000  715.68us  633.14us  867.80us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    7.24%  868.85ms         6  144.81ms  105.83ms  219.39ms  [CUDA memcpy HtoD]
                    2.19%  262.83ms         3  87.611ms  67.519us  262.70ms  [CUDA memcpy DtoH]
                    0.04%  5.3375ms         1  5.3375ms  5.3375ms  5.3375ms  find_min_max_u_kernel(float const *, float*, float*)
                    0.01%  1.7869ms      1000  1.7860us  1.7270us  2.0800us  kernel_add_source_kernel(float*, __int64, float)
      API calls:   88.44%  10.8760s      1000  10.876ms  10.418ms  20.499ms  cudaStreamSynchronize
                    9.27%  1.13981s        10  113.98ms  8.3110us  263.16ms  cudaMemcpy
                    1.84%  226.79ms         8  28.349ms  10.763us  220.02ms  cudaMalloc
                    0.36%  44.785ms      8001  5.5970us  4.7140us  469.60us  cudaLaunchKernel
                    0.08%  9.4345ms         8  1.1793ms  43.269us  3.2199ms  cudaFree
                    0.00%  595.08us         1  595.08us  595.08us  595.08us  cuDeviceTotalMem
                    0.00%  250.98us       101  2.4840us     134ns  108.25us  cuDeviceGetAttribute
                    0.00%  49.678us         1  49.678us  49.678us  49.678us  cuDeviceGetName
                    0.00%  43.446us         1  43.446us  43.446us  43.446us  cudaStreamCreate
                    0.00%  27.987us         1  27.987us  27.987us  27.987us  cudaStreamDestroy
                    0.00%  6.7270us         1  6.7270us  6.7270us  6.7270us  cuDeviceGetPCIBusId
                    0.00%  2.4770us         3     825ns     232ns  1.9980us  cuDeviceGetCount
                    0.00%     634ns         2     317ns     172ns     462ns  cuDeviceGet
                    0.00%     344ns         1     344ns     344ns     344ns  cuDeviceGetUuid
```

### CUDA , on Intel(R), CUDA Shared Memory <600> [Single-Stream] (Fast Math)
```
==30920== NVPROF is profiling process 30920, command: ./main_cuda_smem_u_fastmath_s_opt-gpu_nvcc --grid 600
==30920== ndamp = 27 27 27
grid = 600 600 600
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -71.111687, 37.468643
Time kernel: 10.8413 s
Time modeling: 12.4796 s
Profiling application: ./main_cuda_smem_u_fastmath_s_opt-gpu_nvcc --grid 600
==30920== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.88%  6.58117s      1000  6.5812ms  6.5468ms  6.6187ms  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                   34.37%  4.19897s      6000  699.83us  560.86us  950.45us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    7.21%  881.10ms         6  146.85ms  105.76ms  219.76ms  [CUDA memcpy HtoD]
                    4.48%  546.99ms         3  182.33ms  67.520us  546.82ms  [CUDA memcpy DtoH]
                    0.04%  5.3599ms         1  5.3599ms  5.3599ms  5.3599ms  find_min_max_u_kernel(float const *, float*, float*)
                    0.01%  1.7894ms      1000  1.7890us  1.7270us  12.544us  kernel_add_source_kernel(float*, __int64, float)
      API calls:   86.26%  10.7933s      1000  10.793ms  10.293ms  25.736ms  cudaStreamSynchronize
                   11.48%  1.43631s        10  143.63ms  8.5590us  547.30ms  cudaMemcpy
                    1.82%  227.82ms         8  28.478ms  11.321us  221.12ms  cudaMalloc
                    0.36%  44.600ms      8001  5.5740us  4.5830us  496.24us  cudaLaunchKernel
                    0.08%  9.6496ms         8  1.2062ms  50.635us  3.3058ms  cudaFree
                    0.01%  645.82us         1  645.82us  645.82us  645.82us  cuDeviceTotalMem
                    0.00%  206.19us       101  2.0410us     132ns  89.343us  cuDeviceGetAttribute
                    0.00%  107.16us         1  107.16us  107.16us  107.16us  cudaStreamDestroy
                    0.00%  45.856us         1  45.856us  45.856us  45.856us  cuDeviceGetName
                    0.00%  43.558us         1  43.558us  43.558us  43.558us  cudaStreamCreate
                    0.00%  6.5990us         1  6.5990us  6.5990us  6.5990us  cuDeviceGetPCIBusId
                    0.00%  2.2130us         3     737ns     243ns  1.7060us  cuDeviceGetCount
                    0.00%     727ns         2     363ns     164ns     563ns  cuDeviceGet
                    0.00%     309ns         1     309ns     309ns     309ns  cuDeviceGetUuid
```

### CUDA , on Intel(R), CUDA Shared Memory <600> [Single-Stream] (Fast Math + Reorder)
```
==31285== NVPROF is profiling process 31285, command: ./main_cuda_smem_u_both_s_opt-gpu_nvcc --grid 600
ndamp = 27 27 27
grid = 600 600 600
time step 100 / 1000
time step 200 / 1000
time step 300 / 1000
time step 400 / 1000
time step 500 / 1000
time step 600 / 1000
time step 700 / 1000
time step 800 / 1000
time step 900 / 1000
time step 1000 / 1000
FINAL min_u,  max_u = -71.111687, 37.468643
Time kernel: 10.7736 s
Time modeling: 12.6615 s
==31285== Profiling application: ./main_cuda_smem_u_both_s_opt-gpu_nvcc --grid 600
==31285== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.04%  6.57956s      1000  6.5796ms  6.5449ms  6.6137ms  target_inner_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float const *, float const *)
                   33.33%  4.13432s      6000  689.05us  537.91us  953.05us  target_pml_3d_kernel(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float const *, float*, float const *, float*, float const *)
                    7.28%  903.58ms         6  150.60ms  105.46ms  230.07ms  [CUDA memcpy HtoD]
                    6.29%  779.74ms         3  259.91ms  67.551us  779.60ms  [CUDA memcpy DtoH]
                    0.04%  5.3591ms         1  5.3591ms  5.3591ms  5.3591ms  find_min_max_u_kernel(float const *, float*, float*)
                    0.01%  1.7961ms      1000  1.7960us  1.7280us  13.247us  kernel_add_source_kernel(float*, __int64, float)
      API calls:   84.41%  10.7238s      1000  10.724ms  8.3853ms  21.163ms  cudaStreamSynchronize
                   13.32%  1.69157s        10  169.16ms  9.2380us  780.08ms  cudaMemcpy
                    1.84%  233.76ms         8  29.220ms  10.562us  226.99ms  cudaMalloc
                    0.35%  44.076ms      8001  5.5080us  4.6150us  417.31us  cudaLaunchKernel
                    0.08%  9.5633ms         8  1.1954ms  34.142us  3.2914ms  cudaFree
                    0.01%  660.34us         1  660.34us  660.34us  660.34us  cuDeviceTotalMem
                    0.00%  223.89us       101  2.2160us     136ns  89.563us  cuDeviceGetAttribute
                    0.00%  45.749us         1  45.749us  45.749us  45.749us  cuDeviceGetName
                    0.00%  43.672us         1  43.672us  43.672us  43.672us  cudaStreamCreate
                    0.00%  29.475us         1  29.475us  29.475us  29.475us  cudaStreamDestroy
                    0.00%  6.0870us         1  6.0870us  6.0870us  6.0870us  cuDeviceGetPCIBusId
                    0.00%  2.5450us         3     848ns     228ns  2.0800us  cuDeviceGetCount
                    0.00%     753ns         2     376ns     194ns     559ns  cuDeviceGet
                    0.00%     450ns         1     450ns     450ns     450ns  cuDeviceGetUuid
```

