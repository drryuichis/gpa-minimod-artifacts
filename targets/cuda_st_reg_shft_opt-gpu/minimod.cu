#include <cuda_runtime.h>

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "../../constants.h"
#include "../../grid.h"

#define N_RADIUS 4
#define N_THREADS_X_DIM 32
#define N_THREADS_Y_DIM 32
#define N_THREADS_Z_DIM 0

// Constant memory coefficients
__constant__ float c_coef0;
__constant__ float c_coefx[N_RADIUS+1];
__constant__ float c_coefy[N_RADIUS+1];
__constant__ float c_coefz[N_RADIUS+1];

#ifdef ENABLE_MEMCPY_ASYNC
#include <cuda_pipeline.h>
#endif

__global__ void kernel_7r_25d_inner(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[2][N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint je = min(j0 + N_THREADS_Y_DIM, y4);
    const llint ke = min(k0 + N_THREADS_X_DIM, z4);

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint sje = (j0+N_THREADS_Y_DIM<y4) ? N_THREADS_Y_DIM : ((y4-y3-1)%N_THREADS_Y_DIM+1);
    const llint ske = (k0+N_THREADS_X_DIM<z4) ? N_THREADS_X_DIM : ((z4-z3-1)%N_THREADS_X_DIM+1);

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float infront1, infront2, infront3, infront4; // variables for input “in front of” the current slice
    float behind1, behind2, behind3, behind4; // variables for input “behind” the current slice
    float current; // input value in the current slice

    behind3  = u[IDX3(x3-4,j,k)];
    behind2  = u[IDX3(x3-3,j,k)];
    behind1  = u[IDX3(x3-2,j,k)];
    current  = u[IDX3(x3-1,j,k)];
    infront1 = u[IDX3(x3+0,j,k)];
    infront2 = u[IDX3(x3+1,j,k)];
    infront3 = u[IDX3(x3+2,j,k)];
    infront4 = u[IDX3(x3+3,j,k)];

    int double_buffer_current = 0, double_buffer_next = 1;

    #ifdef ENABLE_MEMCPY_ASYNC

    if (threadIdx.y < N_RADIUS) {
        __pipeline_memcpy_async(&s_u[double_buffer_current][threadIdx.y][suk], &u[IDX3(x3, j - N_RADIUS, k)], sizeof(float));
        __pipeline_memcpy_async(&s_u[double_buffer_current][threadIdx.y+sje+N_RADIUS][suk], &u[IDX3(x3, threadIdx.y+je, k)], sizeof(float));
    }
    if (threadIdx.x < N_RADIUS) {
        __pipeline_memcpy_async(&s_u[double_buffer_current][suj][threadIdx.x], &u[IDX3(x3,j,k - N_RADIUS)], sizeof(float));
        __pipeline_memcpy_async(&s_u[double_buffer_current][suj][threadIdx.x+ske+N_RADIUS], &u[IDX3(x3,j,threadIdx.x+ke)], sizeof(float));
    }
    __pipeline_memcpy_async(&s_u[double_buffer_current][suj][suk], &u[IDX3(x3,j,k)], sizeof(float));
    __pipeline_commit();


    for (llint i = x3; i < x4; i++) {
        // advance the slice (move the thread-front)
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = u[IDX3(i+N_RADIUS,j,k)];

        __pipeline_wait_prior(0);

        __syncthreads();
        
        if (i+1 < x4) {
            if (threadIdx.y < N_RADIUS) {
                __pipeline_memcpy_async(&s_u[double_buffer_next][threadIdx.y][suk], &u[IDX3(i+1, j - N_RADIUS, k)], sizeof(float));
                __pipeline_memcpy_async(&s_u[double_buffer_next][threadIdx.y+sje+N_RADIUS][suk], &u[IDX3(i+1, threadIdx.y+je, k)], sizeof(float));
            }
            if (threadIdx.x < N_RADIUS) {
                __pipeline_memcpy_async(&s_u[double_buffer_next][suj][threadIdx.x], &u[IDX3(i+1,j,k - N_RADIUS)], sizeof(float));
                __pipeline_memcpy_async(&s_u[double_buffer_next][suj][threadIdx.x+ske+N_RADIUS], &u[IDX3(i+1,j,threadIdx.x+ke)], sizeof(float));
            }
            __pipeline_memcpy_async(&s_u[double_buffer_next][suj][suk], &u[IDX3(i+1,j,k)], sizeof(float));
            __pipeline_commit();
        }

        if (j < y4 && k < z4) {
            float lap = __fmaf_rn(c_coef0, current
                      , __fmaf_rn(c_coefx[1], __fadd_rn(infront1,behind1)
                      , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[double_buffer_current][suj+1][suk],s_u[double_buffer_current][suj-1][suk])
                      , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[double_buffer_current][suj][suk+1],s_u[double_buffer_current][suj][suk-1])
                      , __fmaf_rn(c_coefx[2], __fadd_rn(infront2,behind2)
                      , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[double_buffer_current][suj+2][suk],s_u[double_buffer_current][suj-2][suk])
                      , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[double_buffer_current][suj][suk+2],s_u[double_buffer_current][suj][suk-2])
                      , __fmaf_rn(c_coefx[3], __fadd_rn(infront3,behind3)
                      , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[double_buffer_current][suj+3][suk],s_u[double_buffer_current][suj-3][suk])
                      , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[double_buffer_current][suj][suk+3],s_u[double_buffer_current][suj][suk-3])
                      , __fmaf_rn(c_coefx[4], __fadd_rn(infront4,behind4)
                      , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[double_buffer_current][suj+4][suk],s_u[double_buffer_current][suj-4][suk])
                      , __fmul_rn(c_coefz[4], __fadd_rn(s_u[double_buffer_current][suj][suk+4],s_u[double_buffer_current][suj][suk-4])
            )))))))))))));

            v[IDX3(i,j,k)] = __fmaf_rn(2.f, current,
                __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3(i,j,k)])
            );
        }

        double_buffer_current = 1 - double_buffer_current;
        double_buffer_next = 1 - double_buffer_next;
    }

    #else

    if (threadIdx.y < N_RADIUS) {
      s_u[double_buffer_current][threadIdx.y][suk] = u[IDX3(x3, j - N_RADIUS, k)];
      s_u[double_buffer_current][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(x3, threadIdx.y+je, k)];
    }
    if (threadIdx.x < N_RADIUS) {
      s_u[double_buffer_current][suj][threadIdx.x] = u[IDX3(x3,j,k - N_RADIUS)];
      s_u[double_buffer_current][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(x3,j,threadIdx.x+ke)];
    }
    s_u[double_buffer_current][suj][suk] = u[IDX3(x3,j,k)];

    for (llint i = x3; i < x4; i++) {
        // advance the slice (move the thread-front)
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = u[IDX3(i+N_RADIUS,j,k)];

        __syncthreads();

        if (i+1 < x4) {
          if (threadIdx.y < N_RADIUS) {
              s_u[double_buffer_next][threadIdx.y][suk] = u[IDX3(i+1, j - N_RADIUS, k)];
              s_u[double_buffer_next][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(i+1, threadIdx.y+je, k)];
          }
          if (threadIdx.x < N_RADIUS) {
              s_u[double_buffer_next][suj][threadIdx.x] = u[IDX3(i+1,j,k - N_RADIUS)];
              s_u[double_buffer_next][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(i+1,j,threadIdx.x+ke)];
          }
          s_u[double_buffer_next][suj][suk] = u[IDX3(i+1,j,k)];
        }

        if (j < y4 && k < z4) {
            float lap = __fmaf_rn(c_coef0, current
                      , __fmaf_rn(c_coefx[1], __fadd_rn(infront1,behind1)
                      , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[double_buffer_current][suj+1][suk],s_u[double_buffer_current][suj-1][suk])
                      , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[double_buffer_current][suj][suk+1],s_u[double_buffer_current][suj][suk-1])
                      , __fmaf_rn(c_coefx[2], __fadd_rn(infront2,behind2)
                      , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[double_buffer_current][suj+2][suk],s_u[double_buffer_current][suj-2][suk])
                      , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[double_buffer_current][suj][suk+2],s_u[double_buffer_current][suj][suk-2])
                      , __fmaf_rn(c_coefx[3], __fadd_rn(infront3,behind3)
                      , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[double_buffer_current][suj+3][suk],s_u[double_buffer_current][suj-3][suk])
                      , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[double_buffer_current][suj][suk+3],s_u[double_buffer_current][suj][suk-3])
                      , __fmaf_rn(c_coefx[4], __fadd_rn(infront4,behind4)
                      , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[double_buffer_current][suj+4][suk],s_u[double_buffer_current][suj-4][suk])
                      , __fmul_rn(c_coefz[4], __fadd_rn(s_u[double_buffer_current][suj][suk+4],s_u[double_buffer_current][suj][suk-4])
            )))))))))))));

            v[IDX3(i,j,k)] = __fmaf_rn(2.f, current,
                __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3(i,j,k)])
            );
        }

        double_buffer_current = 1 - double_buffer_current;
        double_buffer_next = 1 - double_buffer_next;
    }

    #endif
}

__global__ void kernel_7r_25d_pml(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[2][N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint je = min(j0 + N_THREADS_Y_DIM, y4);
    const llint ke = min(k0 + N_THREADS_X_DIM, z4);

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint sje = (j0+N_THREADS_Y_DIM<y4) ? N_THREADS_Y_DIM : ((y4-y3-1)%N_THREADS_Y_DIM+1);
    const llint ske = (k0+N_THREADS_X_DIM<z4) ? N_THREADS_X_DIM : ((z4-z3-1)%N_THREADS_X_DIM+1);

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float infront1, infront2, infront3, infront4; // variables for input “in front of” the current slice
    float behind1, behind2, behind3, behind4; // variables for input “behind” the current slice
    float current; // input value in the current slice

    behind3  = u[IDX3(x3-4,j,k)];
    behind2  = u[IDX3(x3-3,j,k)];
    behind1  = u[IDX3(x3-2,j,k)];
    current  = u[IDX3(x3-1,j,k)];
    infront1 = u[IDX3(x3+0,j,k)];
    infront2 = u[IDX3(x3+1,j,k)];
    infront3 = u[IDX3(x3+2,j,k)];
    infront4 = u[IDX3(x3+3,j,k)];

    int double_buffer_current = 0, double_buffer_next = 1;

    #ifdef ENABLE_MEMCPY_ASYNC

    if (threadIdx.y < N_RADIUS) {
        __pipeline_memcpy_async(&s_u[double_buffer_current][threadIdx.y][suk], &u[IDX3(x3, j - N_RADIUS, k)], sizeof(float));
        __pipeline_memcpy_async(&s_u[double_buffer_current][threadIdx.y+sje+N_RADIUS][suk], &u[IDX3(x3, threadIdx.y+je, k)], sizeof(float));
    }
    if (threadIdx.x < N_RADIUS) {
        __pipeline_memcpy_async(&s_u[double_buffer_current][suj][threadIdx.x], &u[IDX3(x3,j,k - N_RADIUS)], sizeof(float));
        __pipeline_memcpy_async(&s_u[double_buffer_current][suj][threadIdx.x+ske+N_RADIUS], &u[IDX3(x3,j,threadIdx.x+ke)], sizeof(float));
    }
    __pipeline_memcpy_async(&s_u[double_buffer_current][suj][suk], &u[IDX3(x3,j,k)], sizeof(float));
    __pipeline_commit();

    for (llint i = x3; i < x4; i++) {
        // advance the slice (move the thread-front)
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = u[IDX3(i+N_RADIUS,j,k)];

        __pipeline_wait_prior(0);

        __syncthreads();

        if (i+1 < x4) {
            if (threadIdx.y < N_RADIUS) {
                __pipeline_memcpy_async(&s_u[double_buffer_next][threadIdx.y][suk], &u[IDX3(i+1, j - N_RADIUS, k)], sizeof(float));
                __pipeline_memcpy_async(&s_u[double_buffer_next][threadIdx.y+sje+N_RADIUS][suk], &u[IDX3(i+1, threadIdx.y+je, k)], sizeof(float));
            }
            if (threadIdx.x < N_RADIUS) {
                __pipeline_memcpy_async(&s_u[double_buffer_next][suj][threadIdx.x], &u[IDX3(i+1,j,k - N_RADIUS)], sizeof(float));
                __pipeline_memcpy_async(&s_u[double_buffer_next][suj][threadIdx.x+ske+N_RADIUS], &u[IDX3(i+1,j,threadIdx.x+ke)], sizeof(float));
            }
            __pipeline_memcpy_async(&s_u[double_buffer_next][suj][suk], &u[IDX3(i+1,j,k)], sizeof(float));
            __pipeline_commit();
        }

        if (j < y4 && k < z4) {
            float lap = __fmaf_rn(c_coef0, current
                      , __fmaf_rn(c_coefx[1], __fadd_rn(infront1,behind1)
                      , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[double_buffer_current][suj+1][suk],s_u[double_buffer_current][suj-1][suk])
                      , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[double_buffer_current][suj][suk+1],s_u[double_buffer_current][suj][suk-1])
                      , __fmaf_rn(c_coefx[2], __fadd_rn(infront2,behind2)
                      , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[double_buffer_current][suj+2][suk],s_u[double_buffer_current][suj-2][suk])
                      , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[double_buffer_current][suj][suk+2],s_u[double_buffer_current][suj][suk-2])
                      , __fmaf_rn(c_coefx[3], __fadd_rn(infront3,behind3)
                      , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[double_buffer_current][suj+3][suk],s_u[double_buffer_current][suj-3][suk])
                      , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[double_buffer_current][suj][suk+3],s_u[double_buffer_current][suj][suk-3])
                      , __fmaf_rn(c_coefx[4], __fadd_rn(infront4,behind4)
                      , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[double_buffer_current][suj+4][suk],s_u[double_buffer_current][suj-4][suk])
                      , __fmul_rn(c_coefz[4], __fadd_rn(s_u[double_buffer_current][suj][suk+4],s_u[double_buffer_current][suj][suk-4])
            )))))))))))));

            const float s_eta_c = eta[IDX3(i,j,k)];

            v[IDX3(i,j,k)] = __fdiv_rn(
                __fmaf_rn(
                    __fmaf_rn(2.f, s_eta_c,
                        __fsub_rn(2.f,
                            __fmul_rn(s_eta_c, s_eta_c)
                        )
                    ),
                    current,
                    __fmaf_rn(
                        vp[IDX3(i,j,k)],
                        __fadd_rn(lap, phi[IDX3(i,j,k)]),
                        -v[IDX3(i,j,k)]
                    )
                ),
                __fmaf_rn(2.f, s_eta_c, 1.f)
            );

            phi[IDX3(i,j,k)] = __fdiv_rn(
                    __fsub_rn(
                        phi[IDX3(i,j,k)],
                        __fmaf_rn(
                        __fmul_rn(
                            __fsub_rn(eta[IDX3(i+1,j,k)], eta[IDX3(i-1,j,k)]),
                            __fsub_rn(infront1,behind1)
                        ), hdx_2,
                        __fmaf_rn(
                        __fmul_rn(
                            __fsub_rn(eta[IDX3(i,j+1,k)], eta[IDX3(i,j-1,k)]),
                            __fsub_rn(s_u[double_buffer_current][suj+1][suk], s_u[double_buffer_current][suj-1][suk])
                        ), hdy_2,
                        __fmul_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i,j,k+1)], eta[IDX3(i,j,k-1)]),
                                __fsub_rn(s_u[double_buffer_current][suj][suk+1], s_u[double_buffer_current][suj][suk-1])
                            ),
                        hdz_2)
                        ))
                    )
                ,
                __fadd_rn(1.f, s_eta_c)
            );
        }

        double_buffer_current = 1 - double_buffer_current;
        double_buffer_next = 1 - double_buffer_next;
    }

    #else

    if (threadIdx.y < N_RADIUS) {
      s_u[double_buffer_current][threadIdx.y][suk] = u[IDX3(x3, j - N_RADIUS, k)];
      s_u[double_buffer_current][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(x3, threadIdx.y+je, k)];
    }
    if (threadIdx.x < N_RADIUS) {
      s_u[double_buffer_current][suj][threadIdx.x] = u[IDX3(x3,j,k - N_RADIUS)];
      s_u[double_buffer_current][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(x3,j,threadIdx.x+ke)];
    }

    s_u[double_buffer_current][suj][suk] = u[IDX3(x3,j,k)];

    for (llint i = x3; i < x4; i++) {
        // advance the slice (move the thread-front)
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = u[IDX3(i+N_RADIUS,j,k)];

        __syncthreads();

        if (i+1 < x4) {
          if (threadIdx.y < N_RADIUS) {
              s_u[double_buffer_next][threadIdx.y][suk] = u[IDX3(i+1, j - N_RADIUS, k)];
              s_u[double_buffer_next][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(i+1, threadIdx.y+je, k)];
          }
          if (threadIdx.x < N_RADIUS) {
              s_u[double_buffer_next][suj][threadIdx.x] = u[IDX3(i+1,j,k - N_RADIUS)];
              s_u[double_buffer_next][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(i+1,j,threadIdx.x+ke)];
          }

          s_u[double_buffer_next][suj][suk] = u[IDX3(i+1,j,k)];
        }

        if (j < y4 && k < z4) {
            float lap = __fmaf_rn(c_coef0, current
                      , __fmaf_rn(c_coefx[1], __fadd_rn(infront1,behind1)
                      , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[double_buffer_current][suj+1][suk],s_u[double_buffer_current][suj-1][suk])
                      , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[double_buffer_current][suj][suk+1],s_u[double_buffer_current][suj][suk-1])
                      , __fmaf_rn(c_coefx[2], __fadd_rn(infront2,behind2)
                      , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[double_buffer_current][suj+2][suk],s_u[double_buffer_current][suj-2][suk])
                      , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[double_buffer_current][suj][suk+2],s_u[double_buffer_current][suj][suk-2])
                      , __fmaf_rn(c_coefx[3], __fadd_rn(infront3,behind3)
                      , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[double_buffer_current][suj+3][suk],s_u[double_buffer_current][suj-3][suk])
                      , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[double_buffer_current][suj][suk+3],s_u[double_buffer_current][suj][suk-3])
                      , __fmaf_rn(c_coefx[4], __fadd_rn(infront4,behind4)
                      , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[double_buffer_current][suj+4][suk],s_u[double_buffer_current][suj-4][suk])
                      , __fmul_rn(c_coefz[4], __fadd_rn(s_u[double_buffer_current][suj][suk+4],s_u[double_buffer_current][suj][suk-4])
            )))))))))))));

            const float s_eta_c = eta[IDX3(i,j,k)];

            v[IDX3(i,j,k)] = __fdiv_rn(
                __fmaf_rn(
                    __fmaf_rn(2.f, s_eta_c,
                        __fsub_rn(2.f,
                            __fmul_rn(s_eta_c, s_eta_c)
                        )
                    ),
                    current,
                    __fmaf_rn(
                        vp[IDX3(i,j,k)],
                        __fadd_rn(lap, phi[IDX3(i,j,k)]),
                        -v[IDX3(i,j,k)]
                    )
                ),
                __fmaf_rn(2.f, s_eta_c, 1.f)
            );

            phi[IDX3(i,j,k)] = __fdiv_rn(
                    __fsub_rn(
                        phi[IDX3(i,j,k)],
                        __fmaf_rn(
                        __fmul_rn(
                            __fsub_rn(eta[IDX3(i+1,j,k)], eta[IDX3(i-1,j,k)]),
                            __fsub_rn(infront1,behind1)
                        ), hdx_2,
                        __fmaf_rn(
                        __fmul_rn(
                            __fsub_rn(eta[IDX3(i,j+1,k)], eta[IDX3(i,j-1,k)]),
                            __fsub_rn(s_u[double_buffer_current][suj+1][suk], s_u[double_buffer_current][suj-1][suk])
                        ), hdy_2,
                        __fmul_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i,j,k+1)], eta[IDX3(i,j,k-1)]),
                                __fsub_rn(s_u[double_buffer_current][suj][suk+1], s_u[double_buffer_current][suj][suk-1])
                            ),
                        hdz_2)
                        ))
                    )
                ,
                __fadd_rn(1.f, s_eta_c)
            );
        }

        double_buffer_current = 1 - double_buffer_current;
        double_buffer_next = 1 - double_buffer_next;
    }

    #endif
}

__global__ void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}

extern "C" void target(
    uint nsteps, double *time_kernel,
    const grid_t grid,
    llint sx, llint sy, llint sz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ coefx, const float *__restrict__ coefy, const float *__restrict__ coefz,
    float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta, const float *__restrict__ source
) {
    struct timespec start, end;

    float *d_u = allocateDeviceGrid(grid);
    float *d_v = allocateDeviceGrid(grid);
    float *d_phi = allocateDeviceGrid(grid);
    float *d_eta = allocateDeviceGrid(grid);
    float *d_vp = allocateDeviceGrid(grid);

    cudaMemset (d_u, 0, gridSize(grid));
    cudaMemset (d_v, 0, gridSize(grid));
    cudaMemcpy(d_vp, vp, gridSize(grid), cudaMemcpyDefault);
    cudaMemcpy(d_phi, phi, gridSize(grid), cudaMemcpyDefault);
    cudaMemcpy(d_eta, eta, gridSize(grid), cudaMemcpyDefault);

    float coef0 = coefx[0] + coefy[0] + coefz[0];
    cudaMemcpyToSymbol (c_coef0, &coef0, sizeof (float));
    cudaMemcpyToSymbol (c_coefx, coefx, (N_RADIUS + 1) * sizeof (float));
    cudaMemcpyToSymbol (c_coefy, coefy, (N_RADIUS + 1) * sizeof (float));
    cudaMemcpyToSymbol (c_coefz, coefz, (N_RADIUS + 1) * sizeof (float));

    const llint xmin = 0; const llint xmax = grid.nx;
    const llint ymin = 0; const llint ymax = grid.ny;

    dim3 threadsPerBlock(N_THREADS_X_DIM, N_THREADS_Y_DIM, 1);

    int num_streams = 1;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking);
    }

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        dim3 n_block_front(
            (grid.z2-grid.z1+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.ny+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_front, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin, xmax, ymin, ymax, grid.z1, grid.z2,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_top(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y2-grid.y1+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_top, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,grid.y1,grid.y2,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_left(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_left, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x1,grid.x2,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_center(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_inner<<<n_block_center, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x3,grid.x4,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_right(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_right, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x5,grid.x6,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_bottom(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y6-grid.y5+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_bottom, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,grid.y5,grid.y6,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_back(
            (grid.z6-grid.z5+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.ny+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        kernel_7r_25d_pml<<<n_block_back, threadsPerBlock, 0, streams[0]>>>(
            grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,ymin,ymax,grid.z5,grid.z6,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        kernel_add_source_kernel<<<1, 1>>>(d_v, IDX3_grid(sx,sy,sz,grid), source[istep-1]);
        clock_gettime(CLOCK_REALTIME, &end);
        *time_kernel += (end.tv_sec  - start.tv_sec) +
                        (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

        float *t = d_u;
        d_u = d_v;
        d_v = t;

        // Print out
        if (istep % npo == 0) {
            printf("time step %u / %u\n", istep, nsteps);
        }
    }


    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }


    cudaMemcpy(u, d_u, gridSize(grid), cudaMemcpyDeviceToHost);

    freeDeviceGrid(d_u, grid);
    freeDeviceGrid(d_v, grid);
    freeDeviceGrid(d_vp, grid);
    freeDeviceGrid(d_phi, grid);
    freeDeviceGrid(d_eta, grid);
}
