#pragma once

// Kernel CUDA (equivalente em GPU do calculate_ssim de ssim_impl.hpp). Fica num .cuh
// separado (incluido de parallel_cuda.cu) so por organizacao - nao precisa de nenhuma
// flag extra no nvcc, o -I./src que ja existe para image_reader/mhd_reader.hpp resolve
// esse #include tambem.

#include <cuda_runtime.h>

// =====================================================================================
// Kernel CUDA: um bloco por corte de B (grid com blockIdx.x = indice do corte).
// As threads do bloco varrem os pixels validos do corte em passadas (grid-stride loop)
// e cada uma calcula a janela deslizante (win_size x win_size) completa para o seu
// pixel - mesma formula de metrics::calculate_ssim, so que executada em paralelo.
// O resultado (SSIM medio do corte) sai de uma reducao em memoria compartilhada.
// Ver README, secao "Como as threads, blocos e grids sao organizados", para uma
// explicacao passo a passo com exemplos numericos de como threadIdx/blockIdx/blockDim
// se combinam aqui.
//
// Nota de precisao: as somas de janela (sum_a, sum_b, sum_sq_a, sum_sq_b, sum_a_times_b)
// sao somas de inteiros (valores em `short`) representados em double e permanecem exatas
// independente da ordem, entao sao bit-a-bit identicas a versao sequencial. Ja o
// acumulo final (soma do SSIM de cada pixel do corte) e feito em arvore por varias
// threads, e ponto flutuante nao e associativo - o valor final pode divergir da CPU
// por ~1e-12, bem abaixo da tolerancia de 1e-3 ja usada nos notebooks deste projeto.

// funcao __global__ do kernel roda na GPU e e chamada a partir da CPU.
__global__ void ssim_kernel(const short* __restrict__ image_a,
                             const short* __restrict__ volume_b,
                             int width, int height, int win_size,
                             double data_range,
                             double* ssim_per_slice) {
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double L = data_range;
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);
    extern __shared__ double partial_sums[];

    int slice_index = blockIdx.x;
    const short* image_b = volume_b + static_cast<std::size_t>(slice_index) * width * height;

    int offset = win_size / 2;
    int valid_width = width - 2 * offset;
    int valid_height = height - 2 * offset;
    long long valid_pixel_count = (valid_width > 0 && valid_height > 0)
                                       ? static_cast<long long>(valid_width) * valid_height
                                       : 0;

    double window_area = static_cast<double>(win_size) * static_cast<double>(win_size);
    double thread_ssim_sum = 0.0;

    for (long long pixel_index = threadIdx.x; pixel_index < valid_pixel_count; pixel_index += blockDim.x) {
        int y = offset + static_cast<int>(pixel_index / valid_width);
        int x = offset + static_cast<int>(pixel_index % valid_width);

        double sum_a = 0.0, sum_b = 0.0, sum_sq_a = 0.0, sum_sq_b = 0.0, sum_a_times_b = 0.0;
        for (int window_dy = -offset; window_dy <= offset; ++window_dy) {
            int row_start = (y + window_dy) * width;
            for (int window_dx = -offset; window_dx <= offset; ++window_dx) {
                int flat_index = row_start + (x + window_dx);
                double pixel_a = static_cast<double>(image_a[flat_index]);
                double pixel_b = static_cast<double>(image_b[flat_index]);
                sum_a += pixel_a;
                sum_b += pixel_b;
                sum_sq_a += pixel_a * pixel_a;
                sum_sq_b += pixel_b * pixel_b;
                sum_a_times_b += pixel_a * pixel_b;
            }
        }

        double mean_a = sum_a / window_area;
        double mean_b = sum_b / window_area;
        double var_a  = sum_sq_a / window_area - mean_a * mean_a;
        double var_b  = sum_sq_b / window_area - mean_b * mean_b;
        double covariance_ab = sum_a_times_b / window_area - mean_a * mean_b;

        double numerator   = (2.0 * mean_a * mean_b + C1) * (2.0 * covariance_ab + C2);
        double denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);

        thread_ssim_sum += numerator / denominator;
    }

    partial_sums[threadIdx.x] = thread_ssim_sum;
    __syncthreads();

    // soma em arvore para reduzir corte medio
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) 
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ssim_per_slice[slice_index] = (valid_pixel_count > 0)
                                           ? (partial_sums[0] / static_cast<double>(valid_pixel_count))
                                           : 0.0;
    }
}
