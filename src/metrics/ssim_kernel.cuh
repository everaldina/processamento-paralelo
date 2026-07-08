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
// Nota de precisao: a janela deslizante (sum_a, sum_b, sum_sq_a, sum_sq_b, sum_a_times_b,
// media, variancia, covariancia, numerador/denominador do SSIM) roda em `float`, nao
// `double`. GPUs GeForce de consumidor (como a GTX 1060) rodam `double` a 1/32 da
// velocidade de `float` de proposito (segmentacao de mercado da NVIDIA para empurrar
// quem precisa de FP64 de verdade para as linhas profissionais Tesla/Quadro) - como essa
// janela e executada win_size*win_size vezes por pixel, e a parte mais cara do kernel,
// entao e onde compensa trocar para float.
//
// Isso significa que o resultado NAO e mais bit-a-bit identico a versao sequencial (que
// usa double): a soma dos pixels de uma janela cabe exata em float (valores de `short`,
// no maximo 65535, somados no maximo 81 vezes para janela 9x9 - bem dentro dos ~7 digitos
// decimais de precisao do float), mas a variancia (sum_sq/N - media^2, uma subtracao de
// dois numeros grandes e proximos) pode perder precisao de forma mais visivel que em
// double. Isso e esperado, nao e um bug. Depois de mudar isso, vale conferir que o
// "Melhor Slice (Z)" e o "SSIM Maximo Encontrado" continuam batendo com as outras
// versoes dentro da tolerancia de 1e-3 ja usada nos notebooks deste projeto - se um teste
// especifico passar a divergir mais que isso, os proximos candidatos a voltar para double
// seriam a variancia/covariancia (var_a, var_b, covariance_ab).
//
// O acumulo final (soma do SSIM de cada pixel do corte, feito uma vez por pixel, nao
// win_size*win_size vezes) continua em double - e barato de qualquer forma, e ajuda a
// manter a soma de ate ~250 mil termos por bloco mais estavel.

// funcao __global__ do kernel roda na GPU e e chamada a partir da CPU.
__global__ void ssim_kernel(const short* __restrict__ image_a,
                             const short* __restrict__ volume_b,
                             int width, int height, int win_size,
                             double data_range,
                             double* ssim_per_slice) {
    const float L = static_cast<float>(data_range);
    const float C1 = (0.01f * L) * (0.01f * L);
    const float C2 = (0.03f * L) * (0.03f * L);
    extern __shared__ double partial_sums[];

    int slice_index = blockIdx.x;
    const short* image_b = volume_b + static_cast<std::size_t>(slice_index) * width * height;

    int offset = win_size / 2;
    int valid_width = width - 2 * offset;
    int valid_height = height - 2 * offset;
    long long valid_pixel_count = (valid_width > 0 && valid_height > 0)
                                       ? static_cast<long long>(valid_width) * valid_height
                                       : 0;

    float window_area = static_cast<float>(win_size) * static_cast<float>(win_size);
    double thread_ssim_sum = 0.0;

    for (long long pixel_index = threadIdx.x; pixel_index < valid_pixel_count; pixel_index += blockDim.x) {
        int y = offset + static_cast<int>(pixel_index / valid_width);
        int x = offset + static_cast<int>(pixel_index % valid_width);

        float sum_a = 0.0f, sum_b = 0.0f, sum_sq_a = 0.0f, sum_sq_b = 0.0f, sum_a_times_b = 0.0f;
        for (int window_dy = -offset; window_dy <= offset; ++window_dy) {
            int row_start = (y + window_dy) * width;
            for (int window_dx = -offset; window_dx <= offset; ++window_dx) {
                int flat_index = row_start + (x + window_dx);
                float pixel_a = static_cast<float>(image_a[flat_index]);
                float pixel_b = static_cast<float>(image_b[flat_index]);
                sum_a += pixel_a;
                sum_b += pixel_b;
                sum_sq_a += pixel_a * pixel_a;
                sum_sq_b += pixel_b * pixel_b;
                sum_a_times_b += pixel_a * pixel_b;
            }
        }

        float mean_a = sum_a / window_area;
        float mean_b = sum_b / window_area;
        float var_a  = sum_sq_a / window_area - mean_a * mean_a;
        float var_b  = sum_sq_b / window_area - mean_b * mean_b;
        float covariance_ab = sum_a_times_b / window_area - mean_a * mean_b;

        float numerator   = (2.0f * mean_a * mean_b + C1) * (2.0f * covariance_ab + C2);
        float denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);

        thread_ssim_sum += static_cast<double>(numerator / denominator);
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
