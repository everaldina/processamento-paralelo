#pragma once
#include <vector>

namespace metrics {
    // Calcula o SSIM (Structural Similarity Index Measure) médio
    // Usa uma janela deslizante (box filter) uniforme, inspirada no artigo original/Skimage.
    // im1 e im2: vetores contendo os dados da imagem.
    // width, height: dimensoes do corte.
    // data_range: o alcance dinâmico dos dados (ex: 255.0 para 8-bit, 65535.0 para 16-bit)
    // win_size: tamanho da janela (padrao 7).
    template<typename T>
    double calculate_ssim(const std::vector<T>& im1, const std::vector<T>& im2, int width, int height, double data_range = 255.0, int win_size = 7);
}

// Inclusão da implementação do template
#include "ssim_impl.hpp"
