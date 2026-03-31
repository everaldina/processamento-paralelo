#pragma once
#include <vector>

namespace metrics {
    // Calcula o SSIM (Structural Similarity Index Measure) médio
    // Usa uma janela deslizante (box filter) uniforme, inspirada no artigo original/Skimage.
    // im1 e im2: vetores de 8-bits contendo os dados da imagem.
    // width, height: dimensoes do corte.
    // win_size: tamanho da janela (padrao 7).
    double calculate_ssim(const std::vector<unsigned char>& im1, const std::vector<unsigned char>& im2, int width, int height, int win_size = 7);
}
