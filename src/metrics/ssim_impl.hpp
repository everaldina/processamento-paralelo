#pragma once

#include "ssim.hpp"
#include <cmath>

namespace metrics {
    template<typename T>
    double calculate_ssim(const T* im1, const T* im2, int width, int height, double data_range, int win_size) {
        if (!im1 || !im2) {
            return 0.0;
        }

        const double K1 = 0.01;
        const double K2 = 0.03;
        const double L = data_range;
        const double C1 = (K1 * L) * (K1 * L);
        const double C2 = (K2 * L) * (K2 * L);

        double mssim = 0.0;
        int count = 0;
        int offset = win_size / 2;

        // itera pixel a pixel, aplicando a janela 
        // ignora as bordas onde a janela não caberia perfeitamente
        for (int y = offset; y < height - offset; ++y) {
            for (int x = offset; x < width - offset; ++x) {
                double sum1 = 0.0, sum2 = 0.0;
                double sum_sq1 = 0.0, sum_sq2 = 0.0;
                double sum_mult = 0.0;

                // janela local
                for (int wy = -offset; wy <= offset; ++wy) {
                    for (int wx = -offset; wx <= offset; ++wx) {
                        int idx = (y + wy) * width + (x + wx);
                        double v1 = static_cast<double>(im1[idx]);
                        double v2 = static_cast<double>(im2[idx]);

                        sum1 += v1;
                        sum2 += v2;
                        sum_sq1 += v1 * v1;
                        sum_sq2 += v2 * v2;
                        sum_mult += v1 * v2;
                    }
                }

                double N = win_size * win_size;
                double mean1 = sum1 / N;
                double mean2 = sum2 / N;

                // variâncias e covariância
                double var1 = (sum_sq1 / N) - (mean1 * mean1);
                double var2 = (sum_sq2 / N) - (mean2 * mean2);
                double cov12 = (sum_mult / N) - (mean1 * mean2);

                double num = (2.0 * mean1 * mean2 + C1) * (2.0 * cov12 + C2);
                double den = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);

                mssim += num / den;
                count++;
            }
        }

        if (count == 0) return 0.0;
        return mssim / static_cast<double>(count);
    }
}
