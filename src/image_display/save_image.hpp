#pragma once

#include <string>
#include <vector>

namespace image_display {

        // Configuracao para extrair um slice de um arquivo MHD
        struct SliceConfig {
                std::string filepath;
                int axis; // 0 = X, 1 = Y, 2 = Z
                int sliceIndex;
                std::string title; // Nome do arquivo final (ex: "corte_z10")
        };

        class ImageSaver {
        public:
                ImageSaver(const std::string& outputFolder = "data/");

                // salva uma unica imagem
                bool save(const std::string& filepath, int axis, int sliceIndex, const std::string& title);
                bool save(const SliceConfig& config);

                // salva duas imagens separadas em uma unica chamada
                bool save(const SliceConfig& config1, const SliceConfig& config2);

                // Salva multiplas imagens
                bool save(const std::vector<SliceConfig>& configs);

        private:
                std::string outputFolder;
                bool write_png(const std::string& filename, int width, int height, const std::vector<unsigned char>& data);
        };
}
