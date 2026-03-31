#include "save_image.hpp"
#include "../image_reader/mhd_reader.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using image_reader::MHDReader;

namespace image_display {

ImageSaver::ImageSaver(const std::string& folder) : outputFolder(folder) {
if (!outputFolder.empty() && outputFolder.back() != '/' && outputFolder.back() != '\\') {
outputFolder += "/";
}
}

bool ImageSaver::write_png(const std::string& filename, int width, int height, const std::vector<unsigned char>& data) {
std::string path = outputFolder + filename;
if (path.length() < 4 || path.substr(path.length() - 4) != ".png") {
path += ".png";
}

cv::Mat img(height, width, CV_8UC1, const_cast<unsigned char*>(data.data()));

if (!cv::imwrite(path, img)) {
std::cerr << "Erro ao usar OpenCV para salvar: " << path << std::endl;
return false;
}

std::cout << "Imagem salva com sucesso: " << path << std::endl;
return true;
}

bool ImageSaver::save(const std::string& filepath, int axis, int sliceIndex, const std::string& title) {
MHDReader reader(filepath);
if (!reader.readHeader() || !reader.readData()) {
            std::cerr << "Falha na leitura do arquivo MHD/RAW." << std::endl;
            return false;
        }

        int w = 0, h = 0;
        std::vector<unsigned char> data = reader.getSlice(axis, sliceIndex, w, h);

        if (data.empty()) {
std::cerr << "Erro ao extrair o slice." << std::endl;
return false;
}
return write_png(title, w, h, data);
}

bool ImageSaver::save(const SliceConfig& config) {
return save(config.filepath, config.axis, config.sliceIndex, config.title);
}

bool ImageSaver::save(const SliceConfig& config1, const SliceConfig& config2) {
bool ok1 = save(config1);
bool ok2 = save(config2);
return ok1 && ok2;
}

bool ImageSaver::save(const std::vector<SliceConfig>& configs) {
bool all_ok = true;
for (const auto& cfg : configs) {
if (!save(cfg)) all_ok = false;
}
return all_ok;
}
}
