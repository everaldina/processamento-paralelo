#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cctype>
#include <iostream>
#include <zlib.h>

#include "mhd_reader.hpp"
#include <algorithm>

namespace image_reader {

    // Função interna para normalizar os dados para 8 bits
    static void normalize_to_uchar(const unsigned char* src, std::size_t count, int bytesPerVoxel, std::vector<unsigned char>& out) {
        out.resize(count);
        if (bytesPerVoxel == 1) {
            std::copy(src, src + count, out.begin());
            return;
        }
        if (bytesPerVoxel == 2) {
            const short* ptr = reinterpret_cast<const short*>(src);
            short minVal = ptr[0], maxVal = ptr[0];
            for (std::size_t i = 1; i < count; ++i) {
                if (ptr[i] < minVal) minVal = ptr[i];
                if (ptr[i] > maxVal) maxVal = ptr[i];
            }
            double range = static_cast<double>(maxVal - minVal);
            if (range == 0.0) range = 1.0;
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = static_cast<unsigned char>((static_cast<double>(ptr[i] - minVal) / range) * 255.0);
            }
            return;
        }
        
        std::vector<unsigned char> tmp(count);
        for (std::size_t i = 0; i < count; ++i) tmp[i] = src[i * bytesPerVoxel];
        unsigned char minVal = tmp[0], maxVal = tmp[0];
        for (std::size_t i = 1; i < count; ++i) {
            if (tmp[i] < minVal) minVal = tmp[i];
            if (tmp[i] > maxVal) maxVal = tmp[i];
        }
        int range = static_cast<int>(maxVal) - static_cast<int>(minVal);
        if (range == 0) range = 1;
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<unsigned char>(((static_cast<int>(tmp[i]) - minVal) * 255) / range);
        }
    }

    MHDReader::MHDReader(const std::string& filepath) {
        this->filepath = filepath;
        this->compressedData = false;
    }

    MHDReader::~MHDReader() {}

    bool MHDReader::readHeader() {
        std::ifstream in(filepath);
        if (!in.is_open()) {
            return false;
        }

        dimensions.clear();
        spacing.clear();
        origin.clear();
        data.clear();
        dataFilename.clear();
        elementType.clear();

        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            std::size_t pos = line.find('=');
            if (pos == std::string::npos) {
                continue;
            }

            std::string key = line.substr(0, pos);
            // tira espaços à direita do nome da chave
            while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) {
                key.pop_back();
            }

            std::string value = line.substr(pos + 1);

            if (key == "DimSize") {
                std::istringstream iss(value);
                int v;
                while (iss >> v) {
                    dimensions.push_back(v);
                }
            } else if (key == "ElementSpacing") {
                std::istringstream iss(value);
                float v;
                while (iss >> v) {
                    spacing.push_back(v);
                }
            } else if (key == "Offset") { // origem
                std::istringstream iss(value);
                float v;
                while (iss >> v) {
                    origin.push_back(v);
                }
            } else if (key == "ElementDataFile") {
                std::istringstream iss(value);
                iss >> dataFilename;
            } else if (key == "ElementType") {
                std::istringstream iss(value);
                iss >> elementType;
            } else if (key == "CompressedData") {
                std::istringstream iss(value);
                std::string bVal;
                iss >> bVal;
                if (bVal == "True" || bVal == "true") {
                    compressedData = true;
                }
            }
        }

        return !dimensions.empty() && !dataFilename.empty();
    }

    bool MHDReader::readData() {
        if (dimensions.empty() || dataFilename.empty()) {
            return false;
        }

        // diretório do arquivo .mhd
        std::string dir;
        std::size_t pos = filepath.find_last_of("/\\");
        if (pos != std::string::npos) {
            dir = filepath.substr(0, pos + 1);
        }

        std::string dataPath = dir + dataFilename;

        std::ifstream in(dataPath, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        // número de voxels
        std::size_t numVoxels = 1;
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            numVoxels *= static_cast<std::size_t>(dimensions[i]);
        }

        // tamanho em bytes de cada voxel conforme ElementType
        std::size_t bytesPerVoxel = 1;
        if (elementType == "MET_SHORT" || elementType == "MET_USHORT") {
            bytesPerVoxel = 2;
        } else if (elementType == "MET_FLOAT") {
            bytesPerVoxel = 4;
        } else if (elementType == "MET_DOUBLE") {
            bytesPerVoxel = 8;
        } else {
            // MET_UCHAR, MET_CHAR, ou não definido: 1 byte
            bytesPerVoxel = 1;
        }

        std::size_t totalBytes = numVoxels * bytesPerVoxel;
        data.resize(totalBytes);

        in.seekg(0, std::ios::end);
        std::streamsize actualFileSize = in.tellg();
        in.seekg(0, std::ios::beg);

        // std::cout << "[DEBUG] Dimensões calculadas: " << dimensions[0] << "x" << dimensions[1] << "x" << dimensions[2] << std::endl;
        // std::cout << "[DEBUG] Element Type: " << elementType << " (" << bytesPerVoxel << " bytes/voxel)" << std::endl;
        // std::cout << "[DEBUG] Tamanho experado (totalBytes): " << totalBytes << " bytes" << std::endl;
        // std::cout << "[DEBUG] Tamanho real em disco: " << actualFileSize << " bytes" << std::endl;

        bool isZraw = (dataPath.size() > 5 && dataPath.substr(dataPath.size() - 5) == ".zraw");
        if (compressedData || isZraw) {
            std::vector<unsigned char> compressedBuffer(actualFileSize);
            in.read(reinterpret_cast<char*>(compressedBuffer.data()), actualFileSize);
            
            uLongf destLen = totalBytes;
            int ret = uncompress(reinterpret_cast<Bytef*>(data.data()), &destLen, 
                                 reinterpret_cast<const Bytef*>(compressedBuffer.data()), actualFileSize);
            
            if (ret != Z_OK || destLen != totalBytes) {
                std::cerr << "Erro ao descompactar os dados (zlib) do arquivo: " << dataPath << " (code: " << ret << ")" << std::endl;
                return false;
            }
            // std::cout << "[DEBUG] Descompressao com zlib concluida com sucesso!" << std::endl;
        } else {
            in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(totalBytes));
            if (!in) {
                // leitura incompleta
                std::cerr << "Erro ao ler os dados do arquivo RAW: " << dataPath << std::endl;
                return false;
            }
        }

        return true;
    }

    const std::vector<int>& MHDReader::getDimensions() const {
        return dimensions;
    }

    const std::vector<float>& MHDReader::getSpacing() const {
        return spacing;
    }

    const std::vector<float>& MHDReader::getOrigin() const {
        return origin;
    }

    const std::vector<unsigned char>& MHDReader::getData() const {
        return data;
    }

    std::vector<unsigned char> MHDReader::getSlice(int axis, int sliceIndex, int& outWidth, int& outHeight) const {
        std::vector<unsigned char> result;
        if (dimensions.size() != 3) return result;

        int dimX = dimensions[0], dimY = dimensions[1], dimZ = dimensions[2];
        if (axis < 0 || axis > 2) return result;

        int maxIndex = (axis == 0) ? dimX : (axis == 1) ? dimY : dimZ;
        if (sliceIndex < 0 || sliceIndex >= maxIndex) return result;

        std::size_t numVoxels = static_cast<std::size_t>(dimX) * dimY * dimZ;
        if (data.empty() || data.size() % numVoxels != 0) return result;
        int bytesPerVoxel = static_cast<int>(data.size() / numVoxels);

        std::vector<unsigned char> sliceBuffer;
        
        if (axis == 2) { // Z
            outWidth = dimX; outHeight = dimY;
            sliceBuffer.resize(outWidth * outHeight * bytesPerVoxel);
            std::size_t offset = static_cast<std::size_t>(sliceIndex) * dimX * dimY * bytesPerVoxel;
            std::copy(data.begin() + offset, data.begin() + offset + sliceBuffer.size(), sliceBuffer.begin());
        } else if (axis == 1) { // Y
            outWidth = dimX; outHeight = dimZ;
            sliceBuffer.resize(outWidth * outHeight * bytesPerVoxel);
            for (int z = 0; z < dimZ; ++z) {
                for (int x = 0; x < dimX; ++x) {
                    std::size_t vIdx = (static_cast<std::size_t>(z) * dimY * dimX + sliceIndex * dimX + x) * bytesPerVoxel;
                    std::size_t sIdx = (static_cast<std::size_t>(z) * dimX + x) * bytesPerVoxel;
                    std::copy(data.begin() + vIdx, data.begin() + vIdx + bytesPerVoxel, sliceBuffer.begin() + sIdx);
                }
            }
        } else { // X
            outWidth = dimY; outHeight = dimZ;
            sliceBuffer.resize(outWidth * outHeight * bytesPerVoxel);
            for (int z = 0; z < dimZ; ++z) {
                for (int y = 0; y < dimY; ++y) {
                    std::size_t vIdx = (static_cast<std::size_t>(z) * dimY * dimX + y * dimX + sliceIndex) * bytesPerVoxel;
                    std::size_t sIdx = (static_cast<std::size_t>(z) * dimY + y) * bytesPerVoxel;
                    std::copy(data.begin() + vIdx, data.begin() + vIdx + bytesPerVoxel, sliceBuffer.begin() + sIdx);
                }
            }
        }

        normalize_to_uchar(sliceBuffer.data(), outWidth * outHeight, bytesPerVoxel, result);
        return result;
    }

}