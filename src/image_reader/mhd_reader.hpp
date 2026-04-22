#include <string>
#include <vector>

namespace image_reader {
    class MHDReader {
    public:
        MHDReader(const std::string& filepath);
        ~MHDReader();

        bool readHeader();
        bool readData();

        const std::vector<int>& getDimensions() const;
        const std::vector<float>& getSpacing() const;
        const std::vector<float>& getOrigin() const;
        const std::vector<unsigned char>& getData() const;

        // Retorna a imagem 2D do corte normalizada em 8 bits (0-255).
        std::vector<unsigned char> getSlice(int axis, int sliceIndex, int& outWidth, int& outHeight) const;

        // Retorna a imagem 2D do corte convertida para o tipo T requisitado, sem normalizacao
        template<typename T>
        std::vector<T> getSliceAs(int axis, int sliceIndex, int& outWidth, int& outHeight) const;

        // Retorna todo o volume 3D convertido para o tipo T, como um unico bloco contíguo de memoria
        template<typename T>
        std::vector<T> getVolumeAs() const;

    private:
        std::string filepath;                // caminho do .mhd (absoluto ou relativo)
        std::string dataFilename;            // nome/relativo do arquivo RAW (ElementDataFile)
        std::string elementType;             // tipo de dado (ElementType)
        bool compressedData;                 // Indica se os dados estão comprimidos (zlib)

        std::vector<int> dimensions;         // DimSize
        std::vector<float> spacing;          // ElementSpacing
        std::vector<float> origin;           // Offset
        std::vector<unsigned char> data;     // dados brutos lidos do RAW
    };

    template<typename T>
    std::vector<T> MHDReader::getSliceAs(int axis, int sliceIndex, int& outWidth, int& outHeight) const {
        std::vector<T> result;
        if (dimensions.size() != 3) return result;

        int dimX = dimensions[0], dimY = dimensions[1], dimZ = dimensions[2];
        if (axis < 0 || axis > 2) return result;

        int maxIndex = (axis == 0) ? dimX : (axis == 1) ? dimY : dimZ;
        if (sliceIndex < 0 || sliceIndex >= maxIndex) return result;

        std::size_t numVoxels = static_cast<std::size_t>(dimX) * dimY * dimZ;
        if (data.empty() || data.size() % numVoxels != 0) return result;
        int bytesPerVoxel = static_cast<int>(data.size() / numVoxels);

        auto get_pixel_as_T = [&](std::size_t byte_offset) -> T {
            if (elementType == "MET_SHORT" && bytesPerVoxel == 2) {
                short val = *reinterpret_cast<const short*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_USHORT" && bytesPerVoxel == 2) {
                unsigned short val = *reinterpret_cast<const unsigned short*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_FLOAT" && bytesPerVoxel == 4) {
                float val = *reinterpret_cast<const float*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_DOUBLE" && bytesPerVoxel == 8) {
                double val = *reinterpret_cast<const double*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else {
                return static_cast<T>(data[byte_offset]);
            }
        };

        if (axis == 2) {
            outWidth = dimX; outHeight = dimY;
            result.resize(outWidth * outHeight);
            std::size_t base_offset = static_cast<std::size_t>(sliceIndex) * dimX * dimY * bytesPerVoxel;
            for (int i = 0; i < outWidth * outHeight; ++i) {
                result[i] = get_pixel_as_T(base_offset + i * bytesPerVoxel);
            }
        } else if (axis == 1) {
            outWidth = dimX; outHeight = dimZ;
            result.resize(outWidth * outHeight);
            for (int z = 0; z < dimZ; ++z) {
                for (int x = 0; x < dimX; ++x) {
                    std::size_t vIdx = (static_cast<std::size_t>(z) * dimY * dimX + sliceIndex * dimX + x) * bytesPerVoxel;
                    std::size_t sIdx = static_cast<std::size_t>(z) * dimX + x;
                    result[sIdx] = get_pixel_as_T(vIdx);
                }
            }
        } else {
            outWidth = dimY; outHeight = dimZ;
            result.resize(outWidth * outHeight);
            for (int z = 0; z < dimZ; ++z) {
                for (int y = 0; y < dimY; ++y) {
                    std::size_t vIdx = (static_cast<std::size_t>(z) * dimY * dimX + y * dimX + sliceIndex) * bytesPerVoxel;
                    std::size_t sIdx = static_cast<std::size_t>(z) * dimY + y;
                    result[sIdx] = get_pixel_as_T(vIdx);
                }
            }
        }

        return result;
    }

    template<typename T>
    std::vector<T> MHDReader::getVolumeAs() const {
        std::vector<T> result;
        if (dimensions.empty()) return result;

        std::size_t numVoxels = 1;
        for (int d : dimensions) numVoxels *= static_cast<std::size_t>(d);

        if (data.empty() || data.size() % numVoxels != 0) return result;
        int bytesPerVoxel = static_cast<int>(data.size() / numVoxels);

        result.resize(numVoxels);

        auto get_pixel_as_T = [&](std::size_t byte_offset) -> T {
            if (elementType == "MET_SHORT" && bytesPerVoxel == 2) {
                short val = *reinterpret_cast<const short*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_USHORT" && bytesPerVoxel == 2) {
                unsigned short val = *reinterpret_cast<const unsigned short*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_FLOAT" && bytesPerVoxel == 4) {
                float val = *reinterpret_cast<const float*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else if (elementType == "MET_DOUBLE" && bytesPerVoxel == 8) {
                double val = *reinterpret_cast<const double*>(&data[byte_offset]);
                return static_cast<T>(val);
            } else {
                return static_cast<T>(data[byte_offset]);
            }
        };

        for (std::size_t i = 0; i < numVoxels; ++i) {
            result[i] = get_pixel_as_T(i * bytesPerVoxel);
        }

        return result;
    }
}
