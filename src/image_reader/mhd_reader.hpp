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
        // Pode pegar o ponteiro do vetor ( via método .data() ) após receber o retorno.
        std::vector<unsigned char> getSlice(int axis, int sliceIndex, int& outWidth, int& outHeight) const;

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
}