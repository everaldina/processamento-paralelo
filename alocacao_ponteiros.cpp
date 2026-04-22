#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include "image_reader/mhd_reader.hpp"

int main() {
    std::string path = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images/TRV1P1CTAI.mhd";
    
    image_reader::MHDReader reader(path);

    if (!reader.readHeader() || !reader.readData()) {
        std::cerr << "Falha ao ler a imagem: " << path << std::endl;
        return 1;
    }
    
    auto dim = reader.getDimensions();
    int width = dim[0];
    int height = dim[1];
    int profundidade_z = dim[2];

    std::cout << "\n=== VERIFICANDO ABORDAGEM ATUAL DA MAIN (getVolumeAs) ===" << std::endl;
    
    // Obtem o volume 3D completo em um unico vector, igual a main.cpp atual
    std::vector<short> volume = reader.getVolumeAs<short>();
    int pixels_per_slice = width * height;
    long tamanho_fatia_bytes = pixels_per_slice * sizeof(short);

    std::cout << "Dimensoes X/Y (Largura/Altura): " << width << "x" << height << "\n";
    std::cout << "Pixels por slice: " << pixels_per_slice << "\n";
    std::cout << "Distancia ESPERADA entre cada Z contiguo: " << tamanho_fatia_bytes << " bytes.\n\n";

    for(int z = 0; z < profundidade_z; z++) {
        const short* ptr_slice_z = volume.data() + (z * pixels_per_slice);
        
        printf("Endreco do inicio da Fatia Z=%d: %p", z, (void*)ptr_slice_z);
        
        if (z > 0) {
            const short* ponteiro_anterior = volume.data() + ((z-1) * pixels_per_slice);
            long distancia = (char*)ptr_slice_z - (char*)ponteiro_anterior;
            printf("  (Distancia em C++: %ld bytes)\n", distancia);
            
            if (distancia != tamanho_fatia_bytes) {
                printf("  [ALERTA] Distancia atípica! Memoria pode nao ser contigua!\n");
            }
        } else {
            printf("\n");
        }

        // omitindo o miolo pra nao poluir
        if (z == 4 && profundidade_z > 8) {
            printf("...\n");
            z = profundidade_z - 4; 
        }
    }

    return 0;
}

