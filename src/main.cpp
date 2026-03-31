#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include "image_display/save_image.hpp"
#include "image_reader/mhd_reader.hpp"
#include "metrics/ssim.hpp"

// Função para salvar imagens para debug, isolada. 
// Para debugar basta descomentar a chamada dela em process_image_pair ou na main()
void save_debug_images(const std::string& pathA, const std::string& pathB, int eixo, int centerZ_A, int best_slice_B) {
    std::cout << "Salvando imagens de debug em data/..." << std::endl;
    image_display::ImageSaver saver("data/");
    saver.save(pathA, eixo, centerZ_A, "ImgA_CentroZ");
    saver.save(pathB, eixo, best_slice_B, "ImgB_MaisParecida_com_A");
}

void process_image_pair(const std::string& base_path, const std::string& image_id, std::ofstream& log_file) {
    auto t_start_total = std::chrono::high_resolution_clock::now();

    std::string pathA = base_path + "/" + image_id + "CTAI.mhd";
    std::string pathB = base_path + "/" + image_id + "CTI.mhd";

    std::cout << "\n=======================================" << std::endl;
    std::cout << "Processando ID: " << image_id << std::endl;

    auto t_start_read = std::chrono::high_resolution_clock::now();
    image_reader::MHDReader readerA(pathA);
    image_reader::MHDReader readerB(pathB);

    if (!readerA.readHeader() || !readerA.readData()) {
        std::cerr << "Falha ao ler a imagem A: " << pathA << std::endl;
        return;
    }
    
    if (!readerB.readHeader() || !readerB.readData()) {
        std::cerr << "Falha ao ler a imagem B: " << pathB << std::endl;
        return;
    }
    auto t_end_read = std::chrono::high_resolution_clock::now();
    double time_read = std::chrono::duration<double>(t_end_read - t_start_read).count();

    auto dimA = readerA.getDimensions();
    auto dimB = readerB.getDimensions();
    
    int eixo = 2; // Eixo Z para central
    int centerZ_A = dimA[eixo] / 2;
    int cortes_B = dimB[eixo];

    int wA = 0, hA = 0;
    std::vector<unsigned char> central_sliceA = readerA.getSlice(eixo, centerZ_A, wA, hA);

    if (central_sliceA.empty()) {
        std::cerr << "Erro: Fatia central de A está vazia." << std::endl;
        return;
    }

    auto t_start_search = std::chrono::high_resolution_clock::now();
    
    double max_ssim = -1.0;
    int best_slice_B = -1;
    double total_ssim_time = 0.0;
    int ssim_calc_count = 0;

    std::cout << "Buscando melhor match entre " << cortes_B << " fatias de B..." << std::endl;

    for (int i = 0; i < cortes_B; ++i) {
        int wTemp = 0, hTemp = 0;
        std::vector<unsigned char> sliceB = readerB.getSlice(eixo, i, wTemp, hTemp);
        
        if (!sliceB.empty() && wTemp == wA && hTemp == hA) {
            auto t_start_ssim = std::chrono::high_resolution_clock::now();
            double current_ssim = metrics::calculate_ssim(central_sliceA, sliceB, wA, hA, 7);
            auto t_end_ssim = std::chrono::high_resolution_clock::now();
            
            total_ssim_time += std::chrono::duration<double>(t_end_ssim - t_start_ssim).count();
            ssim_calc_count++;

            if (current_ssim > max_ssim) {
                max_ssim = current_ssim;
                best_slice_B = i;
            }
        }
    }

    auto t_end_search = std::chrono::high_resolution_clock::now();
    double time_search = std::chrono::duration<double>(t_end_search - t_start_search).count();

    auto t_end_total = std::chrono::high_resolution_clock::now();
    double time_total = std::chrono::duration<double>(t_end_total - t_start_total).count();
    
    double time_avg_ssim = (ssim_calc_count > 0) ? (total_ssim_time / ssim_calc_count) : 0.0;

    if (best_slice_B != -1) {
        std::cout << "Melhor corte de B: Z=" << best_slice_B << " | SSIM: " << max_ssim << std::endl;
        
        // --- LOG ---
        log_file << "ID da Imagem: " << image_id << "\n";
        log_file << "Quantidade de cortes B avaliados: " << cortes_B << "\n";
        log_file << "Tempo de Leitura Total (A+B): " << time_read << " s\n";
        log_file << "Tempo Medio Calculo SSIM (por slice): " << time_avg_ssim << " s\n";
        log_file << "Tempo de Busca (SSIM loop): " << time_search << " s\n";
        log_file << "Tempo Total do Processo: " << time_total << " s\n";
        log_file << "Melhor Slice (Z): " << best_slice_B << "\n";
        log_file << "SSIM Maximo Encontrado: " << max_ssim << "\n";
        log_file << "--------------------------------------------------------\n";
        
        // --- DEBUG DE SALVAR IMAGENS ---
        // Descomente a linha abaixo para gerar os .png na pasta data/
        // save_debug_images(pathA, pathB, eixo, centerZ_A, best_slice_B);
    } else {
        std::cerr << "Nenhum corte valido encontrado em B." << std::endl;
    }
}

int main() {
    std::string base_path = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images";
    std::vector<std::string> ids_imagens = {
        "TRV1P1", "TRV1P2", "TRV1P3", "TRV1P4", "TRV1P5", 
        "TRV1P6", "TRV1P7", "TRV1P8", "TRV2P1", "TRV2P2", "TRV2P3", 
        "TRV2P4", "TRV2P5", "TRV2P6", "TRV2P7", "TRV2P8", "TRV3P1", 
        "TRV3P2", "TRV3P3", "TRV3P4", "TRV3P5", "TRV3P6", "TRV3P7", 
        "TRV3P8", "TRV4P1", "TRV4P2", "TRV4P3", "TRV4P4", "TRV4P5", 
        "TRV4P6", "TRV4P7", "TRV4P8",
    };

    std::ofstream log_file("data/log.txt");
    if (!log_file.is_open()) {
        std::cerr << "Falha ao abrir ou criar o data/log.txt!" << std::endl;
        return 1;
    }

    log_file << "=== INICIANDO BATERIA DE VERIFICACAO ===\n";

    for (const auto& id : ids_imagens) {
        process_image_pair(base_path, id, log_file);
    }

    log_file.close();
    std::cout << "\nProcessamento finalizado! Log salvo em data/log.txt." << std::endl;

    return 0;
}
