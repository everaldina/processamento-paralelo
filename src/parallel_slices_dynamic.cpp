#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include "image_display/save_image.hpp"
#include "image_reader/mhd_reader.hpp"
#include "metrics/ssim.hpp"
#include <iomanip>
#include <limits> 
#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif

#define OUTPUT_FOLDER "output/"

// descomentar a chamada em main para salvar imagens
void save_debug_images(int id, const std::string& pathA, const std::string& pathB, int eixo, int centerZ_A, int best_slice_B) {
    std::cout << "Salvando imagens de debug em " << OUTPUT_FOLDER << "..." << std::endl;
    image_display::ImageSaver saver(OUTPUT_FOLDER);
    saver.save(pathA, eixo, centerZ_A, id + "_ImgA_CentroZ");
    saver.save(pathB, eixo, best_slice_B, id + "_ImgB_MaisParecida_com_A");
}

void process_image_pair(const std::string& image_id, const std::string& pathA, const std::string& pathB, 
                        std::ofstream& log_file, int window_size = 7, int numThreads = 1, int size_parallel = 10) {
    auto t_start_total = std::chrono::steady_clock::now();
    std::cout << "\n=======================================" << std::endl;
    std::cout << "Processando ID: " << image_id << std::endl;

    auto t_start_read = std::chrono::steady_clock::now();
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
    std::vector<short> volume_B = readerB.getVolumeAs<short>();
    auto t_end_read = std::chrono::steady_clock::now();
    double time_read = std::chrono::duration<double>(t_end_read - t_start_read).count();

    auto dimA = readerA.getDimensions();
    auto dimB = readerB.getDimensions();
    
    int eixo = 2; // Eixo Z para central
    int centerZ_A = dimA[eixo] / 2;
    int cortes_B = dimB[eixo];

    int widthA = 0, heightA = 0;
    std::vector<short> central_sliceA = readerA.getSliceAs<short>(eixo, centerZ_A, widthA, heightA);

    if (central_sliceA.empty()) {
        std::cerr << "Erro: Fatia central de A está vazia." << std::endl;
        return;
    }

    auto t_start_search = std::chrono::steady_clock::now();
    
    double max_ssim = -1.0;
    int best_slice_B = -1;
    double total_ssim_time;
    int ssim_calc_count;

    std::cout << "Buscando melhor match entre " << cortes_B << " fatias de B..." << std::endl;

    int pixels_per_slice = widthA * heightA;

    if (dimB[0] != widthA || dimB[1] != heightA) {
        std::cerr << "Dimensoes de A e B sao incompativeis!" << std::endl;
        return;
    }

    const short* ptr_central_A = central_sliceA.data();

    #pragma omp parallel num_threads(numThreads)
    {
        double local_max_ssim = -1.0;
        int local_best_slice = -1;

        if (size_parallel > 0) {
            #pragma omp for nowait reduction(+:total_ssim_time, ssim_calc_count) schedule(dynamic, size_parallel)
            for (int i = 0; i < cortes_B; ++i) {
                const short* ptr_slice_B = volume_B.data() + (i * pixels_per_slice);
        
                auto t_start_slice = std::chrono::steady_clock::now();
                double data_range = 65535.0; // short range
        
                double current_ssim = metrics::calculate_ssim(ptr_central_A, ptr_slice_B, widthA, heightA, data_range, window_size);
                auto t_end_slice = std::chrono::steady_clock::now();
        
                total_ssim_time += std::chrono::duration<double>(t_end_slice - t_start_slice).count();
                ssim_calc_count++;
        
                if (current_ssim > local_max_ssim) {
                    local_max_ssim = current_ssim;
                    local_best_slice = i;
                }
            }
        } else {
            #pragma omp for nowait reduction(+:total_ssim_time, ssim_calc_count) schedule(dynamic)
            for (int i = 0; i < cortes_B; ++i) {
                const short* ptr_slice_B = volume_B.data() + (i * pixels_per_slice);
        
                auto t_start_slice = std::chrono::steady_clock::now();
                double data_range = 65535.0; // short range
        
                double current_ssim = metrics::calculate_ssim(ptr_central_A, ptr_slice_B, widthA, heightA, data_range, window_size);
                auto t_end_slice = std::chrono::steady_clock::now();
        
                total_ssim_time += std::chrono::duration<double>(t_end_slice - t_start_slice).count();
                ssim_calc_count++;
        
                if (current_ssim > local_max_ssim) {
                    local_max_ssim = current_ssim;
                    local_best_slice = i;
                }
            }
        }

        #pragma omp critical
        {            
            if (local_max_ssim > max_ssim) {
                max_ssim = local_max_ssim;
                best_slice_B = local_best_slice;
            }
        }
    }


    auto t_end_search = std::chrono::steady_clock::now();
    double time_search = std::chrono::duration<double>(t_end_search - t_start_search).count();

    auto t_end_total = std::chrono::steady_clock::now();
    double time_total = std::chrono::duration<double>(t_end_total - t_start_total).count();
    
    double time_avg_ssim = (ssim_calc_count > 0) ? (total_ssim_time / ssim_calc_count) : 0.0;

    if (best_slice_B != -1) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Melhor corte de B: Z=" << best_slice_B << " | SSIM: " << max_ssim << std::endl;
        
        // --- LOG ---
        log_file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
        log_file << "ID da Imagem: " << image_id << "\n";
        log_file << "Kernel Window Size: " << window_size << "\n";
        log_file << "Quantidade de cortes B avaliados: " << cortes_B << "\n";
        log_file << "Tempo de Leitura Total (A+B): " << time_read << " s\n";
        log_file << "Tempo Medio Calculo SSIM (por slice): " << time_avg_ssim << " s\n";
        log_file << "Tempo de Busca (SSIM loop): " << time_search << " s\n";
        log_file << "Tempo Total do Processo: " << time_total << " s\n";
        log_file << "Melhor Slice (Z): " << best_slice_B << "\n";
        log_file << "SSIM Maximo Encontrado: " << max_ssim << "\n";
        log_file << "--------------------------------------------------------\n";
        
        // gerar os .png na pasta data/
        // save_debug_images(image_id, pathA, pathB, eixo, centerZ_A, best_slice_B);
    } else {
        std::cerr << "Nenhum corte valido encontrado em B." << std::endl;
    }
}

bool contains(const std::vector<std::string>& vec, const std::string& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

void get_args(int argc, char* argv[], int& numThreads, int& size_parallel, int& kernel_size) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--size") && i + 1 < argc) {
            size_parallel = std::stoi(argv[++i]);
        } else if ((arg == "-k" || arg == "--kernel") && i + 1 < argc) {
            kernel_size = std::stoi(argv[++i]);
        } else {
            std::cerr << "Uso: " << argv[0] << " [-t threads] [-s block_size] [-k kernel_size]\n";
            std::cerr << "Exemplo: " << argv[0] << " -t 4 -s 16 -k 7\n";
            exit(1);
        }
    }
}

int main(int argc, char* argv[]) {
    int numTheads = omp_get_max_threads();
    int size_parallel = 16;
    int kernel_size = 7;

    get_args(argc, argv, numTheads, size_parallel, kernel_size);

    std::string file_name = "slices_dynamic" + std::to_string(size_parallel) + 
                            "_t" + std::to_string(numTheads) + 
                            "_k" + std::to_string(kernel_size) + ".txt";
    std::string log_file_name = std::string(OUTPUT_FOLDER) + file_name;

    std::ofstream log_file(log_file_name);
    if (!log_file.is_open()) {
        std::cerr << "Falha ao abrir ou criar o " << log_file_name << "!" << std::endl;
        return 1;
    }

    auto date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log_file << "=== INICIANDO BATERIA DE VERIFICACAO - " 
         << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";
    log_file.flush();

    std::cout << "Usando " << numTheads << " threads para processamento." << std::endl;
    std::cout << "Tamanho do bloco para paralelismo: " << size_parallel << std::endl;
    std::cout << "Kernel size para SSIM: " << kernel_size << std::endl;
    
    int numRepeticoes = 5; // número de vezes para repetir o processo (para média de tempos)
    std::string base_path_train = "/home/everaldina/dados/Challenge_data/Training_set/Images";
    std::string base_path_test = "/home/everaldina/dados/Challenge_data/Test_set/Images";
    std::vector<std::string> ids_imagens_train = {
        "TRV1P1", "TRV1P2", "TRV1P3", "TRV1P4", "TRV1P5", "TRV1P6", "TRV1P7", "TRV1P8",
        "TRV3P1", "TRV3P2", "TRV3P3", "TRV3P4", "TRV3P5", "TRV3P6", "TRV3P7", "TRV3P8", 
        "TRV4P1", "TRV4P2", "TRV4P3", "TRV4P4", "TRV4P5", "TRV4P6", "TRV4P7", "TRV4P8",
    };    

    std::vector<std::string> ids_imagens_test = {
        "TEV1P1", "TEV1P2", "TEV1P3", "TEV1P4", "TEV1P5", "TEV1P6", "TEV1P7", "TEV1P8",  
        "TEV3P1", "TEV3P2", "TEV3P3", "TEV3P4", "TEV3P5", "TEV3P6", "TEV3P7", "TEV3P8", 
        "TEV4P1", "TEV4P2", "TEV4P3", "TEV4P4", "TEV4P5", "TEV4P6", "TEV4P7", "TEV4P8", 
        "VAV1P1", "VAV1P2", "VAV2P1", "VAV2P2", "VAV3P1", "VAV3P2", "VAV4P1", "VAV4P2", 
    };

    std::vector<std::string> ids_imagens;
    ids_imagens.reserve(ids_imagens_train.size() + ids_imagens_test.size());
    ids_imagens.insert(ids_imagens.end(), ids_imagens_train.begin(), ids_imagens_train.end());
    ids_imagens.insert(ids_imagens.end(), ids_imagens_test.begin(), ids_imagens_test.end());
    
    std::string pathA, pathB, base_path;
    for (const auto& id : ids_imagens) {
        base_path = (contains(ids_imagens_train, id)) ? base_path_train : base_path_test;
        
        // deslizando B sobre slice central de A
        pathA = base_path + "/" + id + "CTI.mhd";
        pathB = base_path + "/" + id + "CTAI.mhd";

        for (int rep = 0; rep < numRepeticoes; ++rep) {
            process_image_pair(id, pathA, pathB, log_file, kernel_size, numTheads, size_parallel);
            log_file.flush(); // garantir que o log seja escrito a cada repetição
        }

        // adicional para ter mais dados de tempo
        pathA = base_path + "/" + id + "CTAI.mhd";
        pathB = base_path + "/" + id + "CTI.mhd";
        for (int rep = 0; rep < numRepeticoes; ++rep) {
            process_image_pair(id, pathA, pathB, log_file, kernel_size, numTheads, size_parallel);
            log_file.flush(); // garantir que o log seja escrito a cada repetição
        }
    }

    date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log_file << "=== FIM DA BATERIA DE VERIFICACAO - " 
         << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";

    log_file.close();
    std::cout << "\nProcessamento finalizado! Log salvo em " << log_file_name << "." << std::endl;

    return 0;
}
