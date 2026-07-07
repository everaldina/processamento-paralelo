// Compilar/rodar: ver secao "Implementacao CUDA (GPU)" no README.md
//
// Este arquivo usa o mesmo leitor MHD/zlib das outras implementacoes
// (image_reader/mhd_reader.hpp + mhd_reader.cpp), pois os dados deste projeto sao
// .zraw comprimidos e exigem zlib para descompactar. Por isso o comando de build
// passa os DOIS arquivos (parallel_cuda.cu e mhd_reader.cpp) numa unica chamada do
// nvcc: no Windows o nvcc usa o MSVC (cl.exe) como compilador hospedeiro, entao
// mhd_reader.cpp precisa ser recompilado por ele (nao reaproveitar o .o que o
// g++/MSYS2 gera para os outros executaveis) para nao misturar ABIs incompativeis
// (STL/CRT diferentes entre MinGW e MSVC). Isso tambem significa que o zlib usado
// aqui precisa ser uma copia compilada para MSVC (ex.: via vcpkg), separada da que
// o MSYS2 ja usa nos outros builds - ver README para o passo a passo.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include "image_reader/mhd_reader.hpp"
#include "metrics/ssim_kernel.cuh"

#define OUTPUT_FOLDER "output/"


bool check_cuda(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        std::cerr << "Erro CUDA em " << call << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

void process_image_pair(const std::string& image_id, const std::string& pathA, const std::string& pathB,
                         std::ofstream& log_file, int window_size, int threads_per_block) {
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

    auto dimA = readerA.getDimensions();
    auto dimB = readerB.getDimensions();

    int eixo = 2; 
    int centerZ_A = dimA[eixo] / 2;
    int cortes_B  = dimB[eixo];

    int widthA = 0, heightA = 0;
    std::vector<short> central_sliceA = readerA.getSliceAs<short>(eixo, centerZ_A, widthA, heightA);

    if (central_sliceA.empty()) {
        std::cerr << "Erro: Fatia central de A esta vazia." << std::endl;
        return;
    }
    if (dimB[0] != widthA || dimB[1] != heightA) {
        std::cerr << "Dimensoes de A e B sao incompativeis!" << std::endl;
        return;
    }
    if (cortes_B <= 0) {
        std::cerr << "Volume B nao possui cortes validos." << std::endl;
        return;
    }

    std::vector<short> volume_B = readerB.getVolumeAs<short>();
    int pixels_per_slice = widthA * heightA;


    
    std::size_t slice_bytes = static_cast<std::size_t>(pixels_per_slice) * sizeof(short);
    std::size_t volumeB_bytes  = static_cast<std::size_t>(cortes_B) * pixels_per_slice * sizeof(short);
    // armazena os resultados do SSIM de cada corte de B
    std::size_t ssim_bytes    = static_cast<std::size_t>(cortes_B) * sizeof(double);

    // Confere se cabe na VRAM livre antes de alocar
    std::size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (slice_bytes + volumeB_bytes + ssim_bytes > freeMem) {
        std::cerr << "Memoria insuficiente na GPU: par precisa de ~" << (volumeB_bytes / (1024 * 1024))
                   << " MB, ha apenas " << (freeMem / (1024 * 1024)) << " MB livres. Pulando este par." << std::endl;
        return;
    }

    // Aloca memoria na GPU para a fatia central de A, o volume B e os resultados do SSIM
    short* centralA_device = nullptr;
    short* volumeB_device = nullptr;
    double* ssim_results_device = nullptr;
    if (!check_cuda(cudaMalloc(&centralA_device, slice_bytes), "cudaMalloc centralA_device")) {
        return;
    }
    if (!check_cuda(cudaMalloc(&volumeB_device, volumeB_bytes), "cudaMalloc volumeB_device")) {
        cudaFree(centralA_device);
        return;
    }
    if (!check_cuda(cudaMalloc(&ssim_results_device, ssim_bytes), "cudaMalloc ssim_results_device")) {
        cudaFree(centralA_device);
        cudaFree(volumeB_device);
        return;
    }

    // Copia os dados da fatia central de A e do volume B para a GPU
    bool copyOk = check_cuda(cudaMemcpy(centralA_device, central_sliceA.data(), slice_bytes, cudaMemcpyHostToDevice),
                              "cudaMemcpy centralA_device") &&
                  check_cuda(cudaMemcpy(volumeB_device, volume_B.data(), volumeB_bytes, cudaMemcpyHostToDevice),
                              "cudaMemcpy volumeB_device");
    if (!copyOk) {
        cudaFree(centralA_device);
        cudaFree(volumeB_device);
        cudaFree(ssim_results_device);
        return;
    }

    // libera a copia host assim que os dados estao na GPU
    volume_B.clear();
    volume_B.shrink_to_fit();

    auto t_end_read = std::chrono::steady_clock::now();
    double time_read = std::chrono::duration<double>(t_end_read - t_start_read).count();

    std::cout << "Buscando melhor match entre " << cortes_B << " fatias de B..." << std::endl;


    auto t_start_search = std::chrono::steady_clock::now();

    // Calcula o SSIM de cada corte de B em paralelo, usando um bloco por corte
    const double data_range = 65535.0; 
    std::size_t shared_bytes = static_cast<std::size_t>(threads_per_block) * sizeof(double);
    ssim_kernel<<<cortes_B, threads_per_block, shared_bytes>>>(
        centralA_device, volumeB_device, widthA, heightA, window_size, data_range, ssim_results_device);

    bool launchOk = check_cuda(cudaGetLastError(), "lancamento do kernel") &&
                    check_cuda(cudaDeviceSynchronize(), "execucao do kernel");

    std::vector<double> ssim_per_slice(cortes_B, 0.0);
    if (launchOk) {
        launchOk = check_cuda(cudaMemcpy(ssim_per_slice.data(), ssim_results_device, ssim_bytes, cudaMemcpyDeviceToHost),
                               "cudaMemcpy ssim_per_slice");
    }

    cudaFree(centralA_device);
    cudaFree(volumeB_device);
    cudaFree(ssim_results_device);

    if (!launchOk) return;

    // encontra o corte de B com o maior SSIM
    double max_ssim = -1.0;
    int best_slice_B = -1;
    for (int i = 0; i < cortes_B; ++i) {
        if (ssim_per_slice[i] > max_ssim) {
            max_ssim = ssim_per_slice[i];
            best_slice_B = i;
        }
    }

    auto t_end_search = std::chrono::steady_clock::now();
    double time_search = std::chrono::duration<double>(t_end_search - t_start_search).count();

    auto t_end_total = std::chrono::steady_clock::now();
    double time_total = std::chrono::duration<double>(t_end_total - t_start_total).count();

    // calcula o tempo medio de calculo do SSIM por slice
    double time_avg_ssim = (cortes_B > 0) ? (time_search / cortes_B) : 0.0;

    if (best_slice_B != -1) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Melhor corte de B: Z=" << best_slice_B << " | SSIM: " << max_ssim << std::endl;

        log_file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
        log_file << "ID da Imagem: " << image_id << "\n";
        log_file << "Kernel Window Size: " << window_size << "\n";
        log_file << "Threads por Bloco: " << threads_per_block << "\n";
        log_file << "Quantidade de cortes B avaliados: " << cortes_B << "\n";
        log_file << "Tempo de Leitura Total (A+B): " << time_read << " s\n";
        log_file << "Tempo Medio Calculo SSIM (por slice): " << time_avg_ssim << " s\n";
        log_file << "Tempo de Busca (SSIM loop): " << time_search << " s\n";
        log_file << "Tempo Total do Processo: " << time_total << " s\n";
        log_file << "Melhor Slice (Z): " << best_slice_B << "\n";
        log_file << "SSIM Maximo Encontrado: " << max_ssim << "\n";
        log_file << "--------------------------------------------------------\n";
    } else {
        std::cerr << "Nenhum corte valido encontrado em B." << std::endl;
    }
}

bool contains(const std::vector<std::string>& vec, const std::string& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

int round_up_pow2(int value) {
    if (value < 1) value = 1;
    int powered = 1;
    while (powered < value) powered <<= 1;
    return powered;
}

void get_args(int argc, char* argv[], int& kernel_size, int& threads_per_block, int& device_id) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-k" || arg == "--kernel") && i + 1 < argc) {
            kernel_size = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads_per_block = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else {
            std::cerr << "Uso: " << argv[0] << " [-k kernel_size] [-t threads_per_block] [-d device_id]\n";
            std::cerr << "Exemplo: " << argv[0] << " -k 7 -t 256\n";
            std::exit(1);
        }
    }
}

int main(int argc, char* argv[]) {
    int kernel_size = 7;
    int threads_per_block = 256;
    int device_id = 0;

    get_args(argc, argv, kernel_size, threads_per_block, device_id);

    // Numero de threads por bloco é uma potencia de 2 menor ou igual a 1024
    int adjusted = round_up_pow2(threads_per_block);
    if (adjusted > 1024) adjusted = 1024;
    if (adjusted != threads_per_block) {
        std::cout << "Aviso: threads_per_block ajustado de " << threads_per_block << " para " << adjusted
                  << " (precisa ser potencia de 2, ate 1024, para a reducao em memoria compartilhada)." << std::endl;
        threads_per_block = adjusted;
    }

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) return 1;

    cudaDeviceProp prop;
    if (!check_cuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties")) return 1;

    std::size_t freeMem = 0, totalMem = 0;
    check_cuda(cudaMemGetInfo(&freeMem, &totalMem), "cudaMemGetInfo");

    std::string log_file_name = std::string(OUTPUT_FOLDER) + "cuda_t" + std::to_string(threads_per_block) +
                                 "_k" + std::to_string(kernel_size) + ".txt";
    std::ofstream log_file(log_file_name);
    if (!log_file.is_open()) {
        std::cerr << "Falha ao abrir ou criar o " << log_file_name << "!" << std::endl;
        return 1;
    }

    auto date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log_file << "=== INICIANDO BATERIA DE VERIFICACAO - "
             << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";
    log_file.flush();

    std::cout << "Usando GPU: " << prop.name << " (compute capability " << prop.major << "." << prop.minor
              << ", device " << device_id << ")" << std::endl;
    std::cout << "Memoria da GPU: " << (freeMem / (1024 * 1024)) << " MB livres de "
              << (totalMem / (1024 * 1024)) << " MB totais." << std::endl;
    std::cout << "Threads por bloco: " << threads_per_block << std::endl;
    std::cout << "Kernel size para SSIM: " << kernel_size << std::endl;

    int numRepeticoes = 5;
    std::string base_path_train = "/home/everaldina/dados/Challenge_data/Training_set/Images";
    std::string base_path_test  = "/home/everaldina/dados/Challenge_data/Test_set/Images";

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

        pathA = base_path + "/" + id + "CTI.mhd";
        pathB = base_path + "/" + id + "CTAI.mhd";
        for (int rep = 0; rep < numRepeticoes; ++rep) {
            process_image_pair(id, pathA, pathB, log_file, kernel_size, threads_per_block);
            log_file.flush();
        }

        pathA = base_path + "/" + id + "CTAI.mhd";
        pathB = base_path + "/" + id + "CTI.mhd";
        for (int rep = 0; rep < numRepeticoes; ++rep) {
            process_image_pair(id, pathA, pathB, log_file, kernel_size, threads_per_block);
            log_file.flush();
        }
    }

    date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log_file << "=== FIM DA BATERIA DE VERIFICACAO - "
             << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";
    log_file.close();

    std::cout << "\nProcessamento finalizado! Log salvo em " << log_file_name << "." << std::endl;

    return 0;
}
