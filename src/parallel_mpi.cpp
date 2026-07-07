#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <fstream>
#include "image_display/save_image.hpp"
#include "image_reader/mhd_reader.hpp"
#include "metrics/ssim.hpp"
#include <iomanip>
#include <limits>
#include <mpi.h>

#define OUTPUT_FOLDER "output/"

// descomentar a chamada em process_image_pair para salvar imagens
void save_debug_images(const std::string& id, const std::string& pathA, const std::string& pathB, int eixo, int centerZ_A, int best_slice_B) {
    std::cout << "Salvando imagens de debug em " << OUTPUT_FOLDER << "..." << std::endl;
    image_display::ImageSaver saver(OUTPUT_FOLDER);
    saver.save(pathA, eixo, centerZ_A, id + "_ImgA_CentroZ");
    saver.save(pathB, eixo, best_slice_B, id + "_ImgB_MaisParecida_com_A");
}

// layout {double, int} compativel com MPI_DOUBLE_INT para uso com MPI_MAXLOC
struct SliceBestSSIM {
    double ssim;
    int    slice_index;
};

void process_image_pair(const std::string& image_id, const std::string& pathA, const std::string& pathB,
                        std::ofstream& log_file, int window_size, int rank, int num_procs) {

    auto t_start_total = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "\n=======================================" << std::endl;
        std::cout << "Processando ID: " << image_id << std::endl;
    }

    
    auto t_start_read = std::chrono::high_resolution_clock::now();

    std::vector<short> central_sliceA;
    std::vector<short> volume_B;
    int cortes_B = 0, widthA = 0, heightA = 0;
    int centerZ_A = 0, eixo = 2;

    // rank 0 le os dados dos arquivos
    int success = 1;
    if (rank == 0) {
        image_reader::MHDReader readerA(pathA);
        image_reader::MHDReader readerB(pathB);

        if (!readerA.readHeader() || !readerA.readData()) {
            std::cerr << "Falha ao ler a imagem A: " << pathA << std::endl;
            success = 0;
        } else if (!readerB.readHeader() || !readerB.readData()) {
            std::cerr << "Falha ao ler a imagem B: " << pathB << std::endl;
            success = 0;
        } else {
            auto dimA = readerA.getDimensions();
            auto dimB = readerB.getDimensions();

            centerZ_A = dimA[eixo] / 2;
            cortes_B  = dimB[eixo];

            central_sliceA = readerA.getSliceAs<short>(eixo, centerZ_A, widthA, heightA);

            if (central_sliceA.empty()) {
                std::cerr << "Erro: Fatia central de A esta vazia." << std::endl;
                success = 0;
            } else if (dimB[0] != widthA || dimB[1] != heightA) {
                std::cerr << "Dimensoes de A e B sao incompativeis!" << std::endl;
                success = 0;
            } else {
                volume_B = readerB.getVolumeAs<short>();
            }
        }
    }

    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!success) return;

    // broadcast das dimensoe
    int meta[4] = {cortes_B, widthA, heightA, centerZ_A};
    MPI_Bcast(meta, 4, MPI_INT, 0, MPI_COMM_WORLD);
    cortes_B  = meta[0];
    widthA    = meta[1];
    heightA   = meta[2];
    centerZ_A = meta[3];

    int pixels_per_slice = widthA * heightA;

    // broadcast da fatia central
    if (rank != 0)
        central_sliceA.resize(pixels_per_slice);
    MPI_Bcast(central_sliceA.data(), pixels_per_slice, MPI_SHORT, 0, MPI_COMM_WORLD);



    int slices_per_proc = cortes_B / num_procs;
    int remainder = cortes_B % num_procs;
    int start_slice, local_count;
    
    // calculo de divisao de fatias para cada processo
    /* EXEMPLO DE DIVISAO DE FATIAS
        30 cortes e 4 processos
        slices_per_proc = 7; remainder = 2
        0: start=0 (0 + 0), count=8 (7 + 1), end=8
        1: start=8 (7 + 1), count=7 (7 + 1), end=15
        2: start=16 (14 + 2), count=7 (7 + 0), end=23
        3: start=23 (21 + 2), count=7 (7 + 0), end=30
    */
   std::vector<int> cortes_enviados(num_procs), ptr_start(num_procs);
   if (rank == 0) {
       for (int n_rank = 0; n_rank < num_procs; ++n_rank) {
           int start_slice = n_rank * slices_per_proc + std::min(n_rank, remainder);
           int num_cortes = slices_per_proc + (n_rank < remainder ? 1 : 0);
           cortes_enviados[n_rank] = num_cortes * pixels_per_slice;
           ptr_start[n_rank] = start_slice * pixels_per_slice;
        }
    }
    start_slice = rank * slices_per_proc + std::min(rank, remainder);
    local_count = slices_per_proc + (rank < remainder ? 1 : 0);

    // alocamento de memoria B local
    std::vector<short> local_volume_B(local_count * pixels_per_slice);
    MPI_Scatterv(volume_B.data(), cortes_enviados.data(), ptr_start.data(), MPI_SHORT,
                 local_volume_B.data(), local_count * pixels_per_slice, MPI_SHORT,
                 0, MPI_COMM_WORLD);

    // rank 0 libera o volume inteiro
    volume_B.clear();
    volume_B.shrink_to_fit();

    auto t_end_read = std::chrono::high_resolution_clock::now();
    double time_read = std::chrono::duration<double>(t_end_read - t_start_read).count();

    if (rank == 0)
        std::cout << "Buscando melhor match entre " << cortes_B << " fatias de B..." << std::endl;

    auto t_start_search = std::chrono::high_resolution_clock::now();

    SliceBestSSIM local_result = {-1.0, -1};
    double local_ssim_time  = 0.0;
    int    local_ssim_count = 0;

    const short* ptr_central_A = central_sliceA.data();
    for (int i = 0; i < local_count; ++i) {
        const short* ptr_slice_B = local_volume_B.data() + (i * pixels_per_slice);

        auto t_s = std::chrono::high_resolution_clock::now();
        double data_range   = 65535.0;
        double current_ssim = metrics::calculate_ssim(ptr_central_A, ptr_slice_B, widthA, heightA, data_range, window_size);
        auto t_e = std::chrono::high_resolution_clock::now();

        local_ssim_time += std::chrono::duration<double>(t_e - t_s).count();
        local_ssim_count++;

        if (current_ssim > local_result.ssim) {
            local_result.ssim        = current_ssim;
            local_result.slice_index = start_slice + i;
        }
    }

    // Reducao das metricas de tempo (soma entre processos -> rank 0)
    double total_ssim_time = 0.0;
    int    ssim_calc_count = 0;
    MPI_Reduce(&local_ssim_time,  &total_ssim_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_ssim_count, &ssim_calc_count, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    // compara resultados locais e encontra o global
    SliceBestSSIM global_result = {-1.0, -1};
    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    auto t_end_search = std::chrono::high_resolution_clock::now();
    double time_search = std::chrono::duration<double>(t_end_search - t_start_search).count();

    auto t_end_total = std::chrono::high_resolution_clock::now();
    double time_total = std::chrono::duration<double>(t_end_total - t_start_total).count();

    // === LOG (somente rank 0) ===
    if (rank == 0) {
        double max_ssim     = global_result.ssim;
        int    best_slice_B = global_result.slice_index;

        double time_avg_ssim = (ssim_calc_count > 0) ? (total_ssim_time / ssim_calc_count) : 0.0;

        if (best_slice_B != -1) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Melhor corte de B: Z=" << best_slice_B << " | SSIM: " << max_ssim << std::endl;

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

            // save_debug_images(image_id, pathA, pathB, eixo, centerZ_A, best_slice_B);
        } else {
            std::cerr << "Nenhum corte valido encontrado em B." << std::endl;
        }
    }
}

bool contains(const std::vector<std::string>& vec, const std::string& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

void get_args(int argc, char* argv[], int& kernel_size) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-k" || arg == "--kernel") && i + 1 < argc) {
            kernel_size = std::stoi(argv[++i]);
        } else {
            std::cerr << "Uso: mpirun -n N " << argv[0] << " [-k kernel_size]\n";
            std::cerr << "Exemplo: mpirun -n 4 " << argv[0] << " -k 7\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int kernel_size = 7;
    get_args(argc, argv, kernel_size);

    std::string log_file_name;
    std::ofstream log_file;

    if (rank == 0) {
        log_file_name = std::string(OUTPUT_FOLDER) + "mpi_t" + std::to_string(num_procs) +
                        "_k" + std::to_string(kernel_size) + ".txt";
        log_file.open(log_file_name);
        if (!log_file.is_open()) {
            std::cerr << "Falha ao abrir ou criar o " << log_file_name << "!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        auto date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        log_file << "=== INICIANDO BATERIA DE VERIFICACAO - "
                 << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";
        log_file.flush();

        std::cout << "Usando " << num_procs << " processos MPI para processamento." << std::endl;
        std::cout << "Kernel size para SSIM: " << kernel_size << std::endl;
    }

    int numRepeticoes = 5;
    std::string base_path_train = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images";
    std::string base_path_test  = "D:/workspace_data/tcc/orcascore/Challenge_data/Test_set/Images";

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
            process_image_pair(id, pathA, pathB, log_file, kernel_size, rank, num_procs);
            if (rank == 0) log_file.flush();
        }

        pathA = base_path + "/" + id + "CTAI.mhd";
        pathB = base_path + "/" + id + "CTI.mhd";
        for (int rep = 0; rep < numRepeticoes; ++rep) {
            process_image_pair(id, pathA, pathB, log_file, kernel_size, rank, num_procs);
            if (rank == 0) log_file.flush();
        }
    }

    if (rank == 0) {
        auto date_time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        log_file << "=== FIM DA BATERIA DE VERIFICACAO - "
                 << std::put_time(std::localtime(&date_time_now), "%c") << " ===\n";
        log_file.close();
        std::cout << "\nProcessamento finalizado! Log salvo em " << log_file_name << "." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
