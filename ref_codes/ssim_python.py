import time
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from io import TextIOWrapper


def process_image_pair(pathA: str, pathB: str, image_id: str, 
                       log_file: TextIOWrapper, win_size: int):
    """
        Processa um par de imagens (A e B) para encontrar o corte de B
        que mais se assemelha ao corte central de A usando SSIM.

        Args:
            pathA (str): Caminho para a imagem A (CTI ou CTAI).
            pathB (str): Caminho para a imagem B (CTAI ou CTI).
            image_id (str): Identificador da imagem para log.
            log_file (TextIOWrapper): Arquivo de log para registrar os resultados.
            win_size (int): Tamanho da janela para cálculo do SSIM.
    """
    t_start_total = time.time()

    print(f"\n=======================================")
    print(f"Processando ID: {image_id}")

    t_start_read = time.time()
    try:
        # sitk returns in order: (Z, Y, X)
        volA = sitk.GetArrayFromImage(sitk.ReadImage(pathA))
        volB = sitk.GetArrayFromImage(sitk.ReadImage(pathB))
    except Exception as e:
        print(f"Falha ao ler as imagens: {e}")
        return

    t_end_read = time.time()
    time_read = t_end_read - t_start_read

    dimZ_A = volA.shape[0]
    dimZ_B = volB.shape[0]

    centerZ_A = dimZ_A // 2
    cortes_B = dimZ_B

    central_sliceA = volA[centerZ_A, :, :]

    t_start_search = time.time()

    max_ssim = -1.0
    best_slice_B = -1
    total_ssim_time = 0.0
    ssim_calc_count = 0

    print(f"Buscando melhor match entre {cortes_B} fatias de B...")

    for i in range(cortes_B):
        sliceB_temp = volB[i, :, :]

        if central_sliceA.shape == sliceB_temp.shape:
            t_start_ssim = time.time()
            current_ssim = ssim(
                im1=central_sliceA,
                im2=sliceB_temp,
                data_range=65535.0,
                win_size=win_size
            )
            t_end_ssim = time.time()

            total_ssim_time += (t_end_ssim - t_start_ssim)
            ssim_calc_count += 1

            if current_ssim > max_ssim:
                max_ssim = current_ssim
                best_slice_B = i

    t_end_search = time.time()
    time_search = t_end_search - t_start_search

    t_end_total = time.time()
    time_total = t_end_total - t_start_total

    time_avg_ssim = (total_ssim_time / ssim_calc_count) if ssim_calc_count > 0 else 0.0

    if best_slice_B != -1:
        print(f"Melhor corte de B: Z={best_slice_B} | SSIM: {max_ssim:.6f}")

        log_file.write(f"ID da Imagem: {image_id}\n")
        log_file.write(f"Kernel Window Size: {win_size}\n")
        log_file.write(f"Quantidade de cortes B avaliados: {cortes_B}\n")
        log_file.write(f"Tempo de Leitura Total (A+B): {time_read:.17g} s\n")
        log_file.write(f"Tempo Medio Calculo SSIM (por slice): {time_avg_ssim:.17g} s\n")
        log_file.write(f"Tempo de Busca (SSIM loop): {time_search:.17g} s\n")
        log_file.write(f"Tempo Total do Processo: {time_total:.17g} s\n")
        log_file.write(f"Melhor Slice (Z): {best_slice_B}\n")
        log_file.write(f"SSIM Maximo Encontrado: {max_ssim:.17g}\n")
        log_file.write("--------------------------------------------------------\n")
    else:
        print("Nenhum corte valido encontrado em B.")


def main():
    base_path_train = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images"
    base_path_test = "D:/workspace_data/tcc/orcascore/Challenge_data/Test_set/Images"

    ids_imagens_train = [
        "TRV1P1", "TRV1P2", "TRV1P3", "TRV1P4", "TRV1P5", "TRV1P6", "TRV1P7",
        "TRV1P8", "TRV2P1", "TRV2P2", "TRV2P3", "TRV2P4", "TRV2P5", "TRV2P6",
        "TRV2P7", "TRV2P8", "TRV3P1", "TRV3P2", "TRV3P3", "TRV3P4", "TRV3P5",
        "TRV3P6", "TRV3P7", "TRV3P8", "TRV4P1", "TRV4P2", "TRV4P3", "TRV4P4",
        "TRV4P5", "TRV4P6", "TRV4P7", "TRV4P8",
    ]
    ids_imagens_test = [
        "TEV1P1", "TEV1P2", "TEV1P3", "TEV1P4", "TEV1P5", "TEV1P6", "TEV1P7",
        "TEV1P8", "TEV2P1", "TEV2P2", "TEV2P3", "TEV2P4", "TEV2P5", "TEV2P6",
        "TEV2P7", "TEV2P8", "TEV3P1", "TEV3P2", "TEV3P3", "TEV3P4", "TEV3P5",
        "TEV3P6", "TEV3P7", "TEV3P8", "TEV4P1", "TEV4P2", "TEV4P3", "TEV4P4",
        "TEV4P5", "TEV4P6", "TEV4P7", "TEV4P8", "VAV1P1", "VAV1P2", "VAV2P1",
        "VAV2P2", "VAV3P1", "VAV3P2", "VAV4P1", "VAV4P2",
    ]

    ids_imagens = ids_imagens_train + ids_imagens_test
    window_sizes = [3, 5, 7, 9]
    
    # script para comparacao de valores de SSIM, nao para teste de performance, entao 1 repeticao é suficiente
    numRepeticoes = 1 

    with open("output/log_python.txt", "w") as log_file:
        log_file.write(f"=== INICIANDO BATERIA DE VERIFICACAO - {time.ctime()} ===\n")

        for image_id in ids_imagens:
            base_path = base_path_train if image_id in ids_imagens_train else base_path_test

            # CTI como A, CTAI como B
            pathA = f"{base_path}/{image_id}CTI.mhd"
            pathB = f"{base_path}/{image_id}CTAI.mhd"
            for win_size in window_sizes:
                for _ in range(numRepeticoes):
                    process_image_pair(pathA, pathB, image_id, log_file, win_size)
                    log_file.flush()

            # CTAI como A, CTI como B
            pathA = f"{base_path}/{image_id}CTAI.mhd"
            pathB = f"{base_path}/{image_id}CTI.mhd"
            for win_size in window_sizes:
                for _ in range(numRepeticoes):
                    process_image_pair(pathA, pathB, image_id, log_file, win_size)
                    log_file.flush()

        log_file.write(f"=== FIM DA BATERIA DE VERIFICACAO - {time.ctime()} ===\n")

    print("\nProcessamento finalizado! Log salvo em data/log_python.txt.")

if __name__ == '__main__':
    main()

