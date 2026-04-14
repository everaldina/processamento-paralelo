import time
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

def process_image_pair(base_path, image_id, log_file):
    t_start_total = time.time()

    pathA = f"{base_path}/{image_id}CTI.mhd"
    pathB = f"{base_path}/{image_id}CTAI.mhd"

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
            current_ssim = ssim(central_sliceA, sliceB_temp, data_range=65535.0, win_size=7)
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
        print(f"Melhor corte de B: Z={best_slice_B} | SSIM: {max_ssim}")
        
        log_file.write(f"ID da Imagem: {image_id}\n")
        log_file.write(f"Quantidade de cortes B avaliados: {cortes_B}\n")
        log_file.write(f"Tempo de Leitura Total (A+B): {time_read} s\n")
        log_file.write(f"Tempo Medio Calculo SSIM (por slice): {time_avg_ssim} s\n")
        log_file.write(f"Tempo de Busca (SSIM loop): {time_search} s\n")
        log_file.write(f"Tempo Total do Processo: {time_total} s\n")
        log_file.write(f"Melhor Slice (Z): {best_slice_B}\n")
        log_file.write(f"SSIM Maximo Encontrado: {max_ssim}\n")
        log_file.write("--------------------------------------------------------\n")
    else:
        print("Nenhum corte valido encontrado em B.")

def main():
    base_path = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images"
    ids_imagens = [
        "TRV1P1", "TRV1P2", "TRV1P3", "TRV1P4", "TRV1P5", 
        "TRV1P6", "TRV1P7", "TRV1P8", "TRV2P1", "TRV2P2", "TRV2P3", 
        "TRV2P4", "TRV2P5", "TRV2P6", "TRV2P7", "TRV2P8", "TRV3P1", 
        "TRV3P2", "TRV3P3", "TRV3P4", "TRV3P5", "TRV3P6", "TRV3P7", 
        "TRV3P8", "TRV4P1", "TRV4P2", "TRV4P3", "TRV4P4", "TRV4P5", 
        "TRV4P6", "TRV4P7", "TRV4P8",
    ]

    with open("data/log_python.txt", "w") as log_file:
        log_file.write("=== INICIANDO BATERIA DE VERIFICACAO ===\n")
        for image_id in ids_imagens:
            process_image_pair(base_path, image_id, log_file)

    print("\nProcessamento finalizado! Log salvo em data/log_python.txt.")

if __name__ == '__main__':
    main()

