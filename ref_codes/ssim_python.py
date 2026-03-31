import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

def normalize_to_uint8(array):
    min_val = array.min()
    max_val = array.max()
    if max_val - min_val == 0:
        return np.zeros_like(array, dtype=np.uint8)
    norm = ((array - min_val) / (max_val - min_val)) * 255.0
    return norm.astype(np.uint8)

def main():
    pathA = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images/TRV1P1CTAI.mhd"
    pathB = "D:/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images/TRV1P1CTI.mhd"

    print("Carregando imagens com SimpleITK...")
    try:
        # sitk returns in order: (Z, Y, X)
        volA = sitk.GetArrayFromImage(sitk.ReadImage(pathA))
        volB = sitk.GetArrayFromImage(sitk.ReadImage(pathB))
    except Exception as e:
        print(f"Erro ao carregar imagens: {e}")
        return

    # Normalizar para 8-bit assim como fizemos no C++ para ser uma comparacao justa
    volA = normalize_to_uint8(volA)
    volB = normalize_to_uint8(volB)
    
    dimZ_A = volA.shape[0]
    dimZ_B = volB.shape[0]
    
    centerZ_A = dimZ_A // 2
    centerZ_B = dimZ_B // 2

    print(f"Volumes carregados! Dim A: {volA.shape}, Dim B: {volB.shape}")

    sliceA_center = volA[centerZ_A, :, :]
    sliceB_center = volB[centerZ_B, :, :]

    # Calculando os dois centros primeiro
    ssim_val, _ = ssim(sliceA_center, sliceB_center, data_range=255)
    print(f"SSIM (centro A [Z={centerZ_A}] x centro B [Z={centerZ_B}]): {ssim_val:.6f}")

    print(f"\\nCalculando o SSIM do corte central de A contra todos os {dimZ_B} cortes de B...")
    max_ssim = -1.0
    best_slice_B = -1

    for i in range(dimZ_B):
        sliceB_temp = volB[i, :, :]
        
        # Validar as dimensoes (que no SimpleITK sao Y e X nesse slice)
        if sliceA_center.shape == sliceB_temp.shape:
            current_ssim = ssim(sliceA_center, sliceB_temp, data_range=255, win_size=7)
            
            if current_ssim > max_ssim:
                max_ssim = current_ssim
                best_slice_B = i

    print("=======================================")
    print("RESULTADO FINAL PYTHON:")
    print(f"O corte de B mais parecido com o centro de A (Z={centerZ_A}) foi o corte Z={best_slice_B} com um SSIM de {max_ssim:.6f}")

if __name__ == '__main__':
    main()
