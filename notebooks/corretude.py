import pandas as pd

def teste_corretude(df_ref: pd.DataFrame, df_comp: pd.DataFrame, lista_ids: list,
                    nome_ref: str, nome_comp: str,
                    tolerance_ssim: float = 1e-25) -> dict:
    """" 
    Compara os resultados de dois DataFrames para verificar a corretude dos resultados. 
    São comparados o melhor corte encontrado e o valor máximo de SSIM para cada combinação de ID, e número de cortes.
    
    Args:
        df_ref (pd.DataFrame): DataFrame de referência a ser comparado.
        df_comp (pd.DataFrame): DataFrame a ser comparado com o de referência.
        ids (list): Lista de IDs a serem verificados.
        nome_ref (str): Nome do DataFrame de referência para logs.
        nome_comp (str): Nome do DataFrame de comparação para logs.
        tolerance_ssim (float, optional): Tolerância para comparação de valores de SSIM. Defaults to 1e-25.
    Returns:
        dict: Dicionário contendo o número total de comparações realizadas (num_comp), 
        o número de diferenças encontradas nos valores de SSIM (num_ssim_diff) e 
        o número de diferenças encontradas nos melhores cortes (num_index_diff).
    """

    num_comp = 0
    num_index_diff = 0
    num_ssim_diff = 0

    for i, id in enumerate(lista_ids):
        imprimir = (i < 2) or (i >= len(lista_ids) - 2)  # Imprime os primeiros e últimos 5 IDs para verificação
        
        df_select_ref = df_ref[df_ref['id'] == id]
        df_select_comp = df_comp[df_comp['id'] == id]
        if df_select_ref.empty or df_select_comp.empty:
            print(f"Falha: ID {id} não encontrado em um dos DataFrames. Verifique os dados de entrada.")
            continue
        
        # Dois valores de cortes, um com comparaçao de tomografia com contraste como referencia, 
        # e outro sem contraste como referencia, entao é esperado que haja 2 cortes diferentes para cada ID
        cortes = set(list(df_select_ref['cortes'].unique()) + list(df_select_comp['cortes'].unique()))
        if len(cortes) != 2:
            print(f"Alerta: Número inesperado de cortes para ID {id}. Encontrados cortes: {cortes}")

        for corte in cortes:
            df_corte_ref = df_select_ref[df_select_ref['cortes'] == corte]
            df_corte_comp = df_select_comp[df_select_comp['cortes'] == corte]
            
            if df_corte_ref.empty or df_corte_comp.empty:
                print(f"ID {id} | Cortes {corte}: Faltando em um dos DataFrames (ignorando corte).")
                continue
            
            num_comp += 1
            
            #### COMPARAÇÃO DE MELHOR CORTE ENTRE PARALÉLO E SEQUENCIAL ####
            index_set_ref = set(df_corte_ref['melhor_corte'])
            index_set_comp = set(df_corte_comp['melhor_corte'])
            
            indice_correto = (index_set_ref == index_set_comp)
            if not indice_correto:
                num_index_diff += 1
            
            # Logs de warninga para resultados incosistentes
            if len(index_set_ref) > 1:
                print(f"Variação nos indices '{nome_ref}' | ID {id} | Cortes {corte} | Valores: {index_set_ref}")
            if len(index_set_comp) > 1:
                print(f"Variação nos indices '{nome_comp}' | ID {id} | Cortes {corte} | Valores: {index_set_comp}")

            
            #### COMPARAÇÃO DE VALORES DE SSIM MÁXIMO ENTRE PARALÉLO E SEQUENCIAL ####
            set_max_ssim_ref = set(df_corte_ref['ssim_maximo'])
            set_max_ssim_comp = set(df_corte_comp['ssim_maximo'])
            
            max_diff = 0
            list_max_ssim_comp = list(set_max_ssim_comp)
            for i in range(len(list_max_ssim_comp)):
                for j in range(i + 1, len(list_max_ssim_comp)):
                    diff = abs(list_max_ssim_comp[i] - list_max_ssim_comp[j])
                    if diff > max_diff:
                        max_diff = diff
                
            

            # Logs de warninga para resultados incosistentes
            if len(set_max_ssim_ref) > 1:
                print(f"Variação nos valores de SSIM '{nome_ref}' | ID {id} | Cortes {corte} | Valores: {set_max_ssim_ref}")
            if (len(set_max_ssim_comp) > 1) and (max_diff > tolerance_ssim):
                print(f"Variação nos valores de SSIM '{nome_comp}' | ID {id} | Cortes {corte} | Valores: {set_max_ssim_comp}")
            
            ssim_correto = True
            if abs(next(iter(set_max_ssim_ref)) - next(iter(set_max_ssim_comp))) > tolerance_ssim:
                ssim_correto = False
                num_ssim_diff += 1
                
            if imprimir or not indice_correto or not ssim_correto:
                print(f"ID {id} | Cortes {corte}")
                
                if not indice_correto:
                    print(f"  - Índice: {nome_ref} {index_set_ref} != {nome_comp} {index_set_comp}")
                if not ssim_correto:
                    print(f"  - SSIM: {nome_ref} {set_max_ssim_ref} != {nome_comp} {set_max_ssim_comp}")
                    
                if indice_correto and ssim_correto:
                    print(f"  - Resultados consistentes")

    return {
        'num_comp': num_comp,
        'num_index_diff': num_index_diff,
        'num_ssim_diff': num_ssim_diff
    }