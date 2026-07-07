import pandas as pd

def read_log(path: str) -> str:
    """Lê o conteúdo do arquivo de log e retorna como string."""
    with open(path, 'r') as f:
        log_content = f.read()
    return log_content

def parse_log(log_data: str) -> pd.DataFrame:
    """Converte o conteúdo do log estruturado em um DataFrame do pandas."""
    records = []
    current_record = {}

    for line in log_data.strip().split('\n'):
        line = line.strip()
        if line.startswith('ID da Imagem:'):
            current_record['id'] = line.split(': ')[1]
        elif line.startswith('Quantidade de cortes B avaliados:'):
            current_record['cortes'] = int(line.split(': ')[1])
        elif line.startswith('Kernel Window Size:'):
            current_record['windows_size'] = line.split(': ')[1]
        elif line.startswith('Tempo de Leitura Total (A+B):'):
            current_record['tempo_leitura'] = float(line.split(': ')[1].replace(' s', ''))
        elif line.startswith('Tempo Medio Calculo SSIM (por slice):'):
            current_record['tempo_medio_slice'] = float(line.split(': ')[1].replace(' s', ''))
        elif line.startswith('Tempo de Busca (SSIM loop):'):
            current_record['tempo_busca'] = float(line.split(': ')[1].replace(' s', ''))
        elif line.startswith('Tempo Total do Processo:'):
            current_record['tempo_total'] = float(line.split(': ')[1].replace(' s', ''))
        elif line.startswith('Melhor Slice (Z):'):
            current_record['melhor_corte'] = int(line.split(': ')[1])
        elif line.startswith('SSIM Maximo Encontrado:'):
            current_record['ssim_maximo'] = float(line.split(': ')[1])
            records.append(current_record)
            current_record = {}
    return pd.DataFrame(records)