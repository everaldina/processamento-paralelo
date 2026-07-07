import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_time_distribution(df):
    """
    Plota a distribuição de tempos de leitura, busca e tempo total em um boxplot.

    Parâmetros:
    df (DataFrame): DataFrame contendo as colunas 'tempo_leitura', 'tempo_busca' e 'tempo_total'.
    """
    plt.figure(figsize=(8, 6))

    df_melted = df[['tempo_leitura', 'tempo_busca', 'tempo_total']].melt(
        var_name='Etapa do Processamento', 
        value_name='Tempo (s)'
    )

    sns.boxplot(
        data=df_melted,
        x='Etapa do Processamento',
        y='Tempo (s)',
        palette=["#f3a322", '#3498db', "#3D2279"],
        hue='Etapa do Processamento',
    )

    plt.title('Distribuição de Tempos: Leitura, Busca e Tempo Total')
    plt.ylabel('Tempo (Segundos)')
    plt.xlabel('')
    plt.show()
    
    
def scatterline_slices_and_kernels_time(df):
    """
    Plota um gráfico de dispersão com linhas de regressão para mostrar a relação entre a quantidade de cortes e o tempo total, diferenciando por tamanho de janela (kernel).
    
    Parâmetros:
    df (DataFrame): DataFrame contendo as colunas 'cortes', 'tempo_total' e 'windows_size'.
    """
    
    plt.figure(figsize=(10, 6))

    windows_sizes = df['windows_size'].unique()
    colors = sns.color_palette('Set1', n_colors=len(windows_sizes))
    color_map = {ws: colors[i] for i, ws in enumerate(windows_sizes)}

    sns.scatterplot(
        data=df, 
        x='cortes', 
        y='tempo_total', 
        s=100,
        palette=color_map,
        hue='windows_size',
        legend=False
    )

    for ws in windows_sizes:
        subset = df[df['windows_size'] == ws]
        sns.regplot(
            data=subset,
            x='cortes',
            y='tempo_total',
            scatter=False,
            color=color_map[ws],
            line_kws={'linewidth': 1},
            label=f'Window Size {ws}'
        )
    plt.legend(title='Kernel / Window Size')
    plt.title('Tempo total de acordo com a Quantidade de Cortes')
    plt.ylabel('Tempo de total (Segundos)')
    plt.xlabel('Quantidade de Cortes Avaliados (N)')
    plt.show()
