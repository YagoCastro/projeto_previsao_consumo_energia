import pandas as pd
import numpy as np

def sugerir_tipo(valor_min, valor_max, tipo_original):
    """
    Sugere um tipo mais econômico baseado nos valores mínimo e máximo.
    """
    if pd.api.types.is_integer_dtype(tipo_original):
        if valor_min >= 0:
            if valor_max <= 255:
                return 'uint8'
            elif valor_max <= 65535:
                return 'uint16'
            elif valor_max <= 4294967295:
                return 'uint32'
        else:
            if -128 <= valor_min <= valor_max <= 127:
                return 'int8'
            elif -32768 <= valor_min <= valor_max <= 32767:
                return 'int16'
            elif -2147483648 <= valor_min <= valor_max <= 2147483647:
                return 'int32'
        return 'int64'
    
    elif pd.api.types.is_float_dtype(tipo_original):
        return 'float32' if tipo_original == 'float64' else str(tipo_original)

    return str(tipo_original)

def generate_metadata(dataframe):
    """
    Gera um dataframe contendo metadados das colunas do dataframe fornecido.
    """
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'qt_nulos': dataframe.isnull().sum(),
        'percent_nulos': round((dataframe.isnull().sum() / len(dataframe)) * 100, 2),
        'cardinalidade': dataframe.nunique(),
    })

    min_vals = []
    max_vals = []
    sugestoes_tipo = []

    for col in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            min_vals.append(col_min)
            max_vals.append(col_max)
            sugestoes_tipo.append(sugerir_tipo(col_min, col_max, dataframe[col].dtype))
        else:
            min_vals.append(None)
            max_vals.append(None)
            sugestoes_tipo.append(None)

    metadata['min'] = min_vals
    metadata['max'] = max_vals
    metadata['sugestao_tipo'] = sugestoes_tipo

    metadata = metadata.sort_values(by='tipo').reset_index(drop=True)
    return metadata

# Definindo df_estatisticas fora da função
df_estatisticas = pd.DataFrame()

def calcular_estatisticas(df, coluna, nome_variavel):
    """
    Calcula estatísticas descritivas de uma coluna numérica de um DataFrame.

    Parameters:
    df (pd.DataFrame): O DataFrame contendo os dados.
    coluna (str): O nome da coluna para calcular as estatísticas.

    Returns:
    dict: Um dicionário contendo várias estatísticas, ou uma mensagem de erro se a coluna não for numérica.
    """
    global df_estatisticas  # Acessando a variável global

    if coluna not in df.columns:
        return {"erro": "A coluna especificada não existe no DataFrame."}

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        return {"erro": "A coluna especificada não é numérica."}

    # INFORMAÇÕES SOBRE O DATASET
    print('----------- Informações sobre o Dataset -----------')
    num_linhas = df.shape[0]
    print(f'Número de linhas do dataset: {num_linhas}')
    
    # Calculando a quantidade de valores únicos
    valores_unicos = df[coluna].nunique()
    print(f'Valores únicos da coluna: {valores_unicos}')
    
    # MEDIDAS DE POSIÇÃO
    print('----------- Medidas de Posição -----------')
    
    # Calculando a Média
    media = df[coluna].mean()
    print(f'média: {media}')
    
    # Calculando a Mediana
    mediana = df[coluna].median()
    print(f'mediana: {mediana}')
    
    # Calculando a Moda
    frequencia = df[coluna].value_counts()
    max_frequencia = frequencia.max()
    modas = frequencia[frequencia == max_frequencia].index.tolist()
    
    if max_frequencia == 1:
        print("Não há moda.")
    elif len(modas) == len(frequencia):
        print("Não há moda, todos os elementos aparecem com a mesma frequência.")
    else:
        print(f'moda: {modas}')
        print(f"Frequência da moda: {max_frequencia}")
    
    # Calculando os Quartis
    quartis = df[coluna].quantile([0.25, 0.5, 0.75]).tolist()  # Q1, Q2 (mediana), Q3

    # Exibindo os Quartis
    print(f'Primeiro Quartil (Q1) [25%]: {quartis[0]}')
    print(f'Segundo Quartil (Mediana, Q2) [50%]: {quartis[1]}')
    print(f'Terceiro Quartil (Q3) [75%]: {quartis[2]}')
    
    # MEDIDAS DE Dispersão
    print('----------- Medidas de Dispersão -----------')
    
    # Calculando o Máximo do conjunto
    maximo  = df[coluna].max()
    print(f'O máximo do conjunto é  {maximo}')
    
    # Calculando o Mínimo do conjunto
    minimo  = df[coluna].min()
    print(f'O minimo do conjunto é  {minimo}')
    
    # Calculando a Amplitude
    amplitude  =  df[coluna].max() - df[coluna].min()
    print(f'A amplitude do conjunto é  {amplitude}')
    
    # Calculando a Variância
    variancia = df[coluna].var()
    print(f'A variância do conjunto: {variancia}')
    
    # Calculando o Desvio
    desvio = df[coluna].std()
    print(f'O desvio padrão do conjunto: {desvio}')
    
    # Calculando o Erro padrão
    erro_padrao = desvio / (len(df[coluna]) ** 0.5)  # Erro padrão da média
    print(f'O Erro padrão: {erro_padrao}')
    
    # Calculando o Coeficiente de variação
    coef_variacao = (desvio / media) * 100 if media != 0 else None  # Coeficiente de variação
    print(f'O Coeficiente de variação: {coef_variacao}')
    
    # Calculando a Assimetria
    assimetria = df[coluna].skew()  # Assimetria
    print(f'A assimetria: {assimetria}')
    
    # Calculando a Curtose
    curtose = df[coluna].kurtosis()  # Curtose
    print(f'A curtose: {curtose:.2f}')
    
    # Criando o DataFrame de resultado
    resultado = pd.DataFrame({
        'variável': [nome_variavel],
        'média': [media],
        'desvio_padrao': [desvio],
        'assimetria': [assimetria],
        'curtose': [curtose],
        'valor_min': [minimo],
        '90%': [np.percentile(df[coluna], 90)],
        '99%': [np.percentile(df[coluna], 99)],
        'valor_max': [maximo],
        'amplitude': [amplitude]
    })
    
    # Usando pd.concat para adicionar os resultados ao DataFrame global
    df_estatisticas = pd.concat([df_estatisticas, resultado], ignore_index=True)

    # Exibindo o DataFrame com as estatísticas acumuladas
    #display(df_estatisticas)

    return display(df_estatisticas)  # Agora retorna o DataFrame atualizado