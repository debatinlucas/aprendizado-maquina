import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

CAMINHO_ZIP = "dataset.zip"
PASTA_SAIDA = "resultados_random_forest"

UF_ANALISADA = "SC"
RANDOM_STATE = 42

VARIAVEL_ALVO = "grupo_tipo_acidente"

VARIAVEIS_MODELO = [
    "dia_semana",
    "br",
    "km",
    "causa_acidente",
    "fase_dia",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
]

VARIAVEIS_CATEGORICAS = [
    "dia_semana",
    "causa_acidente",
    "fase_dia",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
]

VARIAVEIS_NUMERICAS = [
    "br",
    "km",
]

COLUNAS_NECESSARIAS = VARIAVEIS_MODELO + [VARIAVEL_ALVO, "uf"]

def criar_pasta_saida(pasta: str) -> None:
    os.makedirs(pasta, exist_ok=True)

def carregar_datasets_zip(caminho_zip: str) -> pd.DataFrame:
    if not os.path.exists(caminho_zip):
        raise FileNotFoundError(
            f"Arquivo ZIP não encontrado: {caminho_zip}. "
        )
    lista_datasets = []

    with zipfile.ZipFile(caminho_zip, "r") as zip_ref:
        arquivos_csv = [nome for nome in zip_ref.namelist() if nome.lower().endswith(".csv")]
        if not arquivos_csv:
            raise ValueError("Nenhum arquivo CSV foi encontrado dentro do ZIP.")
        for nome_arquivo in arquivos_csv:
            print(f"Lendo arquivo: {nome_arquivo}")
            with zip_ref.open(nome_arquivo) as arquivo:
                df_temp = pd.read_csv(
                    arquivo,
                    sep=";",
                    encoding="latin1",
                    low_memory=False,
                )
                lista_datasets.append(df_temp)
    df = pd.concat(lista_datasets, ignore_index=True)
    df["grupo_tipo_acidente"] = df["tipo_acidente"].apply(agrupar_tipo_acidente)
    return df

def agrupar_tipo_acidente(tipo):
    tipo = str(tipo).strip().lower()

    if tipo in [
        "colisão traseira",
        "colisão lateral",
        "colisão lateral mesmo sentido",
        "colisão lateral sentido oposto",
        "colisão transversal"
    ]:
        return "Colisão"

    elif tipo in [
        "colisão frontal"
    ]:
        return "Colisão grave"

    elif tipo in [
        "saída de leito carroçável"
    ]:
        return "Saída de pista"

    elif tipo in [
        "tombamento",
        "capotamento",
        "queda de ocupante de veículo"
    ]:
        return "Perda de controle"

    elif tipo in [
        "atropelamento de pedestre",
        "atropelamento de animal"
    ]:
        return "Atropelamento"

    elif tipo in [
        "colisão com objeto",
        "colisão com objeto estático",
        "colisão com objeto em movimento"
    ]:
        return "Colisão com objeto"

    else:
        return "Outros"


def limpar_dataset(df: pd.DataFrame, pasta_saida: str) -> pd.DataFrame:
    df_sc = df[df["uf"] == UF_ANALISADA].copy()
    print(f"\nRegistros de {UF_ANALISADA} antes da seleção de variáveis: {df_sc.shape[0]}")

    df_modelo = df_sc[VARIAVEIS_MODELO + [VARIAVEL_ALVO]].copy()

    print("\nValores nulos antes do tratamento:")
    print(df_modelo.isnull().sum())

    qtd_antes = df_modelo.shape[0]

    df_modelo_limpo = df_modelo.dropna(subset=VARIAVEIS_MODELO + [VARIAVEL_ALVO]).copy()

    df_modelo_limpo["km"] = (
        df_modelo_limpo["km"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df_modelo_limpo["km"] = pd.to_numeric(df_modelo_limpo["km"], errors="coerce")

    df_modelo_limpo["br"] = pd.to_numeric(df_modelo_limpo["br"], errors="coerce")

    df_modelo_limpo = df_modelo_limpo.dropna(subset=["br", "km"]).copy()

    qtd_depois = df_modelo_limpo.shape[0]

    print("\nValores nulos depois do tratamento:")
    print(df_modelo_limpo.isnull().sum())

    print(f"\nQuantidade de registros antes da limpeza: {qtd_antes}")
    print(f"Quantidade de registros depois da limpeza: {qtd_depois}")
    print(f"Registros removidos: {qtd_antes - qtd_depois}")

    caminho_saida = os.path.join(pasta_saida, "dataset_sc_2017_2025_supervisionado_limpo.csv")
    df_modelo_limpo.to_csv(caminho_saida, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nDataset limpo salvo em: {caminho_saida}")

    return df_modelo_limpo


def gerar_grafico_distribuicao_alvo(df: pd.DataFrame, pasta_saida: str) -> None:
    plt.figure(figsize=(12, 6))
    df[VARIAVEL_ALVO].value_counts().plot(kind="bar")
    plt.title("Distribuição dos tipos de acidente")
    plt.xlabel("Tipo de acidente")
    plt.ylabel("Quantidade de registros")
    plt.xticks(rotation=90)
    plt.tight_layout()

    caminho = os.path.join(pasta_saida, "grafico_distribuicao_tipo_acidente.png")
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico de distribuição salvo em: {caminho}")


def dividir_dados(df: pd.DataFrame):
    X = df[VARIAVEIS_MODELO]
    y = df[VARIAVEL_ALVO]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    total = len(X)
    print("\nDivisão dos dados:")
    print(f"Treino: {len(X_train)} registros ({len(X_train) / total:.2%})")
    print(f"Validação: {len(X_val)} registros ({len(X_val) / total:.2%})")
    print(f"Teste: {len(X_test)} registros ({len(X_test) / total:.2%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def criar_modelo_random_forest() -> Pipeline:
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                VARIAVEIS_CATEGORICAS,
            ),
            (
                "numericas",
                "passthrough",
                VARIAVEIS_NUMERICAS,
            ),
        ]
    )
    modelo = Pipeline(
        steps=[
            ("preprocessador", preprocessador),
            (
                "classificador",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return modelo


def avaliar_modelo(modelo, X, y, nome_conjunto: str, pasta_saida: str = None):
    y_pred = modelo.predict(X)
    acuracia = accuracy_score(y, y_pred)

    print(f"\nResultados no conjunto de {nome_conjunto}")
    print(f"Acurácia: {acuracia:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(y, y_pred, zero_division=0))

    classes = sorted(y.unique())
    matriz = confusion_matrix(y, y_pred, labels=classes)

    df_matriz = pd.DataFrame(matriz, index=classes, columns=classes)

    print("\nMatriz de confusão:")
    print(df_matriz)

    if pasta_saida and nome_conjunto.lower() == "teste":
        caminho_matriz = os.path.join(pasta_saida, "matriz_confusao_random_forest.csv")
        df_matriz.to_csv(caminho_matriz, sep=";", encoding="utf-8-sig")
        print(f"\nMatriz de confusão salva em: {caminho_matriz}")

        plt.figure(figsize=(14, 10))
        plt.imshow(df_matriz, aspect="auto")
        plt.title("Matriz de Confusão - Random Forest")
        plt.xlabel("Classe prevista")
        plt.ylabel("Classe real")
        plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=90)
        plt.yticks(ticks=np.arange(len(classes)), labels=classes)
        plt.colorbar(label="Quantidade")
        plt.tight_layout()

        caminho_figura = os.path.join(pasta_saida, "matriz_confusao_random_forest.png")
        plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figura da matriz de confusão salva em: {caminho_figura}")

    return y_pred, acuracia, df_matriz


def gerar_importancia_variaveis(modelo, pasta_saida: str) -> pd.DataFrame:
    rf = modelo.named_steps["classificador"]
    preprocessador = modelo.named_steps["preprocessador"]
    encoder = preprocessador.named_transformers_["categoricas"]

    nomes_categoricas = encoder.get_feature_names_out(VARIAVEIS_CATEGORICAS)
    nomes_variaveis = list(nomes_categoricas) + VARIAVEIS_NUMERICAS

    importancias = rf.feature_importances_

    df_importancias = pd.DataFrame(
        {
            "variavel": nomes_variaveis,
            "importancia": importancias,
        }
    ).sort_values(by="importancia", ascending=False)

    caminho_csv = os.path.join(pasta_saida, "importancia_variaveis_random_forest.csv")
    df_importancias.to_csv(caminho_csv, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nImportância das variáveis salva em: {caminho_csv}")

    top_variaveis = df_importancias.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(top_variaveis["variavel"], top_variaveis["importancia"])
    plt.gca().invert_yaxis()
    plt.title("20 variáveis mais importantes - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()

    caminho_figura = os.path.join(pasta_saida, "importancia_variaveis_random_forest.png")
    plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de importância das variáveis salvo em: {caminho_figura}")

    return df_importancias


def salvar_predicoes(X_test, y_test, y_pred, pasta_saida: str) -> None:
    df_resultados = X_test.copy()
    df_resultados["tipo_acidente_real"] = y_test.values
    df_resultados["tipo_acidente_previsto"] = y_pred

    caminho = os.path.join(pasta_saida, "resultados_random_forest_tipo_acidente.csv")
    df_resultados.to_csv(caminho, sep=";", index=False, encoding="utf-8-sig")

    print(f"\nResultados das predições salvos em: {caminho}")

def main():
    criar_pasta_saida(PASTA_SAIDA)

    print("=" * 70)
    print("RANDOM FOREST - CLASSIFICAÇÃO DO TIPO DE ACIDENTE")
    print("=" * 70)

    df = carregar_datasets_zip(CAMINHO_ZIP)
    print(f"\nDataset completo carregado: {df.shape[0]} registros e {df.shape[1]} colunas")

    df_limpo = limpar_dataset(df, PASTA_SAIDA)

    print("\nDistribuição da variável-alvo tipo_acidente:")
    print(df_limpo[VARIAVEL_ALVO].value_counts())
    gerar_grafico_distribuicao_alvo(df_limpo, PASTA_SAIDA)

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(df_limpo)

    modelo = criar_modelo_random_forest()
    modelo.fit(X_train, y_train)
    print("\nModelo Random Forest treinado com sucesso.")

    avaliar_modelo(modelo, X_val, y_val, "validação")

    y_test_pred, _, _ = avaliar_modelo(modelo, X_test, y_test, "teste", PASTA_SAIDA)

    gerar_importancia_variaveis(modelo, PASTA_SAIDA)

    salvar_predicoes(X_test, y_test, y_test_pred, PASTA_SAIDA)

    print("\nProcessamento concluído.")
    print(f"Arquivos gerados na pasta: {PASTA_SAIDA}")

if __name__ == "__main__":
    main()
