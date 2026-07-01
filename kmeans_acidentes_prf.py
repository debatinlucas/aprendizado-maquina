import os
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

CAMINHO_ZIP = "dataset.zip"
UF_FILTRO = "SC"
K_ESCOLHIDO = 5 # alinhado aos cinco perfis analíticos de resultado
PASTA_RESULTADOS = Path("resultados_kmeans_acidentes_prf")
PASTA_RESULTADOS.mkdir(exist_ok=True)

VARIAVEIS_CATEGORICAS = [
    "final_de_semana",
    "grupo_causa_acidente",
    "grupo_tipo_acidente",
    "acidente_noturno",
    "acidente_com_chuva",
    "curva_pista_simples",
    "acidente_fatal",
]

VARIAVEIS_NUMERICAS = [
    "br",
    "km",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
]

VARIAVEIS_MODELO = VARIAVEIS_CATEGORICAS + VARIAVEIS_NUMERICAS


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

def agrupar_causa_acidente(causa):
    causa = str(causa).strip().lower()

    if any(termo in causa for termo in [
        "falta de atenção",
        "ausência de reação",
        "reação tardia",
        "condutor",
        "pedestre",
        "sono",
        "mal súbito",
        "ingestão de álcool",
        "ingestão de substância",
        "desobediência",
        "não guardar distância",
        "ultrapassagem indevida",
        "conversão proibida",
        "transitar na contramão",
        "acessar a via sem observar",
        "entrada inopinada",
        "manobra",
        "mudança de faixa"
    ]):
        return "Fator humano"

    elif any(termo in causa for termo in [
        "velocidade incompatível",
        "ultrapassagem",
        "distância de segurança",
        "frear bruscamente",
        "retorno proibido"
    ]):
        return "Conduta de risco"

    elif any(termo in causa for termo in [
        "defeito na via",
        "sinalização",
        "acostamento",
        "interseção",
        "obstáculo",
        "obra",
        "pista em desnível",
        "pista esburacada",
        "curva acentuada",
        "declive",
        "via"
    ]):
        return "Infraestrutura viária"

    elif any(termo in causa for termo in [
        "chuva",
        "neblina",
        "fumaça",
        "fenômenos da natureza",
        "pista escorregadia",
        "animais",
        "animal",
        "restrição de visibilidade",
        "iluminação"
    ]):
        return "Condição ambiental"

    elif any(termo in causa for termo in [
        "defeito mecânico",
        "deficiência mecânica",
        "pneu",
        "carga",
        "mal acondicionada",
        "avaria",
        "sistema de freios",
        "sistema de iluminação",
        "veículo"
    ]):
        return "Fator veicular/carga"

    else:
        return "Outros"

def carregar_datasets_do_zip(caminho_zip: str) -> pd.DataFrame:
    if not os.path.exists(caminho_zip):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_zip}. "
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
                df_temp["arquivo_origem"] = nome_arquivo
                lista_datasets.append(df_temp)
    df = pd.concat(lista_datasets, ignore_index=True)
    df["grupo_tipo_acidente"] = df["tipo_acidente"].apply(agrupar_tipo_acidente)
    df["grupo_causa_acidente"] = df["causa_acidente"].apply(agrupar_causa_acidente)
    df["acidente_com_chuva"] = df["condicao_metereologica"].str.contains(
        "chuva", case=False, na=False
    ).astype(int)

    df["acidente_noturno"] = df["fase_dia"].str.contains(
        "noite|madrugada", case=False, na=False
    ).astype(int)

    df["final_de_semana"] = df["dia_semana"].isin(
        ["sábado", "domingo"]
    ).astype(int)

    df["curva_pista_simples"] = (
        df["tracado_via"].str.contains("curva", case=False, na=False) &
        df["tipo_pista"].str.contains("simples", case=False, na=False)
    ).astype(int)

    df["acidente_fatal"] = (df["mortos"] > 0).astype(int)

    df["tem_feridos"] = (
        (df["feridos_leves"] > 0) | (df["feridos_graves"] > 0)
    ).astype(int)
    return df

def converter_numericas(df: pd.DataFrame, colunas_numericas: list[str]) -> pd.DataFrame:
    df_convertido = df.copy()
    for coluna in colunas_numericas:
        df_convertido[coluna] = (
            df_convertido[coluna]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df_convertido[coluna] = pd.to_numeric(df_convertido[coluna], errors="coerce")
    return df_convertido

def gerar_grafico_barras(serie: pd.Series, titulo: str, xlabel: str, ylabel: str, arquivo_saida: Path) -> None:
    plt.figure(figsize=(10, 6))
    serie.plot(kind="bar")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()

def rotular_perfil_cluster(linha: pd.Series) -> str:
    perfil = []
    if linha.get("acidente_com_chuva", 0):
        perfil.append("acidentes associados à chuva")
    if linha.get("acidente_noturno", 0):
        perfil.append("acidentes noturnos")
    if linha.get("curva_pista_simples", 0):
        perfil.append("acidentes em curvas em pista simples")
    if linha.get("final_de_semana", 0):
        perfil.append("acidentes em finais de semana")
    if linha.get("acidente_fatal", 0):
        perfil.append("grupo com maior severidade")
    if not perfil:
        perfil.append("perfil geral de ocorrências")
    return "; ".join(perfil)

def main() -> None:
    print("=" * 80)
    print("K-Means - Agrupamento de acidentes de trânsito PRF 2017 a 2025")
    print("=" * 80)

    df = carregar_datasets_do_zip(CAMINHO_ZIP)
    print(f"\nDataset completo carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")

    df_sc = df[df["uf"] == UF_FILTRO].copy()
    print(f"Registros filtrados para {UF_FILTRO}: {df_sc.shape[0]}")

    colunas_saida = list(dict.fromkeys(VARIAVEIS_MODELO))
    df_modelo = df_sc[colunas_saida].copy()

    print("\nValores nulos antes do tratamento nas variáveis do modelo:")
    print(df_modelo[VARIAVEIS_MODELO].isnull().sum())

    df_modelo = converter_numericas(df_modelo, VARIAVEIS_NUMERICAS)

    quantidade_antes = df_modelo.shape[0]
    df_limpo = df_modelo.dropna(subset=VARIAVEIS_MODELO).copy()
    quantidade_depois = df_limpo.shape[0]

    print("\nValores nulos depois do tratamento nas variáveis do modelo:")
    print(df_limpo[VARIAVEIS_MODELO].isnull().sum())

    print(f"\nQuantidade antes da limpeza: {quantidade_antes}")
    print(f"Quantidade depois da limpeza: {quantidade_depois}")
    print(f"Registros removidos: {quantidade_antes - quantidade_depois}")

    caminho_dataset_limpo = PASTA_RESULTADOS / "dataset_sc_2017_2025_kmeans_limpo.csv"
    df_limpo.to_csv(caminho_dataset_limpo, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nDataset limpo salvo em: {caminho_dataset_limpo}")

    preprocessador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                VARIAVEIS_CATEGORICAS,
            ),
            (
                "numericas",
                StandardScaler(),
                VARIAVEIS_NUMERICAS,
            ),
        ]
    )

    X_processado = preprocessador.fit_transform(df_limpo[VARIAVEIS_MODELO])

    # --------------------------------------------------------
    # 3.6 Método do cotovelo e silhouette score
    # --------------------------------------------------------
    metricas_k = []
    valores_k = range(2, 11)

    print("\nCalculando métricas para escolha de K...")

    for k in valores_k:
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_processado)

        inercia = modelo_temp.inertia_

        # Para silhouette em datasets grandes, usar amostra para reduzir custo computacional.
        tamanho_amostra = min(10000, X_processado.shape[0])
        silhueta = silhouette_score(
            X_processado,
            labels_temp,
            sample_size=tamanho_amostra,
            random_state=42,
        )

        metricas_k.append({
            "k": k,
            "inercia": inercia,
            "silhouette_score": silhueta,
        })

        print(f"K={k} | Inércia={inercia:.2f} | Silhouette={silhueta:.4f}")

    df_metricas_k = pd.DataFrame(metricas_k)
    caminho_metricas = PASTA_RESULTADOS / "metricas_k_cotovelo_silhouette.csv"
    df_metricas_k.to_csv(caminho_metricas, sep=";", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.plot(df_metricas_k["k"], df_metricas_k["inercia"], marker="o")
    plt.title("Método do Cotovelo - K-Means")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inércia")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "grafico_cotovelo_kmeans.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_metricas_k["k"], df_metricas_k["silhouette_score"], marker="o")
    plt.title("Silhouette Score - K-Means")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "grafico_silhouette_kmeans.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 3.7 Treinar modelo final K-Means
    # --------------------------------------------------------
    print(f"\nTreinando K-Means final com K={K_ESCOLHIDO}...")

    modelo_kmeans = KMeans(
        n_clusters=K_ESCOLHIDO,
        random_state=42,
        n_init=10,
    )

    modelo_kmeans.fit(X_processado)

    clusters_todos = modelo_kmeans.predict(X_processado)

    df_com_clusters = df_limpo.copy()
    df_com_clusters["cluster"] = clusters_todos

    caminho_clusters = PASTA_RESULTADOS / "dataset_sc_2017_2025_com_clusters_kmeans.csv"
    df_com_clusters.to_csv(caminho_clusters, sep=";", index=False, encoding="utf-8-sig")
    print(f"Dataset com clusters salvo em: {caminho_clusters}")

    distribuicao_clusters = (
        df_com_clusters["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    distribuicao_clusters.columns = ["cluster", "quantidade"]
    distribuicao_clusters["percentual"] = (
        distribuicao_clusters["quantidade"] / distribuicao_clusters["quantidade"].sum() * 100
    ).round(2)

    caminho_dist = PASTA_RESULTADOS / "distribuicao_clusters.csv"
    distribuicao_clusters.to_csv(caminho_dist, sep=";", index=False, encoding="utf-8-sig")

    gerar_grafico_barras(
        df_com_clusters["cluster"].value_counts().sort_index(),
        "Quantidade de acidentes por cluster",
        "Cluster",
        "Quantidade de acidentes",
        PASTA_RESULTADOS / "grafico_quantidade_por_cluster.png",
    )

    perfil_numerico = df_com_clusters.groupby("cluster")[VARIAVEIS_NUMERICAS].agg(
        ["mean", "median", "sum", "min", "max"]
    )
    caminho_perfil_num = PASTA_RESULTADOS / "perfil_numerico_clusters.csv"
    perfil_numerico.to_csv(caminho_perfil_num, sep=";", encoding="utf-8-sig")

    linhas_moda = []
    for cluster, grupo in df_com_clusters.groupby("cluster"):
        linha = {"cluster": cluster}
        for var in VARIAVEIS_CATEGORICAS:
            moda = grupo[var].mode(dropna=True)
            linha[var] = moda.iloc[0] if not moda.empty else np.nan
        linhas_moda.append(linha)

    perfil_categorico = pd.DataFrame(linhas_moda).sort_values("cluster")
    caminho_perfil_cat = PASTA_RESULTADOS / "perfil_categorico_moda_clusters.csv"
    perfil_categorico.to_csv(caminho_perfil_cat, sep=";", index=False, encoding="utf-8-sig")

    perfil_sintetico = distribuicao_clusters.merge(perfil_categorico, on="cluster", how="left")

    medias_severidade = df_com_clusters.groupby("cluster")[[
        "pessoas", "mortos", "feridos_leves", "feridos_graves"
    ]].mean().reset_index()
    medias_severidade = medias_severidade.rename(columns={
        "pessoas": "pessoas_media",
        "mortos": "mortos_media",
        "feridos_leves": "feridos_leves_media",
        "feridos_graves": "feridos_graves_media",
    })

    somas_severidade = df_com_clusters.groupby("cluster")[[
        "pessoas", "mortos", "feridos_leves", "feridos_graves"
    ]].sum().reset_index()
    somas_severidade = somas_severidade.rename(columns={
        "pessoas": "pessoas_total",
        "mortos": "mortos_total",
        "feridos_leves": "feridos_leves_total",
        "feridos_graves": "feridos_graves_total",
    })

    perfil_sintetico = perfil_sintetico.merge(medias_severidade, on="cluster", how="left")
    perfil_sintetico = perfil_sintetico.merge(somas_severidade, on="cluster", how="left")
    perfil_sintetico["rotulo_sugerido"] = perfil_sintetico.apply(rotular_perfil_cluster, axis=1)

    caminho_perfil_sint = PASTA_RESULTADOS / "perfil_sintetico_clusters.csv"
    perfil_sintetico.to_csv(caminho_perfil_sint, sep=";", index=False, encoding="utf-8-sig")

    print("\nDistribuição dos clusters:")
    print(distribuicao_clusters)

    print("\nPerfil sintético dos clusters:")
    print(perfil_sintetico)


    df_ind = df_com_clusters.copy()
    
    indicadores_cluster = df_ind.groupby("cluster")[[
        "acidente_com_chuva",
        "acidente_noturno",
        "curva_pista_simples",
        "final_de_semana",
        "acidente_fatal",
    ]].mean().reset_index()

    for col in indicadores_cluster.columns:
        if col != "cluster":
            indicadores_cluster[col] = (indicadores_cluster[col] * 100).round(2)

    caminho_indicadores = PASTA_RESULTADOS / "indicadores_perfis_desejados_por_cluster.csv"
    indicadores_cluster.to_csv(caminho_indicadores, sep=";", index=False, encoding="utf-8-sig")

    print("\nIndicadores dos perfis desejados por cluster (%):")
    print(indicadores_cluster)

    print("\nGerando visualização 2D dos clusters...")

    redutor = TruncatedSVD(n_components=2, random_state=42)
    X_2d = redutor.fit_transform(X_processado)

    df_2d = pd.DataFrame({
        "componente_1": X_2d[:, 0],
        "componente_2": X_2d[:, 1],
        "cluster": clusters_todos,
    })
    df_2d.to_csv(PASTA_RESULTADOS / "componentes_2d_clusters.csv", sep=";", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 7))
    for cluster in sorted(df_2d["cluster"].unique()):
        dados_cluster = df_2d[df_2d["cluster"] == cluster]
        plt.scatter(
            dados_cluster["componente_1"],
            dados_cluster["componente_2"],
            label=f"Cluster {cluster}",
            alpha=0.5,
            s=10,
        )

    plt.title("Visualização 2D dos clusters - K-Means")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "grafico_clusters_2d.png", dpi=300)
    plt.close()

    caminho_relatorio = PASTA_RESULTADOS / "resumo_execucao_kmeans.txt"
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("K-Means - Agrupamento de acidentes de trânsito PRF 2017 a 2025\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"UF analisada: {UF_FILTRO}\n")
        f.write(f"Registros antes da limpeza: {quantidade_antes}\n")
        f.write(f"Registros depois da limpeza: {quantidade_depois}\n")
        f.write(f"Registros removidos: {quantidade_antes - quantidade_depois}\n")
        f.write(f"K escolhido: {K_ESCOLHIDO}\n\n")
        f.write("Variáveis categóricas codificadas por One-Hot Encoding:\n")
        f.write(", ".join(VARIAVEIS_CATEGORICAS) + "\n\n")
        f.write("Variáveis numéricas padronizadas com StandardScaler:\n")
        f.write(", ".join(VARIAVEIS_NUMERICAS) + "\n\n")
        f.write("Distribuição dos clusters:\n")
        f.write(distribuicao_clusters.to_string(index=False))
        f.write("\n\nPerfil sintético dos clusters:\n")
        f.write(perfil_sintetico.to_string(index=False))
        f.write("\n\nIndicadores dos perfis desejados por cluster (%):\n")
        f.write(indicadores_cluster.to_string(index=False))

    print("\nArquivos gerados na pasta:", PASTA_RESULTADOS)
    print("Execução concluída com sucesso.")

if __name__ == "__main__":
    main()
