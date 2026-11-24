##########################################################################################
# SCRIPT PARA APRENDIZADO NAO SUPERVISIONADO DA DISCIPLINA DE INTELIGENCIA ARTIFICIAL APLICADA A BIOINFORMATICA
# AUTORA: Viviane Faria Morais Jotta
# DATA: 28/10/2026 (ultima atualizacao: 24/11/2026)
##########################################################################################

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
import matplotlib.pyplot as plt

####################################
# PASTAS DO PROJETO (caminhos relativos a pasta deste script)
####################################

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA, "dados")
PASTA_RESULTADOS = os.path.join(PASTA, "resultados")
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

####################################
# CONFIGURACAO DOS CONJUNTOS DE DADOS
####################################

# Funcoes que definem o grupo de cada amostra a partir do nome
def grupo_biofilme(amostra):
    # "D1-2" -> "D1"
    return amostra.split("-")[0]

def grupo_canga(amostra):
    # S1-S6 -> L1, S7-S12 -> L2, S13-S18 -> L3
    numero = int(amostra.replace("S", ""))
    if numero <= 6:
        return "L1"
    elif numero <= 12:
        return "L2"
    else:
        return "L3"


# Para cada conjunto: caminho do arquivo, perplexidade do t-SNE,
# numero de vizinhos (k) da trustworthiness, funcao de grupos e
# numero de vizinhos do classificador k-NN.
# A perplexidade precisa ser menor que o numero de amostras;
# criterio adotado: aproximadamente n/3 em cada conjunto.
conjuntos = {
    "Biofilme": {
        "arquivo": os.path.join(PASTA_DADOS, "biofilme_generos.csv"),
        "perplexidade": 3,   # ~ n/3 (n = 9)
        "k_vizinhos": 3,     # ~ n/3 (n = 9)
        "grupo": grupo_biofilme,
        "k_knn": 1,
    },
    "Canga": {
        "arquivo": os.path.join(PASTA_DADOS, "canga_generos.csv"),
        "perplexidade": 6,   # ~ n/3 (n = 18)
        "k_vizinhos": 6,     # ~ n/3 (n = 18)
        "grupo": grupo_canga,
        "k_knn": 1,
    },
}


def acuracia_knn_loo(X, grupos, k_knn, metric="euclidean"):
    # Leave-one-out: cada amostra e classificada pelos vizinhos mais proximos
    # entre as demais; retorna a proporcao de acertos
    knn = KNeighborsClassifier(n_neighbors=k_knn, metric=metric)
    return cross_val_score(knn, X, grupos, cv=LeaveOneOut()).mean()


def analisar_conjunto(nome, arquivo, perplexidade, k_vizinhos, funcao_grupo, k_knn):

    print("\n" + "#" * 60)
    print("CONJUNTO:", nome)
    print("#" * 60)

    ####################################
    # PASSO 1 - IMPORTAR DADOS
    ####################################

    df = pd.read_csv(arquivo, sep=";", index_col=0)

    print(df.head())
    print(df.shape)
    print(df.isnull().sum().sum())
    print(df.dtypes.value_counts())

    ####################################
    # PASSO 2 - PREPARAR A MATRIZ
    ####################################

    # Garantir que todas as colunas sejam numéricas
    df = df.apply(pd.to_numeric, errors="raise")

    # Verificar valores negativos
    if (df < 0).any().any():
        raise ValueError("Existem abundâncias negativas.")

    # Verificar valores ausentes
    if df.isnull().any().any():
        raise ValueError("Existem valores ausentes.")

    # Remover gêneros que não aparecem em nenhuma amostra
    df = df.loc[:, df.sum(axis=0) > 0]

    # Verificar amostras com abundância total zero
    if (df.sum(axis=1) == 0).any():
        raise ValueError("Existem amostras sem abundância.")

    print(df.shape)

    ##########################################################
    # PASSO 3 - TRANSFORMAR, NORMALIZAR E CALCULAR DISTÂNCIAS
    ##########################################################

    # Converter abundâncias em proporções
    relative_abundance = df.div(df.sum(axis=1), axis=0)

    # Aplicar transformação de Hellinger
    hellinger = np.sqrt(relative_abundance)

    # Calcular distância Bray-Curtis
    bray_curtis = squareform(
        pdist(relative_abundance, metric="braycurtis")
    )

    ############################
    # PASSO 4 - APLICANDO O PCA
    ############################

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(hellinger)

    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=df.index)

    print("RESULTADOS DO PCA")
    print("Variância explicada:", pca.explained_variance_ratio_)
    print("Variância acumulada:", pca.explained_variance_ratio_.sum())

    ############################
    # PASSO 5 - APLICANDO O NMDS
    ############################

    # metric=False -> MDS nao metrico (NMDS): preserva a ordem das dissimilaridades
    nmds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=False,
        random_state=42,
        n_init=10,
        max_iter=1000,
        normalized_stress="auto"
    )

    nmds_result = nmds.fit_transform(bray_curtis)

    nmds_df = pd.DataFrame(nmds_result, columns=["NMDS1", "NMDS2"], index=df.index)

    print("RESULTADOS DO NMDS")
    print("Stress do NMDS:", nmds.stress_)

    #############################
    # PASSO 6 - APLICANDO O t-SNE
    #############################

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexidade,
        learning_rate="auto",
        max_iter=1000,
        random_state=42
    )

    tsne_result = tsne.fit_transform(bray_curtis)

    tsne_df = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2"], index=df.index)

    print("RESULTADOS DO t-SNE")
    print("KL Divergence:", tsne.kl_divergence_)

    ####################################
    # PASSO 7 - GUARDAR AS PROJECOES PARA A FIGURA FINAL
    ####################################
    # A figura combinada (2 conjuntos x 3 metodos) e montada no final do script.

    paineis = [
        (pca_df, "PC1", "PC2", "PCA",
         f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
         f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"),
        (nmds_df, "NMDS1", "NMDS2", f"NMDS (stress = {nmds.stress_:.3f})",
         "NMDS1", "NMDS2"),
        (tsne_df, "tSNE1", "tSNE2", "t-SNE",
         "t-SNE 1", "t-SNE 2"),
    ]

    #################################################################
    # PASSO 8 - METRICAS QUANTITATIVAS DE COMPARACAO ENTRE OS METODOS
    #################################################################

    # Distâncias originais Bray-Curtis
    dist_original = pdist(relative_abundance, metric="braycurtis")

    resultados = {"PCA": pca_result, "NMDS": nmds_result, "t-SNE": tsne_result}

    # Medida de ajuste propria de cada metodo
    ajuste = {
        "PCA": ("Variância explicada (PC1+PC2)", pca.explained_variance_ratio_.sum()),
        "NMDS": ("Stress", nmds.stress_),
        "t-SNE": ("Divergência KL", tsne.kl_divergence_),
    }

    # Grupo de cada amostra
    grupos = [funcao_grupo(a) for a in df.index]
    print("Grupos:", dict(zip(df.index, grupos)))

    # Referencia: separacao dos grupos na propria matriz Bray-Curtis (sem reduzir)
    linhas = [{
        "Conjunto": nome,
        "N_amostras": df.shape[0],
        "N_generos": df.shape[1],
        "Metodo": "Bray-Curtis original (referência)",
        "Spearman_Bray": np.nan,
        "Trustworthiness": np.nan,
        "k_vizinhos": k_vizinhos,
        "Silhouette": silhouette_score(bray_curtis, grupos, metric="precomputed"),
        "Acuracia_kNN_LOO": acuracia_knn_loo(bray_curtis, grupos, k_knn, metric="precomputed"),
        "k_knn": k_knn,
        "Medida_ajuste": "",
        "Valor_ajuste": np.nan,
    }]

    for metodo, resultado in resultados.items():

        # Correlação de Spearman entre distâncias originais e na projeção
        spearman = spearmanr(dist_original, pdist(resultado, metric="euclidean")).statistic

        # Trustworthiness usando a mesma matriz de referência
        trust = trustworthiness(
            bray_curtis, resultado,
            metric="precomputed",
            n_neighbors=k_vizinhos
        )

        linhas.append({
            "Conjunto": nome,
            "N_amostras": df.shape[0],
            "N_generos": df.shape[1],
            "Metodo": metodo,
            "Spearman_Bray": spearman,
            "Trustworthiness": trust,
            "k_vizinhos": k_vizinhos,
            # Separacao dos grupos na projecao 2D
            "Silhouette": silhouette_score(resultado, grupos),
            "Acuracia_kNN_LOO": acuracia_knn_loo(resultado, grupos, k_knn),
            "k_knn": k_knn,
            "Medida_ajuste": ajuste[metodo][0],
            "Valor_ajuste": ajuste[metodo][1],
        })

    return pd.DataFrame(linhas), paineis, grupos


####################################
# EXECUTAR PARA OS DOIS CONJUNTOS
####################################

tabelas = []
figura = {}   # projecoes de cada conjunto, para a figura final
for nome, cfg in conjuntos.items():
    tabela, paineis, grupos = analisar_conjunto(
        nome, cfg["arquivo"], cfg["perplexidade"], cfg["k_vizinhos"],
        cfg["grupo"], cfg["k_knn"]
    )
    tabelas.append(tabela)
    figura[nome] = (paineis, grupos)

metricas = pd.concat(tabelas, ignore_index=True)

print("\nTABELA FINAL")
print(metricas.round(4).to_string(index=False))

# Salvar tabela (ponto como decimal, para nao repetir o problema da virgula no Excel)
metricas.round(4).to_csv(os.path.join(PASTA_RESULTADOS, "metricas_comparacao.csv"), sep=";", index=False)


####################################
# FIGURA FINAL: 2 LINHAS (CONJUNTOS) x 3 COLUNAS (METODOS)
####################################

# Cores e marcadores por grupo (paleta segura para daltonismo; o marcador
# diferente garante a leitura tambem em impressao preto e branco)
cores = ["#2a78d6", "#eb6834", "#1baf7a"]
marcadores = ["o", "s", "^"]

fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
letras = iter("ABCDEF")

for linha, (nome, (paineis, grupos)) in enumerate(figura.items()):

    grupos = pd.Series(grupos, index=paineis[0][0].index)
    nomes_grupos = sorted(grupos.unique())

    for coluna, (data, x, y, titulo, xlabel, ylabel) in enumerate(paineis):

        ax = axes[linha, coluna]

        for g, nome_grupo in enumerate(nomes_grupos):
            sel = data[grupos == nome_grupo]
            ax.scatter(sel[x], sel[y], s=55, color=cores[g], marker=marcadores[g],
                       edgecolor="white", linewidth=0.8, label=nome_grupo, zorder=3)

        # Nomes das amostras; pontos quase sobrepostos recebem rotulos
        # empilhados para nao ficarem um em cima do outro
        faixa = max(data[x].max() - data[x].min(), data[y].max() - data[y].min())
        posicoes = []
        for amostra, row in data.iterrows():
            vizinhos = sum(
                1 for (px, py) in posicoes
                if abs(px - row[x]) < 0.07 * faixa and abs(py - row[y]) < 0.05 * faixa
            )
            ax.annotate(amostra, (row[x], row[y]), xytext=(4, 3 + 8 * vizinhos),
                        textcoords="offset points", fontsize=7, color="#444444")
            posicoes.append((row[x], row[y]))

        ax.set_title(f"{next(letras)}   {titulo}", fontsize=11, loc="left")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.margins(0.15)

    # Uma legenda por linha, a direita do ultimo painel
    axes[linha, 2].legend(title=nome, fontsize=9, title_fontsize=10, frameon=False,
                          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.savefig(os.path.join(PASTA_RESULTADOS, "Figura1_PCA_NMDS_tSNE.png"), dpi=300, bbox_inches="tight")
plt.show()

