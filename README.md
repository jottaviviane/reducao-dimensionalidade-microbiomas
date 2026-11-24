# Redução de dimensionalidade linear e não linear em comunidades microbianas

Comparação entre **PCA**, **NMDS** e **t-SNE** na representação de duas comunidades microbianas de ambientes extremos: biofilmes de painéis fotovoltaicos (metagenômica shotgun) e solo de canga do Quadrilátero Ferrífero, MG (metataxonômica do gene 16S rRNA).

Trabalho final da disciplina **Inteligência Artificial Aplicada à Bioinformática** (2026/2), Programa de Pós-Graduação em Bioinformática, Universidade Federal de Minas Gerais (UFMG).

**Autora:** Viviane Faria Morais Jotta

---

## O que o script faz

1. Lê as duas matrizes de abundância (amostras × gêneros).
2. Remove gêneros ausentes em todas as amostras e converte as abundâncias em proporções.
3. Aplica a transformação de Hellinger (entrada da PCA) e calcula a dissimilaridade de Bray-Curtis (entrada do NMDS e do t-SNE).
4. Projeta as amostras em duas dimensões com PCA, NMDS e t-SNE.
5. Avalia cada projeção quanto à:
   - **preservação da estrutura global:** correlação de Spearman entre as dissimilaridades originais e as distâncias na projeção;
   - **preservação da estrutura local:** *trustworthiness*;
   - **separação dos grupos amostrais:** coeficiente de *silhouette* e acurácia de um classificador k-NN (k = 1) com validação *leave-one-out*.
6. Gera uma figura com as seis projeções e uma tabela com todas as métricas.

## Estrutura do repositório

```
reducao-dimensionalidade-microbiomas/
├── README.md
├── requirements.txt
├── aprendizado_nao_supervisionado.py   # script da análise
├── dados/
│   ├── biofilme_generos.csv            # 9 amostras × 1099 gêneros (contagem de reads)
│   └── canga_generos.csv               # 18 amostras × 184 gêneros (abundância relativa, %)
└── resultados/
    ├── Figura1_PCA_NMDS_tSNE.png       # projeções dos três métodos nos dois conjuntos
    └── metricas_comparacao.csv         # métricas de avaliação
```

## Dados

| Conjunto | Amostras | Grupos | Técnica | Origem |
|---|---|---|---|---|
| Biofilme de painel fotovoltaico | 9 | D1, D2, D3 (3 subamostras cada) | Metagenômica shotgun | Jotta et al. (2024) |
| Solo de canga | 18 | Transectos L1, L2, L3 (6 amostras cada) | Metataxonômica (16S rRNA, V3–V4) | Jotta et al. (2026) |

Formato dos arquivos: separador `;`, ponto como decimal, primeira coluna (`NAME`) com o nome das amostras e demais colunas com os gêneros. No biofilme, foram mantidas apenas as classificações em nível de gênero (removidas as categorias não classificadas, não mapeadas e sem gênero definido no NCBI).

## Como executar

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
python aprendizado_nao_supervisionado.py
```

Os resultados são salvos na pasta `resultados/`. A figura também é aberta em uma janela; o script termina quando ela é fechada.

## Parâmetros principais

| Parâmetro | Biofilme | Canga |
|---|---|---|
| Perplexidade do t-SNE (≈ n/3) | 3 | 6 |
| k da *trustworthiness* (≈ n/3) | 3 | 6 |
| k do k-NN | 1 | 1 |
| Semente aleatória | 42 | 42 |

Os parâmetros podem ser alterados no bloco `conjuntos`, no início do script.

## Ambiente utilizado

Python 3.10.11, pandas 2.3.3, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9.

## Referências dos dados

- JOTTA, V. F. M. et al. Taxonomic and functional characterization of biofilms from a photovoltaic panel reveals high genetic and metabolic complexity of the communities. **Journal of Applied Microbiology**, v. 135, n. 9, lxae231, 2024. DOI: [10.1093/jambio/lxae231](https://doi.org/10.1093/jambio/lxae231).
- JOTTA, V. F. M. et al. Bacterial diversity in ferruginous duricrust (canga) and the physicochemical variables affecting their prevalence, distribution and predicted metabolic pathways. **Antonie van Leeuwenhoek**, v. 119, n. 5, 2026. DOI: [10.1007/s10482-026-02315-9](https://doi.org/10.1007/s10482-026-02315-9).
