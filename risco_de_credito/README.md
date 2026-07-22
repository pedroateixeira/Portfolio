# Previsão de Inadimplência por Atraso no Pagamento

Projeto de consultoria em ciência de dados que constrói um modelo preditivo para estimar a **probabilidade de inadimplência** de cobranças mensais feitas a clientes de uma empresa fictícia contratante.

Inadimplência é definida como um pagamento realizado com **5 ou mais dias de atraso** em relação à data de vencimento. O objetivo do contratante é receber a probabilidade contínua de inadimplência (não uma classificação binária), para direcionar ações de cobrança proativas e reduzir o índice de calote.

## Bases de dados

Quatro bases relacionadas por `ID_CLIENTE` e `SAFRA_REF` (período de referência da cobrança):

| Arquivo | Conteúdo |
|---|---|
| `data/base_cadastral.csv` | Dados cadastrais únicos por cliente (segmento, porte, DDD, e-mail, etc.) |
| `data/base_info.csv` | Dados mensais (renda do mês anterior, número de funcionários) |
| `data/base_pagamentos_desenvolvimento.csv` | Histórico de cobranças com datas de vencimento e pagamento, usado para treino/validação/teste |
| `data/base_pagamentos_teste.csv` | Cobranças mais recentes, sem informação de pagamento — base de produção para gerar as previsões finais |

## Abordagem

1. **Tratamento de dados e EDA**: checagem de unicidade das chaves, consistência temporal, tratamento de nulos (com investigação de causa antes de imputar) e engenharia de features a partir do histórico de pagamentos do cliente (ex.: `TAXA_ATRASO_CLIENTE`, `QTD_FATURAS_ANTERIORES`, `HISTORICO_INADIMPLENCIA`).
2. **Split treino/validação/teste**: via `StratifiedGroupKFold`, agrupando todas as cobranças de um mesmo cliente em um único conjunto, para evitar vazamento de dados (cerca de 91% dos clientes de teste também apareciam no treino).
3. **Modelagem**: comparação entre Regressão Logística (baseline), LightGBM, XGBoost e um Ensemble (média LightGBM + XGBoost), avaliados na validação por ROC-AUC, Gini, PR-AUC, KS e Brier Score.
4. **Calibração e limiar de decisão**: verificação da curva de calibração das probabilidades e escolha do limiar via F2-score (prioriza recall sobre precisão), já que deixar passar um futuro inadimplente tende a custar mais do que uma cobrança feita à toa.
5. **Avaliação final**: aplicação única do modelo escolhido no conjunto de teste (holdout, nunca usado antes) e geração das previsões para a base de produção.

## Resultado

O modelo final foi o **Ensemble (LightGBM + XGBoost)**, com desempenho no conjunto de teste (holdout):

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.9178 |
| Gini | 0.8355 |
| KS | 0.7077 |
| PR-AUC | 0.5126 |
| Brier Score | 0.0829 |

Com o limiar de decisão de **0.3596** (escolhido via F2), o modelo sinaliza **17,82%** das cobranças da base de produção para ação proativa de cobrança.

As variáveis mais relevantes foram o comportamento histórico do próprio cliente (`TAXA_ATRASO_CLIENTE`, `QTD_FATURAS_ANTERIORES`, `HISTORICO_INADIMPLENCIA`), tornando o modelo mais confiável para clientes já existentes na carteira do que para clientes totalmente novos.

## Arquivos

- `notebook.ipynb`: notebook completo com EDA, modelagem, avaliação e conclusões de negócio.
- `dataset_final.csv`: probabilidades de inadimplência previstas para a base de produção (`ID_CLIENTE`, `SAFRA_REF`, `PROBABILIDADE_INADIMPLENCIA`).
- `requirements.txt`: dependências Python do projeto.

## Tecnologias

Python 3.13, pandas, numpy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn.
