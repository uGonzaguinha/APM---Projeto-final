

# 📚 Trabalho Final — Aprendizado de Máquina

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen.svg)
![Licença](https://img.shields.io/badge/Licença-Acadêmica-orange.svg)

---

## 🎯 Descrição Geral

Este projeto consiste no desenvolvimento e implementação de um **pipeline completo de Ciência de Dados** voltado à **classificação de avaliações (ratings)** de jogos disponíveis na plataforma Steam. O trabalho contempla todas as etapas do fluxo de Machine Learning supervisionado, desde a coleta e preparação dos dados até a avaliação de desempenho dos modelos preditivos, com foco na reprodutibilidade e na validação rigorosa.

---

## 👨‍💻 Autores

* Gustavo Gonzaga dos Santos
* Thiago Gonzaga dos Santos

📅 **Data de Entrega:** 25 de junho de 2025

---

## 🧩 Objetivos

* Construir um pipeline modular e robusto de aprendizado de máquina.
* Explorar, limpar e transformar um conjunto de dados real.
* Comparar diferentes algoritmos classificadores por meio de métricas consistentes.
* Validar o modelo final com técnicas de validação cruzada estratificada.
* Documentar todas as etapas com rigor técnico e clareza acadêmica.

---

## ⚙️ Estrutura do Pipeline

### Etapas Principais:

1. **Coleta de Dados** – Importação do conjunto de dados *Steam Games*.
2. **Análise Exploratória (EDA)** – Visualizações e estatísticas descritivas.
3. **Limpeza de Dados** – Tratamento de inconsistências e dados ausentes.
4. **Conversão de Tipos** – Adequação dos formatos para manipulação.
5. **Tratamento de Outliers** – Utilização do método IQR.
6. **Codificação de Variáveis Categóricas** – Aplicação de LabelEncoder.
7. **Escalonamento** – Normalização com RobustScaler.
8. **Seleção de Atributos** – Técnicas RFE e engenharia de atributos.
9. **Balanceamento de Classes** – Aplicação do algoritmo SMOTE.
10. **Redução de Dimensionalidade** – Implementação opcional de PCA.
11. **Divisão do Conjunto de Dados** – Separação estratificada em treino/teste.
12. **Treinamento de Modelos** – Implementação de quatro algoritmos.
13. **Avaliação de Desempenho** – Uso de métricas padronizadas.
14. **Ajuste de Hiperparâmetros** – Grid Search com validação cruzada.
15. **Testes Finais** – Avaliação global com métricas e relatórios.

---

## 🤖 Algoritmos Utilizados

* **Random Forest**
* **Gradient Boosting**
* **Multi-Layer Perceptron (MLP)**
* **Regressão Logística**

---

## 📊 Métricas de Avaliação

* **Accuracy (Acurácia)**
* **Precision (Precisão)**
* **Recall (Revocação)**
* **F1-Score**
* **Classification Report**
* **Confusion Matrix (Matriz de Confusão)**

---

## 💻 Requisitos Técnicos

### Sistema e Ambiente

* **Versão do Python:** 3.8 ou superior (testado com 3.12)
* **Memória RAM:** 4 GB (mínimo), 8 GB (recomendado)
* **Sistema Operacional:** Windows, Linux ou macOS
* **Espaço em Disco:** 500 MB disponíveis

### Instalação das Dependências

Instale as bibliotecas utilizando o seguinte comando:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

* **Nome:** `games.csv`
* **Origem:** Plataforma Steam (dados públicos)
* **Volume:** \~50.000 registros
* **Tamanho:** \~12 MB
* **Encoding:** UTF-8

### Estrutura das Variáveis

| Coluna            | Tipo    | Descrição                             |
| ----------------- | ------- | ------------------------------------- |
| app\_id           | int     | Identificador único do jogo           |
| title             | string  | Nome do jogo                          |
| date\_release     | date    | Data de lançamento                    |
| win/mac/linux/... | boolean | Compatibilidade com plataformas       |
| rating            | string  | **Variável target** (avaliação final) |
| positive\_ratio   | int     | Porcentagem de avaliações positivas   |
| user\_reviews     | int     | Número total de avaliações            |
| price\_final      | float   | Preço atual em USD                    |
| price\_original   | float   | Preço original em USD                 |
| discount          | float   | Porcentagem de desconto aplicada      |

### Categorias da Variável `rating`

* Overwhelmingly Positive
* Very Positive
* Positive
* Mostly Positive
* Mixed
* Mostly Negative
* Negative
* Very Negative
* Overwhelmingly Negative

---

## ✅ Resultados Obtidos

### Desempenho do Melhor Modelo

* **Modelo:** Random Forest
* **Accuracy:** 100.00%
* **F1-Score:** 100.00%
* **Validação Cruzada:** 100.00% ± 0.00

### Métricas por Classe

* Precision: 1.0000
* Recall: 1.0000
* F1-Score: 1.0000

### Tempo Médio de Execução

* Pipeline completo: 5 a 10 minutos
* Treinamento: 2 a 3 minutos
* Avaliação: 1 a 2 minutos

---

## 🧪 Validações Científicas

### Rigor Acadêmico

* Reprodutibilidade garantida por uso de `random_state`
* Validação com múltiplas métricas e diferentes cenários
* Comparação justa com pré-processamento padronizado
* Documentação completa de cada etapa

### Prevenção de Erros

* Prevenção de **data leakage**
* Controle de **overfitting**
* Estrutura modular e reutilizável

---

## 📈 Visualizações Geradas

* Distribuição das classes
* Matriz de correlação
* Matriz de confusão
* Curvas de desempenho (Precision/Recall)
* Gráficos comparativos entre modelos

---

## 🛠️ Diagnóstico de Problemas

| Erro Comum                         | Possível Solução                                           |
| ---------------------------------- | ---------------------------------------------------------- |
| `ModuleNotFoundError`              | Verifique se as dependências foram instaladas corretamente |
| Dataset ausente                    | Confirme o caminho e o nome do arquivo `games.csv`         |
| Erro de memória                    | Reduza o tamanho do dataset ou aumente a RAM               |
| Incompatibilidade de versão Python | Utilize Python 3.8 ou superior                             |
| Problema de encoding no Windows    | Certifique-se de que o CSV está em UTF-8                   |

---

## 🧱 Estrutura do Projeto

```
├── data/
│   └── games.csv
├── src/
│   ├── eda.py
│   ├── preprocessamento.py
│   ├── modelagem.py
│   ├── avaliacao.py
│   └── utils.py
├── verificar_instalacao.py
├── requirements.txt
└── README.md
```

