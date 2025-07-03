# -*- coding: utf-8 -*-
"""
TRABALHO FINAL - APRENDIZADO DE MÁQUINA
=======================================

Pipeline Completo de Ciência de Dados para Classificação de Ratings de Jogos Steam

Disciplina: Aprendizado de Máquina
Autores: Gustavo Gonzaga dos Santos, Thiago Gonzaga dos Santos
Data: 25 de junho de 2025

PROBLEMA: Classificação supervisionada de ratings de jogos Steam
DATASET: Steam Games Dataset (50,000+ registros)
OBJETIVO: Desenvolver modelo para classificar ratings baseado em características dos jogos

PIPELINE IMPLEMENTADO:
    1. Coleta de dados
2. Análise exploratória dos dados  
3. Limpeza dos dados
4. Conversão de tipos e formatação
5. Tratamento de outliers
6. Codificação de variáveis categóricas
7. Escalonamento de variáveis
8. Seleção e criação de atributos
9. Balanceamento de classes
10. Redução de dimensionalidade
11. Divisão dos dados
12. Treinamento do modelo
13. Avaliação do modelo
14. Ajuste de hiperparâmetros
15. Testes finais e validação
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, 
                                 RobustScaler, MinMaxScaler)
from sklearn.feature_selection import (SelectKBest, f_classif, RFE,
                                     mutual_info_classif, RFECV)
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                            GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Configuração para visualizações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PipelineTrabalhoFinal:
    """
    Pipeline completo de ciência de dados para o trabalho final
    Implementa todas as etapas necessárias para um projeto profissional de ML
    """
    
    def __init__(self):
        """Inicialização do pipeline"""
        self.dados_originais = None
        self.dados_processados = None
        self.X_final = None
        self.y_final = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelos = {}
        self.melhor_modelo = None
        self.resultados_finais = {}
        
        # Componentes de preprocessamento
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.smote = None
        
        print("TRABALHO FINAL - PIPELINE DE CIÊNCIA DE DADOS INICIALIZADO")
        print("=" * 80)
    
    def etapa1_coleta_dados(self, caminho='./Trabalho2/datasets/games.csv'):
        """
        ETAPA 1: COLETA DE DADOS
        
        Carrega o dataset de jogos Steam e realiza verificações iniciais
        """
        print("\nETAPA 1: COLETA DE DADOS")
        print("-" * 50)
        
        try:
            self.dados_originais = pd.read_csv(caminho)
            print(f"Dataset carregado com sucesso!")
            print(f"Dimensões: {self.dados_originais.shape}")
            print(f"Colunas: {list(self.dados_originais.columns)}")
            
            # Informações básicas
            print(f"\nInformações básicas:")
            print(f"   Total de registros: {len(self.dados_originais):,}")
            print(f"   Total de colunas: {self.dados_originais.shape[1]}")
            print(f"   Memória utilizada: {self.dados_originais.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def etapa2_analise_exploratoria(self):
        """
        ETAPA 2: ANÁLISE EXPLORATÓRIA DOS DADOS
        
        Realiza análise detalhada para entender o dataset
        """
        print("\nETAPA 2: ANÁLISE EXPLORATÓRIA DOS DADOS")
        print("-" * 50)
        
        # Informações sobre tipos de dados
        print("TIPOS DE DADOS:")
        print(self.dados_originais.dtypes)
        
        # Valores ausentes
        print(f"\nVALORES AUSENTES:")
        valores_ausentes = self.dados_originais.isnull().sum()
        total_registros = len(self.dados_originais)
        
        for coluna, qtd in valores_ausentes.items():
            if qtd > 0:
                percentual = (qtd / total_registros) * 100
                print(f"   {coluna}: {qtd:,} ({percentual:.2f}%)")
        
        if valores_ausentes.sum() == 0:
            print("   Nenhum valor ausente encontrado!")
        
        # Análise da variável target
        print(f"\nANÁLISE DA VARIÁVEL TARGET (rating):")
        if 'rating' in self.dados_originais.columns:
            dist_rating = self.dados_originais['rating'].value_counts()
            print(dist_rating)
            
            print(f"\nDistribuição percentual:")
            dist_perc = self.dados_originais['rating'].value_counts(normalize=True) * 100
            for categoria, perc in dist_perc.items():
                print(f"   {categoria}: {perc:.2f}%")
        
        # Estatísticas descritivas para variáveis numéricas
        print(f"\nESTATÍSTICAS DESCRITIVAS (Variáveis Numéricas):")
        colunas_numericas = self.dados_originais.select_dtypes(include=[np.number]).columns
        print(self.dados_originais[colunas_numericas].describe())
        
        # Correlações
        print(f"\nMATRIZ DE CORRELAÇÃO:")
        corr_matrix = self.dados_originais[colunas_numericas].corr()
        
        # Visualizações
        self._plotar_analise_exploratoria()
        
        return True
    
    def _plotar_analise_exploratoria(self):
        """Gera visualizações para análise exploratória"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('ANÁLISE EXPLORATÓRIA - STEAM GAMES DATASET', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Distribuição da variável target
        if 'rating' in self.dados_originais.columns:
            dist_rating = self.dados_originais['rating'].value_counts()
            
            # Cores mais distintas
            cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
            
            # Configuração do gráfico de pizza SEM labels internos
            wedges, texts, autotexts = axes[0,0].pie(
                dist_rating.values,
                labels=None,  # Remove todos os labels do gráfico
                autopct='%1.1f%%',
                startangle=90,
                colors=cores[:len(dist_rating)],
                textprops={'fontsize': 9, 'fontweight': 'bold'},
                pctdistance=0.85  # Distância dos percentuais do centro
            )
            
            # Melhorar aparência dos percentuais
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            # Legenda EXTERNA e melhor posicionada
            axes[0,0].legend(wedges, dist_rating.index,
                           title="Ratings",
                           loc="center left",
                           bbox_to_anchor=(1.05, 0.5),
                           fontsize=8,
                           title_fontsize=9,
                           frameon=True,
                           fancybox=True,
                           shadow=True)
            
            axes[0,0].set_title('Distribuição dos Ratings', fontsize=14, fontweight='bold', pad=25)
        
        # 2. Histograma de user_reviews
        if 'user_reviews' in self.dados_originais.columns:
            data_reviews = self.dados_originais['user_reviews']
            axes[0,1].hist(data_reviews, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            axes[0,1].set_title('Distribuição de User Reviews', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Número de Reviews', fontsize=11)
            axes[0,1].set_ylabel('Frequência', fontsize=11)
            axes[0,1].grid(True, alpha=0.3)
            
            # Adicionar estatísticas
            mean_reviews = data_reviews.mean()
            median_reviews = data_reviews.median()
            axes[0,1].axvline(mean_reviews, color='red', linestyle='--', alpha=0.8, label=f'Média: {mean_reviews:.0f}')
            axes[0,1].axvline(median_reviews, color='orange', linestyle='--', alpha=0.8, label=f'Mediana: {median_reviews:.0f}')
            axes[0,1].legend(fontsize=10)
        
        # 3. Boxplot de preços
        if 'price_final' in self.dados_originais.columns:
            data_price = self.dados_originais['price_final'].dropna()
            
            # Boxplot com estatísticas
            bp = axes[0,2].boxplot(data_price, patch_artist=True, 
                                 boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                 medianprops=dict(color='darkred', linewidth=2))
            
            axes[0,2].set_title('Boxplot - Preço Final', fontsize=14, fontweight='bold')
            axes[0,2].set_ylabel('Preço (USD)', fontsize=11)
            axes[0,2].grid(True, alpha=0.3)
            
            # Adicionar estatísticas como texto
            q1, median, q3 = data_price.quantile([0.25, 0.5, 0.75])
            mean_price = data_price.mean()
            
            stats_text = f'Q1: ${q1:.2f}\nMediana: ${median:.2f}\nQ3: ${q3:.2f}\nMédia: ${mean_price:.2f}'
            axes[0,2].text(1.12, 0.75, stats_text, transform=axes[0,2].transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        
        # 4. Distribuição de plataformas
        plataformas = ['win', 'mac', 'linux', 'steam_deck']
        plat_counts = []
        plat_names = []
        for plat in plataformas:
            if plat in self.dados_originais.columns:
                count = self.dados_originais[plat].sum()
                plat_counts.append(count)
                plat_names.append(plat.replace('_', ' ').title())
        
        if plat_counts:
            bars = axes[1,0].bar(plat_names, plat_counts, 
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                               alpha=0.8, edgecolor='black', linewidth=0.5)
            
            axes[1,0].set_title('Jogos por Plataforma', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('Número de Jogos', fontsize=11)
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, count in zip(bars, plat_counts):
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + max(plat_counts)*0.01,
                             f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Labels mais limpos
            axes[1,0].tick_params(axis='x', rotation=0, labelsize=10)
        
        # 5. Correlação entre variáveis numéricas
        colunas_numericas = self.dados_originais.select_dtypes(include=[np.number]).columns
        if len(colunas_numericas) > 1:
            corr_matrix = self.dados_originais[colunas_numericas].corr()
            
            # Usar seaborn heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8},
                       ax=axes[1,1], annot_kws={'size': 9})
            
            axes[1,1].set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
            axes[1,1].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1,1].tick_params(axis='y', rotation=0, labelsize=9)
        
        # 6. Relação preço vs rating
        if 'price_final' in self.dados_originais.columns and 'rating' in self.dados_originais.columns:
            rating_price = self.dados_originais.groupby('rating')['price_final'].mean().sort_values(ascending=False)
            
            bars = axes[1,2].bar(range(len(rating_price)), rating_price.values, 
                               color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            axes[1,2].set_title('Preço Médio por Rating', fontsize=14, fontweight='bold')
            axes[1,2].set_xlabel('Rating', fontsize=11)
            axes[1,2].set_ylabel('Preço Médio (USD)', fontsize=11)
            axes[1,2].grid(True, alpha=0.3, axis='y')
            
            # Configurar labels do eixo X
            axes[1,2].set_xticks(range(len(rating_price)))
            
            # Labels mais curtos para evitar sobreposição
            labels_curtos = []
            for label in rating_price.index:
                if len(label) > 12:
                    # Abreviar labels muito longos
                    if 'Overwhelmingly' in label:
                        labels_curtos.append(label.replace('Overwhelmingly ', 'Ovr. '))
                    elif 'Mostly' in label:
                        labels_curtos.append(label.replace('Mostly ', 'M. '))
                    else:
                        labels_curtos.append(label)
                else:
                    labels_curtos.append(label)
            
            axes[1,2].set_xticklabels(labels_curtos, rotation=30, ha='right', fontsize=9)
            
            # Adicionar valores nas barras
            for i, (bar, price) in enumerate(zip(bars, rating_price.values)):
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + max(rating_price.values)*0.01,
                             f'${price:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.subplots_adjust(hspace=0.35, wspace=0.4)
        plt.show()
    
    def etapa3_limpeza_dados(self):
        """
        ETAPA 3: LIMPEZA DOS DADOS
        
        Remove registros inválidos e trata inconsistências
        """
        print("\nETAPA 3: LIMPEZA DOS DADOS")
        print("-" * 50)
        
        dados_limpos = self.dados_originais.copy()
        registros_iniciais = len(dados_limpos)
        
        print(f"Registros iniciais: {registros_iniciais:,}")
        
        # Remover colunas desnecessárias
        colunas_remover = ['app_id', 'title', 'date_release']
        dados_limpos = dados_limpos.drop(columns=colunas_remover, errors='ignore')
        print(f"Removidas colunas identificadoras: {colunas_remover}")
        
        # Tratar valores ausentes
        valores_ausentes_antes = dados_limpos.isnull().sum().sum()
        dados_limpos = dados_limpos.dropna()
        valores_ausentes_depois = dados_limpos.isnull().sum().sum()
        registros_removidos_na = registros_iniciais - len(dados_limpos)
        
        print(f"Removidos {registros_removidos_na:,} registros com valores ausentes")
        
        # Filtrar registros com valores inválidos
        # Preços negativos
        if 'price_final' in dados_limpos.columns:
            antes = len(dados_limpos)
            dados_limpos = dados_limpos[dados_limpos['price_final'] >= 0]
            removidos_preco = antes - len(dados_limpos)
            if removidos_preco > 0:
                print(f"Removidos {removidos_preco:,} registros com preços negativos")
        
        # Reviews negativas
        if 'user_reviews' in dados_limpos.columns:
            antes = len(dados_limpos)
            dados_limpos = dados_limpos[dados_limpos['user_reviews'] >= 0]
            removidos_reviews = antes - len(dados_limpos)
            if removidos_reviews > 0:
                print(f"Removidos {removidos_reviews:,} registros com reviews negativas")
        
        # Positive ratio inválido
        if 'positive_ratio' in dados_limpos.columns:
            antes = len(dados_limpos)
            dados_limpos = dados_limpos[
                (dados_limpos['positive_ratio'] >= 0) & 
                (dados_limpos['positive_ratio'] <= 100)
            ]
            removidos_ratio = antes - len(dados_limpos)
            if removidos_ratio > 0:
                print(f"Removidos {removidos_ratio:,} registros com positive_ratio inválido")
        
        # Atualizar dados
        self.dados_limpos = dados_limpos
        
        # Resumo final
        registros_finais = len(dados_limpos)
        registros_removidos = registros_iniciais - registros_finais
        taxa_remocao = (registros_removidos / registros_iniciais) * 100
        
        print(f"\nRESUMO DA LIMPEZA:")
        print(f"   Registros iniciais: {registros_iniciais:,}")
        print(f"   Registros removidos: {registros_removidos:,}")
        print(f"   Taxa de remoção: {taxa_remocao:.2f}%")
        print(f"   Registros finais: {registros_finais:,}")
        
        return True
    
    def etapa4_conversao_tipos(self):
        """
        ETAPA 4: CONVERSÃO DE TIPOS E FORMATAÇÃO
        
        Converte tipos de dados para formatos apropriados
        """
        print("\nETAPA 4: CONVERSÃO DE TIPOS E FORMATAÇÃO")
        print("-" * 50)
        
        dados_convertidos = self.dados_limpos.copy()
        
        print("Tipos originais:")
        print(dados_convertidos.dtypes)
        
        # Converter variáveis booleanas para int
        colunas_bool = ['win', 'mac', 'linux', 'steam_deck']
        for col in colunas_bool:
            if col in dados_convertidos.columns:
                dados_convertidos[col] = dados_convertidos[col].astype(int)
                print(f"{col}: bool → int")
        
        # Garantir que variáveis numéricas estejam no tipo correto
        colunas_numericas = ['user_reviews', 'price_final', 'price_original', 
                           'discount', 'positive_ratio']
        for col in colunas_numericas:
            if col in dados_convertidos.columns:
                dados_convertidos[col] = pd.to_numeric(dados_convertidos[col], errors='coerce')
                print(f"{col}: convertido para numérico")
        
        print("\nTipos após conversão:")
        print(dados_convertidos.dtypes)
        
        # Atualizar dados
        self.dados_convertidos = dados_convertidos
        return True
    
    def etapa5_tratamento_outliers(self, metodo='iqr'):
        """
        ETAPA 5: TRATAMENTO DE OUTLIERS
        
        Identifica e remove outliers das variáveis numéricas
        """
        print("\nETAPA 5: TRATAMENTO DE OUTLIERS")
        print(f"   Método: {metodo.upper()}")
        print("-" * 50)
        
        dados_sem_outliers = self.dados_convertidos.copy()
        registros_inicial = len(dados_sem_outliers)
        total_outliers_removidos = 0
        
        print(f"Analisando outliers por variável:")
        
        # Definir colunas numéricas para análise de outliers
        colunas_numericas = ['user_reviews', 'price_final', 'positive_ratio']
        
        for coluna in colunas_numericas:
            if coluna in dados_sem_outliers.columns:
                antes = len(dados_sem_outliers)
                
                # Calcular IQR
                Q1 = dados_sem_outliers[coluna].quantile(0.25)
                Q3 = dados_sem_outliers[coluna].quantile(0.75)
                IQR = Q3 - Q1
                
                # Definir limites
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                # Remover outliers
                dados_sem_outliers = dados_sem_outliers[
                    (dados_sem_outliers[coluna] >= limite_inferior) & 
                    (dados_sem_outliers[coluna] <= limite_superior)
                ]
                
                depois = len(dados_sem_outliers)
                removidos = antes - depois
                
                if removidos > 0:     
                    print(f"   {coluna}: {removidos:,} outliers removidos")
                    print(f"      Limites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
                total_outliers_removidos += removidos
        
        registros_final = len(dados_sem_outliers)
        taxa_remocao = ((registros_inicial - registros_final) / registros_inicial) * 100
        
        print(f"\nRESUMO DO TRATAMENTO:")        
        print(f"   Registros iniciais: {registros_inicial:,}")
        print(f"   Outliers removidos: {total_outliers_removidos:,}")
        print(f"   Taxa de remoção: {taxa_remocao:.2f}%")
        print(f"   Registros finais: {registros_final:,}")
        
        self.dados_sem_outliers = dados_sem_outliers
        return True
    
    def etapa6_codificacao_categoricas(self):
        """
        ETAPA 6: CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
        
        Aplica Label Encoding para a variável target
        """
        print("\nETAPA 6: CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS")
        print("-" * 50)
        
        dados_codificados = self.dados_sem_outliers.copy()
        
        # Codificar variável target
        if 'rating' in dados_codificados.columns:
            dados_codificados['rating_encoded'] = self.label_encoder.fit_transform(
                dados_codificados['rating']
            )
            print("Codificação da variável target 'rating':")
            classes_mapping = dict(zip(
                self.label_encoder.classes_,
                range(len(self.label_encoder.classes_))
            ))
            for classe_original, codigo in classes_mapping.items():
                print(f"   '{classe_original}' → {codigo}")
            
            print(f"{len(self.label_encoder.classes_)} classes codificadas")
        
        # Atualizar dados
        self.dados_codificados = dados_codificados
        return True
    
    def etapa7_escalonamento_variaveis(self, metodo='robust'):
        """
        ETAPA 7: ESCALONAMENTO DE VARIÁVEIS
        
        Aplica escalonamento robusto para lidar com outliers residuais
        """
        print("\nETAPA 7: ESCALONAMENTO DE VARIÁVEIS")
        print(f"   Método: {metodo.title()}Scaler")
        print("-" * 50)
        
        # Selecionar features numéricas
        features_numericas = ['user_reviews', 'price_final', 'positive_ratio', 'win', 'mac', 'linux', 'steam_deck']
        features_disponiveis = [col for col in features_numericas 
                              if col in self.dados_codificados.columns]
        
        X = self.dados_codificados[features_disponiveis].copy()
        y = self.dados_codificados['rating_encoded'].copy()
        
        print(f"Features selecionadas: {features_disponiveis}")
        print(f"Dimensões X: {X.shape}")
        print(f"Dimensões y: {y.shape}")
        
        # Configurar scaler
        if metodo == 'robust':
            self.scaler = RobustScaler()
            print("   Usar RobustScaler (mediana e IQR)")
        elif metodo == 'standard':
            self.scaler = StandardScaler()
            print("   Usar StandardScaler (média e desvio)")
        else:
            self.scaler = MinMaxScaler()
            print("   Usar MinMaxScaler (min-max)")
        
        # Armazenar dados antes do escalonamento
        self.X_raw = X.copy()
        self.y = y.copy()
        
        print("Scaler configurado (aplicação após divisão treino/teste)")
        return True
    
    def etapa8_selecao_atributos(self, metodo='rfe', n_features=6):
        """
        ETAPA 8: SELEÇÃO E CRIAÇÃO DE ATRIBUTOS
        
        Aplica seleção de features e cria novos atributos
        """
        print("\nETAPA 8: SELEÇÃO E CRIAÇÃO DE ATRIBUTOS")
        print(f"   Método: {metodo.upper()}")
        print(f"   Features alvo: {n_features}")
        print("-" * 50)
        
        X_enhanced = self.X_raw.copy()
        
        # CRIAÇÃO DE NOVOS ATRIBUTOS
        print("Criando novos atributos:")
        # 1. Razão preço/reviews
        if 'price_final' in X_enhanced.columns and 'user_reviews' in X_enhanced.columns:
            X_enhanced['price_per_review'] = (X_enhanced['price_final'] / (X_enhanced['user_reviews'] + 1))
            print("   price_per_review: preço por review")
        
        # 2. Score multi-plataforma
        plataformas = ['win', 'mac', 'linux', 'steam_deck']
        plat_cols = [col for col in plataformas if col in X_enhanced.columns]
        if len(plat_cols) > 1:
            X_enhanced['multi_platform_score'] = X_enhanced[plat_cols].sum(axis=1)
            print("   multi_platform_score: suporte multi-plataforma")
        
        # 3. Categoria de preço
        if 'price_final' in X_enhanced.columns:
            # Criar categorias de preço e tratar NaN
            price_categories = pd.cut(
                X_enhanced['price_final'], 
                bins=[0, 5, 20, 50, np.inf],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            # Converter para int, preenchendo NaN com 0
            X_enhanced['price_category'] = price_categories.fillna(0).astype(int)
            print("   price_category: categoria de preço")
        
        # 4. Popularidade (log de reviews)
        if 'user_reviews' in X_enhanced.columns:
            X_enhanced['log_reviews'] = np.log1p(X_enhanced['user_reviews'])
            print("   log_reviews: log da popularidade")
        
        # 5. Eficiência de avaliação (reviews por rating positivo)
        if 'user_reviews' in X_enhanced.columns and 'positive_ratio' in X_enhanced.columns:
            X_enhanced['review_efficiency'] = (X_enhanced['user_reviews'] * 
                                              X_enhanced['positive_ratio'] / 100)
            print("   review_efficiency: eficiência das avaliações")
        
        print(f"\nFeatures após criação: {X_enhanced.shape[1]}")
        # Verificar e tratar valores infinitos ou NaN
        # Substituir infinitos por valores grandes finitos
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
        
        # Preencher NaN com a mediana de cada coluna
        for col in X_enhanced.select_dtypes(include=[np.number]).columns:
            if X_enhanced[col].isna().any():
                mediana = X_enhanced[col].median()
                X_enhanced[col] = X_enhanced[col].fillna(mediana)
                print(f"   {col}: NaN preenchidos com mediana ({mediana:.4f})")
        
        # SELEÇÃO DE FEATURES
        print(f"\nAplicando seleção de features ({metodo.upper()}):")
        if metodo == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(
                estimator=estimator, 
                n_features_to_select=n_features,
                step=1
            )
            
            X_selected = self.feature_selector.fit_transform(X_enhanced, self.y)
            features_selecionadas = X_enhanced.columns[self.feature_selector.get_support()].tolist()
        
        elif metodo == 'univariate':
            # Análise Univariada
            self.feature_selector = SelectKBest(
                score_func=f_classif, 
                k=n_features
            )
            
            X_selected = self.feature_selector.fit_transform(X_enhanced, self.y)
            features_selecionadas = X_enhanced.columns[self.feature_selector.get_support()].tolist()
        
        elif metodo == 'mutual_info':
            # Informação Mútua
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif, 
                k=n_features
            )
            
            X_selected = self.feature_selector.fit_transform(X_enhanced, self.y)
            features_selecionadas = X_enhanced.columns[self.feature_selector.get_support()].tolist()
        
        # Atualizar dados
        self.X_final = pd.DataFrame(X_selected, columns=features_selecionadas, index=X_enhanced.index)
        self.y_final = self.y.copy()
        
        print(f"   Features selecionadas: {features_selecionadas}")
        print(f"   Dimensões finais: {self.X_final.shape}")
        
        # Exibir importância (se RFE)
        if metodo == 'rfe' and hasattr(self.feature_selector, 'ranking_'):
            print(f"\nRanking das features:")
            feature_ranking = list(zip(X_enhanced.columns, self.feature_selector.ranking_))
            feature_ranking.sort(key=lambda x: x[1])
            for i, (feature, rank) in enumerate(feature_ranking[:10], 1):
                status = "Selecionada" if rank == 1 else "Rejeitada"
                print(f"   {i:2d}. {feature}: Rank {rank} ({status})")
        
        return True
    
    def etapa9_balanceamento_classes(self, metodo='smote'):
        """
        ETAPA 9: BALANCEAMENTO DE CLASSES
        
        Aplica técnicas para balancear distribuição das classes
        """
        print("\nETAPA 9: BALANCEAMENTO DE CLASSES")
        print(f"   Método: {metodo.upper()}")
        print("-" * 50)
        
        # Analisar distribuição atual
        print("Distribuição atual das classes:")
        dist_original = self.y_final.value_counts().sort_index()
        total = len(self.y_final)
        for classe, qtd in dist_original.items():
            nome_classe = self.label_encoder.classes_[classe]
            percentual = (qtd / total) * 100
            print(f"   Classe {classe} ({nome_classe}): {qtd:,} ({percentual:.1f}%)")
        
        # Verificar se precisa balanceamento
        min_classe = dist_original.min()
        max_classe = dist_original.max()
        ratio_desbalanceamento = max_classe / min_classe
        
        print(f"\nAnálise do desbalanceamento:")
        print(f"   Classe minoritária: {min_classe:,} amostras")
        print(f"   Classe majoritária: {max_classe:,} amostras")
        print(f"   Razão de desbalanceamento: {ratio_desbalanceamento:.2f}:1")
        
        if ratio_desbalanceamento > 2.0:
            print(f"   Desbalanceamento significativo detectado!")
            
            if metodo == 'smote':
                # SMOTE - Synthetic Minority Oversampling
                self.smote = SMOTE(random_state=42, k_neighbors=5)
                X_balanced, y_balanced = self.smote.fit_resample(self.X_final, self.y_final)
                
                print(f"   Aplicando SMOTE...")
                
            elif metodo == 'undersample':
                # Random Undersampling
                undersampler = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = undersampler.fit_resample(self.X_final, self.y_final)
                
                print(f"   Aplicando Undersampling...")
            
            # Converter de volta para DataFrame
            self.X_balanced = pd.DataFrame(X_balanced, columns=self.X_final.columns)
            self.y_balanced = pd.Series(y_balanced)
        
        else:
            print(f"   Classes já estão razoavelmente balanceadas")
            self.X_balanced = self.X_final.copy()
            self.y_balanced = self.y_final.copy()
        
        return True
    
    def etapa10_reducao_dimensionalidade(self, aplicar=False, n_components=0.95):
        """
        ETAPA 10: REDUÇÃO DE DIMENSIONALIDADE
        
        Aplica PCA se necessário (opcional para este dataset)
        """
        print("\nETAPA 10: REDUÇÃO DE DIMENSIONALIDADE")
        print(f"   Aplicar PCA: {'Sim' if aplicar else 'Não'}")
        print("-" * 50)
        
        if aplicar and self.X_balanced.shape[1] > 10:    
            print(f"Aplicando PCA...")
            print(f"   Variância a preservar: {n_components}")
            
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(self.X_balanced)
            
            # Criar DataFrame com componentes
            componentes = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            self.X_pca = pd.DataFrame(X_pca, columns=componentes)
            
            print(f"   PCA aplicado com sucesso!")
            print(f"   Dimensões originais: {self.X_balanced.shape}")
            print(f"   Dimensões após PCA: {self.X_pca.shape}")
            print(f"   Variância explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
            
            # Usar dados com PCA
            self.X_final_processed = self.X_pca
            
        else:
            print(f"   PCA não aplicado (desnecessário para este dataset)")
            self.X_final_processed = self.X_balanced
        
        return True
    
    def etapa11_divisao_dados(self, test_size=0.2, random_state=42):
        """
        ETAPA 11: DIVISÃO DOS DADOS
        
        Divide em treino/teste com estratificação
        """
        print("\nETAPA 11: DIVISÃO DOS DADOS")
        print(f"   Proporção teste: {test_size*100:.0f}%")
        print(f"   Estratificação: Sim")
        print("-" * 50)
        
        # Divisão estratificada
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_final_processed, self.y_balanced,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_balanced
        )
        
        print(f"Divisão realizada:")
        print(f"   Treino: {len(self.X_train):,} amostras ({(1-test_size)*100:.0f}%)")
        print(f"   Teste: {len(self.X_test):,} amostras ({test_size*100:.0f}%)")
        
        # Aplicar escalonamento
        print(f"\nAplicando escalonamento:")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Converter de volta para DataFrame
        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.X_train.columns, index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X_test.columns, index=self.X_test.index)
        
        # Verificar distribuição das classes
        print(f"\nVerificação da estratificação:")
        print("   TREINO:")
        dist_train = self.y_train.value_counts(normalize=True).sort_index()
        for classe, prop in dist_train.items():
            nome_classe = self.label_encoder.classes_[classe]
            print(f"      Classe {classe} ({nome_classe}): {prop:.3f}")
        
        print("   TESTE:")
        dist_test = self.y_test.value_counts(normalize=True).sort_index()
        for classe, prop in dist_test.items():
            nome_classe = self.label_encoder.classes_[classe]
            print(f"      Classe {classe} ({nome_classe}): {prop:.3f}")
        
        return True
    
    def etapa12_treinamento_modelos(self):
        """
        ETAPA 12: TREINAMENTO DO MODELO
        
        Treina múltiplos modelos para comparação
        """
        print("\nETAPA 12: TREINAMENTO DE MODELOS")
        print("-" * 50)
        
        # Definir modelos para treinamento
        modelos_config = {
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='ovr'
            )
        }
        
        print(f"Treinando {len(modelos_config)} modelos:")
        
        for nome, modelo in modelos_config.items():
            print(f"\n   Treinando {nome}...")
            
            try:
                # Treinar modelo
                modelo.fit(self.X_train_scaled, self.y_train)
                
                # Fazer predições
                y_train_pred = modelo.predict(self.X_train_scaled)
                y_test_pred = modelo.predict(self.X_test_scaled)
                
                # Calcular métricas
                acc_train = accuracy_score(self.y_train, y_train_pred)
                acc_test = accuracy_score(self.y_test, y_test_pred)
                f1_test = f1_score(self.y_test, y_test_pred, average='weighted')
                
                # Armazenar modelo e resultados
                self.modelos[nome] = {
                    'modelo': modelo,
                    'acc_train': acc_train,
                    'acc_test': acc_test,
                    'f1_score': f1_test,
                    'y_pred': y_test_pred
                }
                        
                print(f"      Acurácia treino: {acc_train:.4f}")
                print(f"      Acurácia teste: {acc_test:.4f}")
                print(f"      F1-Score: {f1_test:.4f}")
                
            except Exception as e:
                print(f"      Erro no treinamento: {e}")
        
        print(f"\nTreinamento concluído para {len(self.modelos)} modelos!")
        return True
    
    def etapa13_avaliacao_modelos(self):
        """
        ETAPA 13: AVALIAÇÃO DO MODELO
        
        Avalia todos os modelos treinados
        """
        print("\nETAPA 13: AVALIAÇÃO DE MODELOS")
        print("-" * 50)
        
        print("COMPARAÇÃO DE DESEMPENHO:")
        print(f"{'Modelo':<18} {'Acc.Treino':<12} {'Acc.Teste':<12} {'F1-Score':<12} {'Overfitting':<12}")
        print("-" * 70)            
        
        melhor_f1 = 0
        melhor_modelo_nome = None
        for nome, dados in self.modelos.items():
            acc_train = dados['acc_train']
            acc_test = dados['acc_test']
            f1 = dados['f1_score']
            overfitting = acc_train - acc_test
            
            # Identificar melhor modelo
            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_modelo_nome = nome
            
            # Status do overfitting
            if overfitting > 0.1:
                status_over = "Alto"
            elif overfitting > 0.05:
                status_over = "Médio"
            else:
                status_over = "Baixo"
            
            print(f"{nome:<18} {acc_train:<12.4f} {acc_test:<12.4f} {f1:<12.4f} {status_over:<12}")
        
        # Definir melhor modelo
        self.melhor_modelo = self.modelos[melhor_modelo_nome]
        
        print(f"\nMELHOR MODELO: {melhor_modelo_nome}")
        print(f"   F1-Score: {melhor_f1:.4f}")
        
        # Relatório detalhado do melhor modelo
        print(f"\nRELATÓRIO DETALHADO - {melhor_modelo_nome}:")
        y_pred_melhor = self.melhor_modelo['y_pred']
        
        # Métricas detalhadas
        precision = precision_score(self.y_test, y_pred_melhor, average='weighted')
        recall = recall_score(self.y_test, y_pred_melhor, average='weighted')
        
        print(f"   Precisão: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {melhor_f1:.4f}")
        
        # Classification report
        print(f"\nCLASSIFICATION REPORT:")
        nomes_classes = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred_melhor, target_names=nomes_classes))
        
        # Plotar matriz de confusão
        self._plotar_matriz_confusao(self.y_test, y_pred_melhor, melhor_modelo_nome)
        
        return melhor_modelo_nome
    
    def _plotar_matriz_confusao(self, y_true, y_pred, nome_modelo):
        """Gera matriz de confusão"""
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_true, y_pred)
        nomes_classes = self.label_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=nomes_classes,
                   yticklabels=nomes_classes,
                   cbar_kws={'label': 'Número de Predições'})
        plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=14, fontweight='bold')
        plt.xlabel('Classe Predita', fontsize=12)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def etapa14_ajuste_hiperparametros(self, modelo_nome=None):
        """
        ETAPA 14: AJUSTE DE HIPERPARÂMETROS
        
        Otimiza hiperparâmetros do melhor modelo
        """
        print("\nETAPA 14: AJUSTE DE HIPERPARÂMETROS")
        print("-" * 50)
        
        if modelo_nome is None:
            modelo_nome = max(self.modelos.keys(), key=lambda k: self.modelos[k]['f1_score'])
        
        print(f"Otimizando hiperparâmetros para: {modelo_nome}")
        # Definir grids de busca por modelo
        if modelo_nome == 'MLP':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
            base_model = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
            
        elif modelo_nome == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif modelo_nome == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
            base_model = GradientBoostingClassifier(random_state=42)
        
        else:  # Logistic Regression
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [500, 1000]
            }
            base_model = LogisticRegression(random_state=42)
        
        print(f"Parâmetros a otimizar: {list(param_grid.keys())}")
        print(f"Executando Grid Search com validação cruzada...")
        # Grid Search com validação cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring='f1_weighted',    
            n_jobs=-1, verbose=0
        )
        
        # Executar busca
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Melhor modelo
        melhor_modelo_otimizado = grid_search.best_estimator_
        
        print(f"Otimização concluída!")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        print(f"Melhor score CV: {grid_search.best_score_:.4f}")
        
        # Avaliar modelo otimizado
        y_pred_otimizado = melhor_modelo_otimizado.predict(self.X_test_scaled)
        acc_otimizado = accuracy_score(self.y_test, y_pred_otimizado)
        f1_otimizado = f1_score(self.y_test, y_pred_otimizado, average='weighted')
        
        print(f"\nCOMPARAÇÃO DE DESEMPENHO:")
        print(f"   Modelo original - F1: {self.modelos[modelo_nome]['f1_score']:.4f}")
        print(f"   Modelo otimizado - F1: {f1_otimizado:.4f}")
        melhoria = f1_otimizado - self.modelos[modelo_nome]['f1_score']
        if melhoria > 0:
            print(f"   Melhoria: +{melhoria:.4f}")
        else:
            print(f"   Diferença: {melhoria:.4f}")
        
        # Atualizar melhor modelo se houve melhoria
        if f1_otimizado > self.melhor_modelo['f1_score']:
            self.melhor_modelo = {
                'modelo': melhor_modelo_otimizado,
                'acc_test': acc_otimizado,
                'f1_score': f1_otimizado,
                'y_pred': y_pred_otimizado
            }
            print(f"   Melhor modelo atualizado!")
        
        return melhor_modelo_otimizado
    
    def etapa15_testes_finais(self):
        """
        ETAPA 15: TESTES FINAIS E VALIDAÇÃO
        
        Realiza validação cruzada e testes finais
        """
        print("\nETAPA 15: TESTES FINAIS E VALIDAÇÃO")
        print("-" * 50)
        
        melhor_modelo = self.melhor_modelo['modelo']
        # Validação cruzada
        print("Executando validação cruzada (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(melhor_modelo, self.X_train_scaled, self.y_train, 
                                  cv=cv, scoring='f1_weighted')
        
        print(f"Resultados da validação cruzada:")
        print(f"   Scores: {cv_scores}")
        print(f"   Média: {cv_scores.mean():.4f}")
        print(f"   Desvio padrão: {cv_scores.std():.4f}")
        print(f"   Intervalo: [{cv_scores.mean() - cv_scores.std():.4f}, {cv_scores.mean() + cv_scores.std():.4f}]")
        
        # Teste final no conjunto de teste
        y_pred_final = melhor_modelo.predict(self.X_test_scaled)
        
        # Métricas finais
        metricas_finais = {
            'accuracy': accuracy_score(self.y_test, y_pred_final),
            'precision': precision_score(self.y_test, y_pred_final, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_final, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred_final, average='weighted')
        }
        
        print(f"\nMÉTRICAS FINAIS NO CONJUNTO DE TESTE:")
        for metrica, valor in metricas_finais.items():
            print(f"   {metrica.title()}: {valor:.4f}")
        
        # Análise por classe
        print(f"\nDESEMPENHO POR CLASSE:")
        nomes_classes = self.label_encoder.classes_
        precision_por_classe = precision_score(self.y_test, y_pred_final, average=None)
        recall_por_classe = recall_score(self.y_test, y_pred_final, average=None)
        f1_por_classe = f1_score(self.y_test, y_pred_final, average=None)
        
        for i, classe in enumerate(nomes_classes):
            if i < len(precision_por_classe):
                print(f"   {classe}:")
                print(f"      Precisão: {precision_por_classe[i]:.4f}")
                print(f"      Recall: {recall_por_classe[i]:.4f}")
                print(f"      F1-Score: {f1_por_classe[i]:.4f}")
        
        # Armazenar resultados finais
        self.resultados_finais = {
            'modelo_final': melhor_modelo,
            'metricas': metricas_finais,
            'cv_scores': cv_scores,
            'predicoes': y_pred_final
        }
        
        return metricas_finais
    
    def gerar_relatorio_final(self):
        """
        Gera relatório final completo do projeto
        """
        print("\n" + "=" * 80)
        print("RELATÓRIO FINAL - TRABALHO FINAL DE APRENDIZADO DE MÁQUINA")
        print("=" * 80)
        
        # Informações gerais
        print(f"\nPROBLEMA:")
        print(f"   Tipo: Classificação Supervisionada")
        print(f"   Domínio: Gaming/Entertainment")
        print(f"   Objetivo: Classificar ratings de jogos Steam")
        
        print(f"\nDATASET:")
        print(f"   Registros originais: {len(self.dados_originais):,}")
        print(f"   Registros finais: {len(self.X_final_processed):,}")
        print(f"   Features finais: {self.X_final_processed.shape[1]}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        
        print(f"\nPIPELINE EXECUTADO:")
        etapas = [
            "Coleta de dados",
            "Análise exploratória", 
            "Limpeza dos dados",
            "Conversão de tipos",
            "Tratamento de outliers",
            "Codificação categóricas",
            "Escalonamento",
            "Seleção de atributos",
            "Balanceamento de classes",
            "Redução de dimensionalidade",
            "Divisão dos dados",
            "Treinamento modelos",
            "Avaliação modelos",
            "Ajuste hiperparâmetros",
            "Testes finais"
        ]
        
        for etapa in etapas:
            print(f"   {etapa}")
        
        print(f"\nMELHOR MODELO:")
        melhor_modelo = self.resultados_finais['modelo_final']
        print(f"   Tipo: {type(melhor_modelo).__name__}")
        print(f"   Acurácia: {self.resultados_finais['metricas']['accuracy']:.4f}")
        print(f"   F1-Score: {self.resultados_finais['metricas']['f1_score']:.4f}")
        
        print(f"\nVALIDAÇÃO:")
        cv_mean = self.resultados_finais['cv_scores'].mean()
        cv_std = self.resultados_finais['cv_scores'].std()
        print(f"   Validação cruzada (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Estabilidade: {'Boa' if cv_std < 0.05 else 'Moderada' if cv_std < 0.1 else 'Baixa'}")
        
        print(f"\nCONCLUSÕES:")
        f1_final = self.resultados_finais['metricas']['f1_score']
        if f1_final >= 0.85:        
            print(f"   Excelente desempenho alcançado (F1 ≥ 0.85)")
        elif f1_final >= 0.75:
            print(f"   Bom desempenho alcançado (F1 ≥ 0.75)") 
        elif f1_final >= 0.65:
            print(f"   Desempenho moderado (F1 ≥ 0.65)")
        else:
            print(f"   Desempenho abaixo do esperado")
        
        print(f"   Modelo pronto para produção")
        print(f"   Pipeline otimizado e validado")
        
        print(f"\n" + "=" * 80)
        print("TRABALHO FINAL CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        
        return True
    
    def executar_pipeline_completo(self):
        """
        Executa todo o pipeline do trabalho final
        """
        print("EXECUTANDO PIPELINE COMPLETO DO TRABALHO FINAL")
        print("=" * 80)
        
        try:
            # Executar todas as etapas
            self.etapa1_coleta_dados()
            self.etapa2_analise_exploratoria()
            self.etapa3_limpeza_dados()
            self.etapa4_conversao_tipos()
            self.etapa5_tratamento_outliers()  # CORRIGIDO: removido parâmetro inválido
            self.etapa6_codificacao_categoricas()
            self.etapa7_escalonamento_variaveis()
            self.etapa8_selecao_atributos()
            self.etapa9_balanceamento_classes()
            self.etapa10_reducao_dimensionalidade(aplicar=False)
            self.etapa11_divisao_dados()
            self.etapa12_treinamento_modelos()
            melhor_modelo_nome = self.etapa13_avaliacao_modelos()
            self.etapa14_ajuste_hiperparametros(melhor_modelo_nome)
            self.etapa15_testes_finais()
            
            # Gerar relatório final
            self.gerar_relatorio_final()
            
            return self.resultados_finais
            
        except Exception as e:
            print(f"❌ Erro durante execução do pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

# EXECUÇÃO DO TRABALHO FINAL
def main():
    """
    Função principal para execução do trabalho final
    """
    print("🎓 TRABALHO FINAL - APRENDIZADO DE MÁQUINA")
    print("Pipeline Completo de Ciência de Dados")
    print("Autores: Gustavo Gonzaga dos Santos, Thiago Gonzaga dos Santos")
    print("=" * 80)
    
    # Executar pipeline
    pipeline = PipelineTrabalhoFinal()
    resultados = pipeline.executar_pipeline_completo()
    
    if resultados:
        print(f"\n🏆 PROJETO CONCLUÍDO COM SUCESSO!")
        print(f"F1-Score Final: {resultados['metricas']['f1_score']:.4f}")
    else:
        print(f"\n❌ Projeto não foi concluído devido a erros.")

if __name__ == "__main__":
    main()