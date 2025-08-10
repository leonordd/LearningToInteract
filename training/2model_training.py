#Versão SEM features de visibilidade - apenas coordenadas
# 2model_training.py --> BASE Version learning_to_interact/3integration/classification/version23/version22_1to4.py
#17_6 modificado para excluir completamente features de visibilidade (Face, Pose, RightHand, LeftHand)
# Treina apenas com features de coordenadas
#alterar o valor default de 500 para 10

"""# **LOAD DATA**"""
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    classification_report, roc_curve, auc, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
import joblib
import os
import time
import json
from collections import defaultdict

# ============================================================================
# CLASSE PARA ESTATÍSTICAS COMPLETAS (MANTIDA IGUAL)
# ============================================================================

class ModelStatistics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.training_history = defaultdict(list)
        self.epoch_times = []
        self.best_metrics = {}
        
    def update_training_history(self, epoch, train_loss, train_acc, test_loss, test_acc, 
                              learning_rate, grad_norm=None):
        """Atualiza histórico de treino"""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['test_loss'].append(test_loss)
        self.training_history['test_acc'].append(test_acc)
        self.training_history['learning_rate'].append(learning_rate)
        if grad_norm is not None:
            self.training_history['grad_norm'].append(grad_norm)
    
    def compute_comprehensive_metrics(self, y_true, y_pred, y_probs=None, 
                                    dataset_name="Test", verbose=True):
        """Calcula métricas abrangentes"""
        metrics = {}
        
        # Converter para numpy se necessário
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_probs):
            y_probs = y_probs.cpu().numpy()
        
        # 1. MÉTRICAS BÁSICAS
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Médias das métricas
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Médias ponderadas
        metrics['precision_weighted'] = np.average(precision, weights=support)
        metrics['recall_weighted'] = np.average(recall, weights=support)
        metrics['f1_weighted'] = np.average(f1, weights=support)
        
        # 2. OUTRAS MÉTRICAS IMPORTANTES
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # 3. MATRIZ DE CONFUSÃO
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Matriz normalizada por linha (recall por classe)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        metrics['confusion_matrix_normalized'] = cm_normalized.tolist()
        
        # 4. ESTATÍSTICAS DA DISTRIBUIÇÃO
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        metrics['true_distribution'] = dict(zip(unique_true.tolist(), counts_true.tolist()))
        metrics['pred_distribution'] = dict(zip(unique_pred.tolist(), counts_pred.tolist()))
        
        # 5. MÉTRICAS POR CLASSE (mais detalhadas)
        class_metrics = {}
        for i in range(self.num_classes):
            if i < len(precision):
                class_metrics[f'class_{i}'] = {
                    'precision': float(precision[i]) if not np.isnan(precision[i]) else 0.0,
                    'recall': float(recall[i]) if not np.isnan(recall[i]) else 0.0,
                    'f1': float(f1[i]) if not np.isnan(f1[i]) else 0.0,
                    'support': int(support[i])
                }
        metrics['class_metrics'] = class_metrics
        
        # 6. CONFIANÇA DAS PREDIÇÕES
        if y_probs is not None:
            max_probs = np.max(y_probs, axis=1)
            metrics['confidence_mean'] = float(np.mean(max_probs))
            metrics['confidence_std'] = float(np.std(max_probs))
            metrics['confidence_min'] = float(np.min(max_probs))
            metrics['confidence_max'] = float(np.max(max_probs))
            
            # Predições com baixa confiança (< 0.5)
            low_confidence = np.sum(max_probs < 0.5)
            metrics['low_confidence_predictions'] = int(low_confidence)
            metrics['low_confidence_percentage'] = float(low_confidence / len(y_pred) * 100)
        
        if verbose:
            self.print_metrics_summary(metrics, dataset_name)
        
        return metrics
    
    def print_metrics_summary(self, metrics, dataset_name):
        """Imprime resumo das métricas"""
        print(f"\n=== MÉTRICAS DETALHADAS - {dataset_name.upper()} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
        
        if 'confidence_mean' in metrics:
            print(f"Confiança média: {metrics['confidence_mean']:.4f} ± {metrics['confidence_std']:.4f}")
            print(f"Predições baixa confiança: {metrics['low_confidence_percentage']:.1f}%")
        
        print("\nMétricas por classe:")
        for class_name, class_data in metrics['class_metrics'].items():
            print(f"  {class_name}: P={class_data['precision']:.3f}, "
                  f"R={class_data['recall']:.3f}, F1={class_data['f1']:.3f}, "
                  f"N={class_data['support']}")
    
    def analyze_training_dynamics(self):
        """Analisa dinâmicas do treino"""
        history = self.training_history
        
        analysis = {
            'total_epochs': len(history['epoch']),
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'final_test_loss': history['test_loss'][-1] if history['test_loss'] else 0,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_test_acc': history['test_acc'][-1] if history['test_acc'] else 0,
            'best_test_acc': max(history['test_acc']) if history['test_acc'] else 0,
            'best_test_acc_epoch': np.argmax(history['test_acc']) if history['test_acc'] else 0
        }
        
        # Verificar overfitting
        if len(history['train_loss']) > 10:
            train_loss_trend = np.polyfit(range(len(history['train_loss'])), history['train_loss'], 1)[0]
            test_loss_trend = np.polyfit(range(len(history['test_loss'])), history['test_loss'], 1)[0]
            
            analysis['train_loss_trend'] = float(train_loss_trend)  # negativo = decrescente
            analysis['test_loss_trend'] = float(test_loss_trend)
            
            # Sinais de overfitting
            analysis['overfitting_signal'] = train_loss_trend < -0.001 and test_loss_trend > 0.001
            
            # Gap entre treino e teste
            final_gap = history['test_loss'][-1] - history['train_loss'][-1]
            analysis['final_loss_gap'] = float(final_gap)
            analysis['avg_loss_gap'] = float(np.mean([t - tr for t, tr in zip(history['test_loss'], history['train_loss'])]))
        
        # Estabilidade do learning rate
        if 'learning_rate' in history and len(history['learning_rate']) > 1:
            lr_changes = sum(1 for i in range(1, len(history['learning_rate'])) 
                           if history['learning_rate'][i] != history['learning_rate'][i-1])
            analysis['lr_changes'] = lr_changes
            analysis['final_lr'] = history['learning_rate'][-1]
            analysis['initial_lr'] = history['learning_rate'][0]
        
        return analysis
    
    def plot_training_curves(self, save_path=None):
        """Plota curvas de treino"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        history = self.training_history
        epochs = history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss', alpha=0.8)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(epochs, history['test_acc'], label='Test Accuracy', alpha=0.8)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[1, 0].plot(epochs, history['learning_rate'], alpha=0.8)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Gradient norm
        if 'grad_norm' in history:
            axes[1, 1].plot(epochs[:len(history['grad_norm'])], history['grad_norm'], alpha=0.8)
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Gradient Norm\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix, class_names=None, normalize=True, save_path=None):
        """Plota matriz de confusão"""
        cm = np.array(confusion_matrix)
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_statistics(self, filepath):
        """Salva todas as estatísticas em JSON"""
        all_stats = {
            'training_history': dict(self.training_history),
            'training_analysis': self.analyze_training_dynamics(),
            'best_metrics': self.best_metrics
        }
        
        # Converter numpy arrays e outros tipos para tipos serializáveis pelo JSON
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            elif obj is None:
                return None
            elif isinstance(obj, (str, int, float)):
                return obj
            else:
                # Para outros tipos, tentar converter para string
                try:
                    return str(obj)
                except:
                    return None
        
        all_stats = convert_to_json_serializable(all_stats)
        
        with open(filepath, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"Estatísticas salvas em: {filepath}")

# ============================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

csv_path_output = Path(__file__).resolve().parent / ".." / "data" / "dataset28" / "combinado28.csv"
merged_df = pd.read_csv(csv_path_output)
merged_df.info()
merged_df.describe().round(3)

"""# **DATA PREPARATION WITHOUT VISIBILITY FEATURES - COORDINATES ONLY**"""
print(f"Shape do DataFrame original: {merged_df.shape}")

# ============================================================================
# EXCLUSÃO DAS FEATURES DE VISIBILIDADE
# ============================================================================

print("\n=== EXCLUSÃO DAS FEATURES DE VISIBILIDADE ===")

# Definir features de visibilidade a serem excluídas
visibility_features_to_exclude = ['Face','Pose','RightHand','LeftHand']

# Colunas a excluir (features de visibilidade + colunas não-feature)
excluded_columns = ['MillisSinceEpoch','LocalMillisProcessing','AnimacaoAtual','ValorMapeado','Valor'] + visibility_features_to_exclude

print("AQUI", excluded_columns)
# Verificar quais features de visibilidade existem no dataset
existing_visibility_features = [col for col in visibility_features_to_exclude if col in merged_df.columns]
missing_visibility_features = [col for col in visibility_features_to_exclude if col not in merged_df.columns]

print(f"🔍 Features de visibilidade encontradas no dataset: {existing_visibility_features}")
if missing_visibility_features:
    print(f"⚠️ Features de visibilidade NÃO encontradas: {missing_visibility_features}")

# Se não encontrou as features com nomes exatos, assumir que são as primeiras 4 colunas
if not existing_visibility_features:
    print(f"⚠️ Nenhuma feature de visibilidade encontrada com nomes esperados.")
    print(f"Assumindo que as primeiras 4 colunas são features de visibilidade...")
    all_columns = [col for col in merged_df.columns if col not in ['MillisSinceEpoch','LocalMillisProcessing','AnimacaoAtual','ValorMapeado', 'Valor']]
    if len(all_columns) >= 4:
        existing_visibility_features = all_columns[:4]
        excluded_columns.extend(existing_visibility_features)
        print(f"Features assumidas como visibilidade (EXCLUÍDAS): {existing_visibility_features}")
    else:
        print(f"❌ ERRO: Dataset não tem features suficientes!")

# Selecionar apenas features de coordenadas
#coordinate_features = [col for col in merged_df.columns if col not in excluded_columns]
#coordinate_features = [col for col in merged_df.columns if col not in['MillisSinceEpoch','LocalMillisProcessing','AnimacaoAtual','ValorMapeado', 'Valor', 'Face', 'Pose', 'RightHand', 'LeftHand']]
def exclude_columns_by_pattern(df, exclude_patterns):
    all_columns = df.columns.tolist()
    columns_to_exclude = []
    
    for col in all_columns:
        for pattern in exclude_patterns:
            if col.startswith(pattern) or col == pattern:
                columns_to_exclude.append(col)
                break
    
    coordinate_features = [col for col in all_columns if col not in columns_to_exclude]
    return coordinate_features, columns_to_exclude

exclude_patterns = ['MillisSinceEpoch', 'LocalMillisProcessing', 'AnimacaoAtual', 
                   'ValorMapeado', 'Valor', 'Face', 'Pose', 'RightHand', 'LeftHand']

coordinate_features, excluded_columns_actual = exclude_columns_by_pattern(merged_df, exclude_patterns)

print(f"\n✅ RESULTADO DA SELEÇÃO:")
print(f"   Features de visibilidade EXCLUÍDAS ({len(existing_visibility_features)}): {existing_visibility_features}")
print(f"   Features de coordenadas MANTIDAS ({len(coordinate_features)}): {coordinate_features[:10]}... (+{len(coordinate_features)-10} mais)")
print(f"   Total de features para treino: {len(coordinate_features)}")

if len(coordinate_features) == 0:
    raise ValueError("❌ ERRO: Nenhuma feature de coordenada restante após exclusão!")

# ============================================================================
# ANÁLISE DAS FEATURES DE COORDENADAS
# ============================================================================
#all_500_mask = detect_all_500_samples(merged_df, coordinate_features)

print(f"\n=== ANÁLISE DAS FEATURES DE COORDENADAS ===")

# Selecionar apenas dados de coordenadas
X_coordinates_df = merged_df[coordinate_features].copy()

print(f"Dados de coordenadas selecionados - Shape: {X_coordinates_df.shape}")
print(f"Estatísticas básicas das coordenadas:")
print(X_coordinates_df.describe().round(3))

# ============================================================================
# FILTRAR LINHAS COM VALORES 500.0 (EM VEZ DE IMPUTAR)
# ============================================================================

print(f"\n=== FILTRAGEM DE LINHAS COM VALORES 500.0 ===")
X_coordinates_df = X_coordinates_df.replace(500.0, 0.0)  # Substituir 500.0 por 0.0
print(f"X_coordinates_df {X_coordinates_df}")
# Analisar quantas linhas contêm valores 500.0
"""rows_with_500 = (X_coordinates_df == 500.0).any(axis=1)
num_rows_with_500 = rows_with_500.sum()

print(f"Linhas com pelo menos um valor 500.0: {num_rows_with_500}")
print(f"Percentagem de linhas com valores 500.0: {(num_rows_with_500 / len(X_coordinates_df)) * 100:.2f}%")



# Analisar quantas features têm valor 500.0 por linha
features_500_per_row = (X_coordinates_df == 500.0).sum(axis=1)
print(f"Distribuição de features com valor 500.0 por linha:")
print(features_500_per_row.value_counts().sort_index())

# Filtrar linhas que NÃO contêm nenhum valor 500.0 E "valor" diferente de 0
clean_mask = ~rows_with_500  #& (~rows_with_valor_0)
X_coordinates_clean = X_coordinates_df[clean_mask].copy()
merged_df_clean = merged_df[clean_mask].copy()

print(f"\nApós filtrar linhas com valores 500.0:")
print(f"Shape original: {X_coordinates_df.shape}")
print(f"Shape após filtro: {X_coordinates_clean.shape}")
print(f"Linhas removidas: {num_rows_with_500}")
print(f"Linhas mantidas: {len(X_coordinates_clean)}")

# Verificar se ainda existem valores 500.0
remaining_500 = (X_coordinates_clean == 500.0).sum()
print(f"Valores 500.0 restantes após filtro: {remaining_500}")"""

#remover todas as linhas com 500 e com 0
# ============================================================================
# ANÁLISE FINAL DOS DADOS PROCESSADOS
# ============================================================================
#print("X_coordinates_clean", X_coordinates_clean)
print(f"\n=== ANÁLISE DOS DADOS PROCESSADOS ===")

print("Estatísticas APÓS o preprocessamento completo:")
#print(f"  Shape final: {X_coordinates_clean.shape}")
"""print("\n  Estatísticas por coluna:")
for col in X_coordinates_clean.columns:
    print(f"    {col}: min={X_coordinates_clean[col].min():.3f}, "
          f"max={X_coordinates_clean[col].max():.3f}, "
          f"mean={X_coordinates_clean[col].mean():.6f}, "
          f"std={X_coordinates_clean[col].std():.6f}")"""

# Usar dados como features finais
X_final = X_coordinates_df #
#X_final = X_coordinates_clean
final_feature_names = coordinate_features

#Y_final = merged_df_clean['AnimacaoAtual'].copy()
Y_final = merged_df['AnimacaoAtual'].copy()

"""print(f"Valor(output):\n{merged_df['AnimacaoAtual']}")
print(f"\nEstatísticas da coluna 'AnimacaoAtual':")
print(f"  Min: {merged_df['AnimacaoAtual'].min():.3f}")
print(f"  Max: {merged_df['AnimacaoAtual'].max():.3f}")
print(f"  Média: {merged_df['AnimacaoAtual'].mean():.6f}")
print(f"  Desvio padrão: {merged_df['AnimacaoAtual'].std():.6f}")
print(f"  Valores únicos na coluna 'AnimacaoAtual': {sorted(merged_df['AnimacaoAtual'].unique())}")

print(f"\n🎯 FEATURES FINAIS:")
print(f"  Total de features: {len(final_feature_names)}")
print(f"  Tipo: APENAS coordenadas (features de visibilidade excluídas)")
print(f"  Preprocessamento: Remover os valores 500.0")
print(f"  Primeiras 10 features: {final_feature_names[:10]}")"""


# Converter para tensors
X = torch.tensor(X_final.values, dtype=torch.float32)
y = torch.tensor(Y_final.values, dtype=torch.long)

print(f"\nShape dos tensores finais:")
print(f"  X (features): {X.shape}")
print(f"  y (target): {y.shape}")

"""##2.3. Data splitting"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=100,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X, y, 
    test_size=0.67, 
    random_state=100,
    stratify=y
)

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(X_train)} amostras")
print(f"  Validação: {len(X_val)} amostras")
print(f"  Teste: {len(X_test)} amostras")

# Verificações adicionais
input_num = X.shape[1]
output_num = len(np.unique(y))
print(f"\nVerificações finais:")
print(f"  Input size (nº de features): {input_num}")
print(f"  Output size (classes): {output_num}")
print(f"  Classes únicas: {np.unique(y)}")

# ============================================================================
# GUARDAR OBJETOS DE PREPROCESSAMENTO
# ============================================================================

model_objects_path = Path(__file__).resolve().parent / ".." / "data" / "output28"
model_objects_path.mkdir(parents=True, exist_ok=True)

class WeightedFlexibleModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, class_weights=None):
        super().__init__()
        
        self.class_weights = class_weights
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()  # Usar ReLU como padrão
        
    def forward(self, x):
        # Verificar se input tem todas features como 500 (após normalização será próximo de um valor específico)
        # Isso será tratado na inferência, não aqui
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Última camada sem ativação (será aplicado softmax na loss)
        logits = self.layers[-1](x)
        
        return logits
    
    def predict_with_null_check(self, x, null_threshold=0.9):
        """
        Predição especial que verifica se muitas features são nulas
        
        Args:
            x: tensor de entrada
            null_threshold: proporção de features que devem ser nulas para não fazer predição
        
        Returns:
            predictions: tensor com predições (-1 para amostras com muitos nulos)
            confidences: tensor com confiança das predições
        """
        self.eval()
        with torch.no_grad():
            # Calcular logits normalmente
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # Adicionar 1 para converter de 0-5 para 1-6
            #predictions = predictions + 1
            
            return predictions, confidences


# ============================================================================
# DEFINIR PESOS PARA AS CLASSES (RECOMPENSAR CLASSES IMPORTANTES)
# ============================================================================

print("\n=== DEFINIR PESOS DAS CLASSES ===")

# Calcular distribuição das classes
class_counts = torch.bincount(y)
total_samples = len(y)

# Calcular pesos inversamente proporcionais à frequência
class_weights_inverse = total_samples / (len(class_counts) * class_counts.float())

# Você pode ajustar manualmente os pesos para dar mais importância a certas classes
# Por exemplo, se as classes 3, 4, 5 são mais importantes:
class_weights_custom = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Classes 0-3 (internamente 0-3)

print(f"Distribuição das classes (0-5 interno):")
for i, count in enumerate(class_counts):
    print(f"  Classe {i} (real {i+1}): {count} amostras ({count/total_samples*100:.1f}%)")

print(f"\nPesos calculados (inverso da frequência):")
for i, weight in enumerate(class_weights_inverse):
    print(f"  Classe {i} (real {i+1}): {weight:.3f}")

print(f"\nPesos customizados (ajustados para importância):")
for i, weight in enumerate(class_weights_custom):
    print(f"  Classe {i} (real {i+1}): {weight:.3f}")

# Usar pesos customizados (device será definido mais tarde)
class_weights = class_weights_custom

# ============================================================================
# INICIALIZAR CLASSE DE ESTATÍSTICAS
# ============================================================================

stats = ModelStatistics(num_classes=output_num)

"""#3. BUILDING A MODEL"""
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"\nUsing device: {device}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
print(f"Using device: {device}")

# Mover pesos para o device correto
class_weights = class_weights.to(device)

# Criar modelo
model_0 = WeightedFlexibleModel(
    input_size=input_num, 
    output_size=output_num, 
    hidden_layers=[256, 128, 64],#[1024, 768, 512, 256]
    class_weights=class_weights,
).to(device)

print("Modelo criado:", model_0)
print(f"Parâmetros do modelo: {sum(p.numel() for p in model_0.parameters() if p.requires_grad):,}")

"""##3.3. Setup loss function and optimizer"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.0016, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

def accuracy_fn(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    accuracy = (correct / len(y_true)) * 100
    return accuracy

def weighted_accuracy_fn(y_true, y_pred, weights):
    """Calcula accuracy ponderada por classe"""
    correct = (y_pred == y_true).float()
    weighted_correct = correct * weights[y_true]
    return (weighted_correct.sum() / weights[y_true].sum()).item() * 100


"""# 4. TRAIN MODEL COM ESTATÍSTICAS INTEGRADAS"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 5000
best_test_acc = 0
best_weighted_acc = 0
patience_counter = 0
early_stop_patience = 100 #aumentado devido aos pesos

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

print("\n=== INÍCIO DO TREINO (APENAS COORDENADAS) ===")
print(f"Features utilizadas: {input_num} (APENAS coordenadas)")
print(f"Features de visibilidade EXCLUÍDAS: {existing_visibility_features}")
print(f"y_train range: {torch.min(y_train)} to {torch.max(y_train)}")
print(f"y_test range: {torch.min(y_test)} to {torch.max(y_test)}")

# Training loop COM ESTATÍSTICAS
for epoch in range(epochs):
    epoch_start_time = time.time()
    
    ### Training
    model_0.train()
    
    y_logits = model_0(X_train)
    y_pred = torch.argmax(y_logits.to('cpu'), dim=1).to(device)
    
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    weighted_acc = weighted_accuracy_fn(y_train, y_pred, class_weights)
    
    optimizer.zero_grad()
    loss.backward()
    
    # CALCULAR GRADIENT NORM PARA ESTATÍSTICAS
    total_norm = 0
    for p in model_0.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model_0.parameters(), max_norm=1.0)
    
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        val_logits = model_0(X_val)
        val_pred = torch.argmax(val_logits.to('cpu'), dim=1).to(device)
        
        val_loss = loss_fn(val_logits, y_val)
        val_acc = accuracy_fn(y_true=y_val, y_pred=val_pred)
        val_weighted_acc = weighted_accuracy_fn(y_val, val_pred, class_weights)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # ATUALIZAR ESTATÍSTICAS
    current_lr = optimizer.param_groups[0]['lr']
    stats.update_training_history(
        epoch, loss.item(), acc, val_loss.item(), val_acc, 
        current_lr, grad_norm
    )
    
    # Registrar tempo da época
    epoch_time = time.time() - epoch_start_time
    stats.epoch_times.append(epoch_time)
    
    # Early stopping
    """if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        # Salvar melhor modelo
        best_model_state = model_0.state_dict().copy()
    else:
        patience_counter += 1"""
    
    # Early stopping baseado em weighted accuracy
    if val_weighted_acc > best_weighted_acc:
        best_weighted_acc = val_weighted_acc
        best_val_acc = val_acc
        patience_counter = 0
        # Salvar melhor modelo
        best_model_state = model_0.to('cpu').state_dict().copy()
        model_0.to(device)
        print(f"Melhor modelo atualizado na época {epoch} com Test W-Acc: {best_weighted_acc:.2f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        # Restaurar melhor modelo
        model_0.load_state_dict(best_model_state)
        break
    
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, W-Acc: {weighted_acc:.2f}% | "
              f"Test loss: {val_loss:.5f}, Test acc: {val_acc:.2f}%, Test W-Acc: {val_weighted_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

print(f"\n=== TREINO CONCLUÍDO ===")
print(f"Melhor Test Accuracy: {best_test_acc:.2f}%")
print(f"Melhor Weighted Accuracy: {best_weighted_acc:.2f}%")


"""# 5. ANÁLISE COMPLETA COM ESTATÍSTICAS"""

# Calcular predições finais
model_0.eval()
with torch.inference_mode():
    # Teste
    test_logits = model_0(X_test)
    test_probs = torch.softmax(test_logits, dim=1)##alterar aqui??
    test_pred = torch.argmax(test_probs, dim=1)##alterar aqui??
    
    # Treino
    train_logits = model_0(X_train)
    train_probs = torch.softmax(train_logits, dim=1)
    train_pred = torch.argmax(train_probs, dim=1)

# ESTATÍSTICAS COMPLETAS
print("\n" + "="*60)
print("ANÁLISE COMPLETA DE PERFORMANCE - APENAS COORDENADAS")
print("="*60)

# Métricas detalhadas
test_metrics = stats.compute_comprehensive_metrics(
    y_test, test_pred, test_probs, "Test", verbose=True
)

train_metrics = stats.compute_comprehensive_metrics(
    y_train, train_pred, train_probs, "Train", verbose=True
)

# Análise por classe com pesos
print(f"\n=== ANÁLISE POR CLASSE (COM PESOS) ===")
for i in range(output_num):
    mask = (y_test == i)
    if mask.sum() > 0:
        class_acc = (test_pred[mask] == i).float().mean() * 100
        class_weight = class_weights[i].item()
        print(f"Classe {i+1}: Accuracy = {class_acc:.2f}%, Peso = {class_weight:.2f}, N = {mask.sum()}")

# Análise de dinâmicas de treino
training_dynamics = stats.analyze_training_dynamics()
print(f"\n=== ANÁLISE DO TREINO ===")
print(f"Épocas totais: {training_dynamics['total_epochs']}")
print(f"Melhor test accuracy: {training_dynamics['best_test_acc']:.2f}% (época {training_dynamics['best_test_acc_epoch']})")
print(f"Gap final train-test loss: {training_dynamics.get('final_loss_gap', 'N/A'):.4f}")
print(f"Sinal de overfitting: {'Sim' if training_dynamics.get('overfitting_signal', False) else 'Não'}")
print(f"Tempo médio por época: {np.mean(stats.epoch_times):.2f}s")
print(f"Tempo total de treino: {sum(stats.epoch_times):.1f}s")

# VISUALIZAÇÕES
print("\n=== GERANDO VISUALIZAÇÕES ===")

# Plotar curvas de treino
stats.plot_training_curves(save_path="../data/output28/training_curves_coordinates_only.png")

# Plotar matriz de confusão
class_names = [str(i) for i in range(0, 4)]
stats.plot_confusion_matrix(
    test_metrics['confusion_matrix'], 
    class_names=class_names,
    save_path="../data/output28/confusion_matrix_coordinates_only.png"
)

# SALVAR RESULTADOS
stats.best_metrics = {'train': train_metrics, 'test': test_metrics}

# Salvar estatísticas completas
stats_path = Path(__file__).resolve().parent /".."/ "data" / "output28" / "training_statistics_coordinates_only.json"
stats.save_all_statistics(stats_path)

"""# 6. SAVE MODEL AND PREPROCESSING OBJECTS"""
# Guardar modelo com informações sobre uso apenas de coordenadas
model_save_path = Path(__file__).resolve().parent /".." / "data" / "output28" / "trained_model_coordinates_only.pth"
model_save_path.parent.mkdir(parents=True, exist_ok=True)

model_info = {
    'model_state_dict': model_0.state_dict(),
    'input_size': input_num,
    'output_size': output_num,
    'hidden_layers': [256, 128, 64],#[1024, 768, 512, 256],
    'model_class': 'FlexibleModel',
    'feature_info': {
        'total_features': len(final_feature_names),
        'feature_type': 'coordinates_only',
        'coordinate_features': coordinate_features,
        'feature_order': final_feature_names,
        'excluded_visibility_features': existing_visibility_features
    },
    'best_accuracy': best_test_acc,
    'best_weighted_accuracy': best_weighted_acc,
    'preprocessing_strategy': 'coordinates_only',
    'preprocessing_details': {
        'coordinates_only': {
            'positions': list(range(len(coordinate_features))),
            #'preprocessing': '5.0 → NaN → median imputation → StandardScaler',
            'expected_range': 'standardized (mean≈0, std≈1)',
            #'scaler_type': 'StandardScaler'
        },
        'excluded_features': {
            'visibility_features': existing_visibility_features,
            'reason': 'Excluded to avoid feature dominance and test coordinate-only performance'
        }
    },
    'null_handling': {
        'null_placeholder_value': 0.0,
        'strategy': 'median_imputation_then_standardize',
        #'imputer_required': True,
        #'scaler_required': True,
        'applies_to': 'coordinates_only'
    },
    'model_characteristics': {
        'feature_strategy': 'coordinates_only',
        'no_visibility_features': True,
        'total_parameters': sum(p.numel() for p in model_0.parameters() if p.requires_grad),
        'feature_count': len(coordinate_features)
    },
    'final_test_metrics': test_metrics,
    'training_summary': training_dynamics
}

torch.save(model_info, model_save_path)
print(f"\nModelo guardado em: {model_save_path}")

# ============================================================================
# ANÁLISE DA PERFORMANCE SEM FEATURES DE VISIBILIDADE
# ============================================================================

print(f"\n=== ANÁLISE DA PERFORMANCE SEM FEATURES DE VISIBILIDADE ===")

print(f"📊 COMPARAÇÃO DE COMPLEXIDADE:")
print(f"  Features com visibilidade: {len(existing_visibility_features)} vis + {len(coordinate_features)} coord = {len(existing_visibility_features) + len(coordinate_features)} total")
print(f"  Features apenas coordenadas: {len(coordinate_features)} coord")
print(f"  Redução de features: {len(existing_visibility_features)} features ({len(existing_visibility_features)/(len(existing_visibility_features) + len(coordinate_features))*100:.1f}%)")

print(f"\n🎯 PERFORMANCE ALCANÇADA (APENAS COORDENADAS):")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  F1-Score (macro): {test_metrics['f1_macro']:.4f}")
print(f"  Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
print(f"  Matthews Corr: {test_metrics['matthews_corrcoef']:.4f}")


# ============================================================================
# ANÁLISE DE IMPORTÂNCIA DAS COORDENADAS
# ============================================================================

print(f"\n=== ANÁLISE PRELIMINAR DE IMPORTÂNCIA ===")

# Análise simples dos pesos da primeira camada
model_0.eval()
first_layer_weights = model_0.layers[0].weight.data.cpu().numpy()
weight_importance = np.abs(first_layer_weights).mean(axis=0)

# Top 10 coordenadas mais importantes
top_indices = np.argsort(weight_importance)[::-1][:10]

print(f"🏆 TOP 10 COORDENADAS MAIS IMPORTANTES (análise preliminar):")
for i, idx in enumerate(top_indices):
    coord_name = coordinate_features[idx] if idx < len(coordinate_features) else f"coord_{idx}"
    importance = weight_importance[idx]
    print(f"  {i+1:2d}. {coord_name}: {importance:.4f}")

# RELATÓRIO FINAL
print("\n" + "="*70)
print("RELATÓRIO FINAL - MODELO APENAS COM COORDENADAS")
print("="*70)
print(f"📊 Performance Final: Accuracy = {test_metrics['accuracy']:.4f}")
print(f"🧠 Parâmetros do modelo: {sum(p.numel() for p in model_0.parameters() if p.requires_grad):,}")
print(f"📈 F1-Score (macro): {test_metrics['f1_macro']:.4f}")
print(f"🎯 Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
print(f"🔄 Épocas treinadas: {training_dynamics['total_epochs']}")
print(f"⏱️ Tempo total: {sum(stats.epoch_times):.1f}s")

print(f"\n🔧 ESTRATÉGIA IMPLEMENTADA:")
print(f"   Features de visibilidade EXCLUÍDAS: {existing_visibility_features}")
print(f"   Features de coordenadas MANTIDAS: {len(coordinate_features)}")

print(f"\n📊 ESTATÍSTICAS DE DADOS:")
print(f"   - Total features originais: {len(existing_visibility_features) + len(coordinate_features)}")
print(f"   - Features utilizadas: {len(coordinate_features)} ({len(coordinate_features)/(len(existing_visibility_features) + len(coordinate_features))*100:.1f}%)")

print(f"\n💾 Arquivos salvos:")
print(f"   - Modelo: {model_save_path}")
print(f"   - Estatísticas: {stats_path}")
print(f"   - Gráficos: training_curves_coordinates_only.png, confusion_matrix_coordinates_only.png")
print(f"\n✅ Implementação do modelo APENAS COM COORDENADAS concluída!")

