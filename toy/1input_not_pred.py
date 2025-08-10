#correr este codigo
import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# =============== MODELO DA REDE NEURONAL MODIFICADO ===============
class WeightedFlexibleModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[10], class_weights=None):
        super().__init__()
        
        self.class_weights = class_weights
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()  # Usar ReLU
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Última camada sem ativação (será aplicado softmax na loss)
        logits = self.layers[-1](x)
        
        return logits

# =============== CLASSE PARA PREDIÇÃO DO MODELO ===============
class ModelPredictor:
    def __init__(self, model_path: str, scaler_path: str = None, imputer_path: str = None,
                 null_threshold: float = 0.95, confidence_threshold: float = 0.3):
        """
        Inicializa o preditor do modelo.
        
        Args:
            model_path: Caminho para o modelo treinado
            scaler_path: Caminho para o StandardScaler
            imputer_path: Caminho para o SimpleImputer
            null_threshold: Proporção de features 500.0 para considerar amostra inválida
            confidence_threshold: Threshold mínimo de confiança para fazer predição
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.null_threshold = null_threshold
        self.confidence_threshold = confidence_threshold
        print(f"A usar o device: {self.device}")
        
        # Carregar informações do modelo
        self.model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = self.load_model()
        
        # Guardar informações importantes do modelo
        self.input_size = self.model_data['input_size']
        self.output_size = self.model_data['output_size']
        self.preprocessing_strategy = self.model_data.get('preprocessing_strategy', 'none')
        
        # Extrair mapeamento de classes
        if 'class_mapping' in self.model_data:
            self.internal_to_real = self.model_data['class_mapping']['internal_to_real']
            self.real_to_internal = self.model_data['class_mapping']['real_to_internal']
            # Criar nomes de classes baseados no mapeamento real (1-4)
            self.class_names = [f"Classe {self.internal_to_real[i]}" for i in range(self.output_size)]
        else:
            # Fallback se não houver mapeamento
            self.internal_to_real = {i: i+1 for i in range(self.output_size)}
            self.class_names = [f"Classe {i+1}" for i in range(self.output_size)]
        
        # Carregar preprocessadores salvos do treino
        """if scaler_path and imputer_path:
            try:
                self.scaler = joblib.load(scaler_path)
                self.imputer = joblib.load(imputer_path)
                print("✓ Scaler e Imputer carregados do treino")
                
                # Verificar medianas do imputer
                if hasattr(self.imputer, 'statistics_'):
                    print(f"✓ Medianas carregadas: {len(self.imputer.statistics_)} features")
                    print(f"  Exemplo medianas: {self.imputer.statistics_[:5]}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar preprocessadores: {e}")
                self.scaler = None
                self.imputer = None
        else:
            self.scaler = None
            self.imputer = None
            print("⚠️ AVISO: Sem Scaler/Imputer - preprocessamento pode estar incorreto!")"""
        self.scaler = None
        self.imputer = None
        print("✓ Preprocessamento: Básico (sem scaler/imputer)")

        # Debug do modelo carregado
        print("\n=== INFO DO MODELO ===")
        print(f"✓ Modelo carregado: {self.input_size} inputs, {self.output_size} outputs")
        print(f"✓ Classes de saída: 1-4 (internamente 0-3)")
        print(f"Hidden layers: {self.model_data.get('hidden_layers', 'N/A')}")
        print(f"Preprocessing: {self.preprocessing_strategy}")
        print(f"Classes: {self.class_names}")
        print(f"Null threshold: {self.null_threshold*100:.0f}%")
        print(f"Confidence threshold: {self.confidence_threshold*100:.0f}%")
        
        # Mostrar pesos das classes se disponíveis
        if 'class_weights' in self.model_data:
            weights = self.model_data['class_weights']
            print(f"Pesos das classes:")
            for i, weight in enumerate(weights):
                real_class = self.internal_to_real.get(i, i+1)
                print(f"  Classe {real_class}: peso {weight:.2f}")

    def load_model(self) -> nn.Module:
        """Carrega o modelo treinado."""
        state_dict = self.model_data['model_state_dict']
        input_size = self.model_data['input_size']
        output_size = self.model_data['output_size']
        hidden_layers = self.model_data['hidden_layers']
        class_weights = self.model_data.get('class_weights', None)
        
        # Criar modelo com a mesma arquitetura
        model = WeightedFlexibleModel(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            class_weights=torch.tensor(class_weights) if class_weights else None
        ).to(self.device)
        
        # Carregar pesos
        model.load_state_dict(state_dict)
        model.eval()  # Modo de avaliação
        
        return model
    
    def check_invalid_sample(self, raw_data: List[float]) -> Tuple[bool, float]:
        """
        Verifica se a amostra tem muitos valores 500.0 (não detectados).
        
        Returns:
            (is_invalid, null_proportion)
        """
        null_count = raw_data.count(500.0)
        null_proportion = null_count / len(raw_data)
        is_invalid = null_proportion >= self.null_threshold
        
        return is_invalid, null_proportion
    
    def preprocess_data(self, raw_data: List[float], debug: bool = False) -> torch.Tensor:
        """
        Preprocessa os dados EXATAMENTE como no treino.
        
        Args:
            raw_data: Lista com os valores das landmarks
            debug: Se deve imprimir informações de debug
            
        Returns:
            Tensor processado pronto para o modelo
        """
        if debug:
            print(f"\n=== DEBUG: Preprocessamento ===")
            print(f"Estratégia de preprocessamento: {self.preprocessing_strategy}")
        
        # Converter para numpy array e reshape para 2D (necessário para sklearn)
        data = np.array(raw_data, dtype=np.float32).reshape(1, -1)
        
        if debug:
            num_500s = (data == 500.0).sum()
            total_values = data.size
            print(f"Shape dos dados: {data.shape}")
            print(f"Valores 500.0 encontrados: {num_500s} de {total_values} ({num_500s/total_values*100:.1f}%)")
            print(f"Min: {np.min(data):.3f}, Max: {np.max(data):.3f}, Mean: {np.mean(data):.3f}")
        
        """if self.scaler is not None and self.imputer is not None:
            # USAR O MESMO PIPELINE DO TREINO
            
            # 1. Substituir 500.0 por NaN
            if debug:
                print("\n1. Substituindo 500.0 por NaN...")
            
            data_with_nan = data.copy()
            data_with_nan[data_with_nan == 500.0] = np.nan
            
            if debug:
                nan_count = np.isnan(data_with_nan).sum()
                print(f"   Valores NaN após substituição: {nan_count}")
            
            # 2. Imputar com mediana (a usar o imputer treinado)
            if debug:
                print("2. Aplicar imputação com mediana do treino...")
            
            data_imputed = self.imputer.transform(data_with_nan)
            
            if debug:
                print(f"   Após imputação - Min: {data_imputed.min():.3f}, Max: {data_imputed.max():.3f}")
                remaining_nans = np.isnan(data_imputed).sum()
                print(f"   NaNs restantes: {remaining_nans}")
            
            # 3. Normalizar com StandardScaler (a usar o scaler treinado)
            if debug:
                print("3. Aplicar normalização StandardScaler do treino...")
            
            data_normalized = self.scaler.transform(data_imputed)
            
            if debug:
                print(f"   Após normalização - Min: {data_normalized.min():.3f}, Max: {data_normalized.max():.3f}")
                print(f"   Média: {data_normalized.mean():.6f}, Std: {data_normalized.std():.6f}")
                
                # Verificar valores extremos
                extreme_high = (data_normalized > 3).sum()
                extreme_low = (data_normalized < -3).sum()
                if extreme_high > 0 or extreme_low > 0:
                    print(f"   ⚠️ Valores extremos: {extreme_high} > 3, {extreme_low} < -3")
            
            data = data_normalized
        
        else:
            # Fallback se não tiver os preprocessadores
            print("⚠️ AVISO: Usa preprocessamento básico (não recomendado)")
            print("   O modelo pode não funcionar corretamente!")
            
            # Substituir 500.0 por 0.5 (centro)
            data[data == 500.0] = 0.5
            
            # Clip básico para garantir range [0, 1]
            data = np.clip(data, 0, 1)
            
            if debug:
                print(f"   Preprocessamento básico - Min: {data.min():.3f}, Max: {data.max():.3f}")
        """
        # Preprocessamento básico: substituir valores ausentes por 0.0
        if debug:
            print("Preprocessamento básico: substituir 0.0 e normalizar para [0,1]")

        # Substituir valores ausentes (500.0) por 0.0
        data[data == 500.0] = 0.0

        # (Opcional) Clip dos dados para garantir valores entre 0 e 1
        data = np.clip(data, 0, 1)

        if debug:
            print(f"   Min: {data.min():.3f}, Max: {data.max():.3f}, Mean: {data.mean():.3f}")

        # Converter para tensor PyTorch
        tensor_data = torch.FloatTensor(data).to(self.device)
        
        if debug:
            print(f"\nShape do tensor final: {tensor_data.shape}")
        
        return tensor_data
    
    def predict(self, raw_data: List[float], debug: bool = False) -> Dict[str, any]:
        """
        Faz a predição com os dados de entrada.
        
        NOVO COMPORTAMENTO:
        - Retorna classe -1 (sem predição) se muitos valores são 500.0
        - Retorna classes 1-4 (não 0-3)
        - Aplica threshold de confiança
        
        Args:
            raw_data: Lista com os valores das landmarks
            debug: Se deve imprimir informações de debug
            
        Returns:
            Dicionário com a classe predita e probabilidades
        """
        if debug:
            print(f"\n=== DEBUG: Predição ===")
        
        # Verificar se temos o número correto de features
        if len(raw_data) != self.input_size:
            if debug:
                print(f"ERRO: Número incorreto de features!")
            return {
                'error': f'Número incorreto de features: esperado {self.input_size}, recebido {len(raw_data)}',
                'predicted_class': -1,
                'confidence': 0.0,
                'valid': False
            }
        
        # VERIFICAR SE A AMOSTRA É INVÁLIDA (muitos 500.0)
        is_invalid, null_proportion = self.check_invalid_sample(raw_data)
        
        if is_invalid:
            if debug or null_proportion > 0.8:
                print(f"\n⚠️ AMOSTRA INVÁLIDA: {null_proportion*100:.1f}% dos valores são 500.0")
                print(f"   Threshold: {self.null_threshold*100:.0f}%")
                print(f"   → Não fazendo predição")
            
            return {
                'predicted_class': -1,
                'class_name': 'Sem Predição (muitos valores ausentes)',
                'confidence': 0.0,
                'all_probabilities': np.zeros(self.output_size),
                'valid': False,
                'null_proportion': null_proportion,
                'reason': 'too_many_nulls'
            }
        
        # Analisar dados brutos antes do preprocessamento
        num_500s = raw_data.count(500.0)
        if debug and num_500s > 0:
            print(f"\n⚠️ {num_500s} valores 500.0 nos dados de entrada ({num_500s/len(raw_data)*100:.1f}%)")
            print(f"   Estes valores serão imputados com as medianas do treino")
        
        # Preprocessar dados
        processed_data = self.preprocess_data(raw_data, debug)
        
        # Fazer predição
        with torch.no_grad():
            outputs = self.model(processed_data)
            
            if debug:
                print(f"\nOutputs do modelo (logits): {outputs[0].cpu().numpy()}")
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class_internal = torch.max(probabilities, 1)
            
            # Converter de classe interna (0-3) para classe real (1-4)
            predicted_class_internal = predicted_class_internal.item()
            predicted_class_real = self.internal_to_real.get(predicted_class_internal, predicted_class_internal + 1)
            
            if debug:
                # Imprimir top 3 classes
                k = min(3, self.output_size)
                top_probs, top_classes = torch.topk(probabilities[0], k)
                print(f"\nTop {k} predições:")
                for i, (prob, cls) in enumerate(zip(top_probs, top_classes)):
                    real_class = self.internal_to_real.get(cls.item(), cls.item() + 1)
                    print(f"  {i+1}. Classe {real_class}: {prob.item()*100:.1f}%")
        
        # Converter resultados para Python nativo
        confidence = confidence.item()
        all_probabilities = probabilities[0].cpu().numpy()
        
        # VERIFICAR THRESHOLD DE CONFIANÇA
        if confidence < self.confidence_threshold:
            if debug:
                print(f"\n⚠️ CONFIANÇA BAIXA: {confidence*100:.1f}% < {self.confidence_threshold*100:.0f}%")
                print(f"   → Não fazendo predição")
            
            return {
                'predicted_class': -1,
                'class_name': 'Sem Predição (confiança baixa)',
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'valid': False,
                'null_proportion': null_proportion,
                'reason': 'low_confidence',
                'best_guess': predicted_class_real
            }
        
        # PREDIÇÃO VÁLIDA
        return {
            'predicted_class': predicted_class_real,
            'class_name': f"Classe {predicted_class_real}",
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'valid': True,
            'null_proportion': null_proportion
        }

# =============== CONFIGURAÇÃO DO SISTEMA ===============
class PoseCaptureConfig:
    """Configurações do sistema de captura."""
    
    # Definições da câmara
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 60
    
    # Número de landmarks por tipo
    NUM_POSE_LANDMARKS = 33
    NUM_HAND_LANDMARKS = 21
    NUM_FACE_LANDMARKS = 468
    
    # Configurações do MediaPipe
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 0  # 0 = mais rápido, 2 = mais preciso
    
    # Valor padrão para landmarks não detectadas
    DEFAULT_LANDMARK_VALUE = 0.0

# =============== PROCESSADOR DE LANDMARKS ===============
class LandmarkProcessor:
    """Processador de landmarks do MediaPipe."""
    
    @staticmethod
    def extract_features_for_model(results, feature_config: Dict, debug: bool = False) -> List[float]:
        """
        Extrai todas as features necessárias para o modelo (1629 features).
        
        NOTA: Features de visibilidade (Face, Pose, RightHand, LeftHand) foram REMOVIDAS
        conforme o modelo treinado.
        
        Ordem das features:
        1. Landmarks da pose: 33 pontos × 3 coordenadas = 99 valores
        2. Landmarks da mão esquerda: 21 pontos × 3 coordenadas = 63 valores
        3. Landmarks da mão direita: 21 pontos × 3 coordenadas = 63 valores
        4. Landmarks da face: 468 pontos × 3 coordenadas = 1404 valores
        
        Total: 99 + 63 + 63 + 1404 = 1629 features (apenas coordenadas)
        """
        features = []
        
        if debug:
            print("\n=== DEBUG: Extração de Features ===")
            print("NOTA: Modelo treinado SEM features de visibilidade (apenas coordenadas)")
        
        # ========== LANDMARKS DA POSE (99 features) ==========
        pose_start = len(features)
        if results.pose_landmarks:
            landmarks_to_use = list(results.pose_landmarks.landmark)[:PoseCaptureConfig.NUM_POSE_LANDMARKS]
            
            if debug:
                # Debug: imprimir primeiros 3 landmarks da pose
                for i, landmark in enumerate(landmarks_to_use[:3]):
                    print(f"  Pose Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
            
            for landmark in landmarks_to_use:
                features.extend([landmark.x, landmark.y, landmark.z])
            remaining = PoseCaptureConfig.NUM_POSE_LANDMARKS - len(landmarks_to_use)
            if remaining > 0:
                features.extend([PoseCaptureConfig.DEFAULT_LANDMARK_VALUE] * (remaining * 3))
        else:
            features.extend([PoseCaptureConfig.DEFAULT_LANDMARK_VALUE] *  (PoseCaptureConfig.NUM_POSE_LANDMARKS * 3))
        
        if debug:
            print(f"Features da Pose (índices {pose_start}-{len(features)-1}): {len(features)-pose_start} valores")
        
        # ========== LANDMARKS DA MÃO ESQUERDA (63 features) ==========
        lh_start = len(features)
        if results.left_hand_landmarks:
            landmarks_to_use = list(results.left_hand_landmarks.landmark)[:PoseCaptureConfig.NUM_HAND_LANDMARKS]
            
            if debug and landmarks_to_use:
                print(f"  Mão Esq. Landmark 0: x={landmarks_to_use[0].x:.3f}, y={landmarks_to_use[0].y:.3f}, z={landmarks_to_use[0].z:.3f}")
            
            for landmark in landmarks_to_use:
                features.extend([landmark.x, landmark.y, landmark.z])
            remaining = PoseCaptureConfig.NUM_HAND_LANDMARKS - len(landmarks_to_use)
            if remaining > 0:
                features.extend([500.0] * (remaining * 3))
        else:
            features.extend([500.0] * (PoseCaptureConfig.NUM_HAND_LANDMARKS * 3))
        
        if debug:
            print(f"Features Mão Esq. (índices {lh_start}-{len(features)-1}): {len(features)-lh_start} valores")
        
        # ========== LANDMARKS DA MÃO DIREITA (63 features) ==========
        rh_start = len(features)
        if results.right_hand_landmarks:
            landmarks_to_use = list(results.right_hand_landmarks.landmark)[:PoseCaptureConfig.NUM_HAND_LANDMARKS]
            
            if debug and landmarks_to_use:
                print(f"  Mão Dir. Landmark 0: x={landmarks_to_use[0].x:.3f}, y={landmarks_to_use[0].y:.3f}, z={landmarks_to_use[0].z:.3f}")
            
            for landmark in landmarks_to_use:
                features.extend([landmark.x, landmark.y, landmark.z])
            remaining = PoseCaptureConfig.NUM_HAND_LANDMARKS - len(landmarks_to_use)
            if remaining > 0:
                features.extend([500.0] * (remaining * 3))
        else:
            features.extend([500.0] * (PoseCaptureConfig.NUM_HAND_LANDMARKS * 3))
        
        if debug:
            print(f"Features Mão Dir. (índices {rh_start}-{len(features)-1}): {len(features)-rh_start} valores")
        
        # ========== LANDMARKS DA FACE (1404 features) ==========
        face_start = len(features)
        if results.face_landmarks:
            landmarks_to_use = list(results.face_landmarks.landmark)[:PoseCaptureConfig.NUM_FACE_LANDMARKS]
            
            if debug and landmarks_to_use:
                print(f"  Face Landmark 0: x={landmarks_to_use[0].x:.3f}, y={landmarks_to_use[0].y:.3f}, z={landmarks_to_use[0].z:.3f}")
            
            for landmark in landmarks_to_use:
                features.extend([landmark.x, landmark.y, landmark.z])
            remaining = PoseCaptureConfig.NUM_FACE_LANDMARKS - len(landmarks_to_use)
            if remaining > 0:
                features.extend([500.0] * (remaining * 3))
            
            actual_count = len(results.face_landmarks.landmark)
            if actual_count != PoseCaptureConfig.NUM_FACE_LANDMARKS:
                print(f"AVISO: Face tem {actual_count} landmarks, esperado {PoseCaptureConfig.NUM_FACE_LANDMARKS}")
        
        else:
            features.extend([500.0] * (PoseCaptureConfig.NUM_FACE_LANDMARKS * 3))
        
        if debug:
            print(f"Features Face (índices {face_start}-{len(features)-1}): {len(features)-face_start} valores")
            print(f"\nTotal de features extraídas: {len(features)}")
            print(f"Primeiras 10 features: {features[:10]}")
            print(f"Últimas 10 features: {features[-10:]}")
            
            # Verificar valores especiais
            num_zeros = features.count(0)
            num_default = features.count(500.0)
            num_ones = features.count(1)
            print(f"Valores especiais: {num_zeros} zeros, {num_ones} uns, {num_default} valores 500.0")
            
            # Estatísticas das coordenadas válidas
            valid_coords = [f for f in features if f != 500.0]
            if valid_coords:
                print(f"Coordenadas válidas: {len(valid_coords)}, Min={min(valid_coords):.3f}, Max={max(valid_coords):.3f}")
        
        # Verificar que temos exatamente 1629 features (sem as 4 de visibilidade)
        EXPECTED_FEATURES = 1629
        if len(features) != EXPECTED_FEATURES:
            raise ValueError(f"Número de features incorreto: {len(features)} (esperado: {EXPECTED_FEATURES})")
        
        return features

# =============== SISTEMA DE CAPTURA COM PREDIÇÃO ===============
class VideoCaptureWithPrediction:
    """Sistema de captura com predição em tempo real."""
    
    def __init__(self, model_path: str, show_visualization: bool = True, target_fps: float = 30.0,
                 null_threshold: float = 0.95, confidence_threshold: float = 0.3):
        """
        Inicializa o sistema de captura com modelo.
        
        Args:
            model_path: Caminho para o modelo treinado
            scaler_path: Caminho para o StandardScaler do treino
            imputer_path: Caminho para o SimpleImputer do treino
            show_visualization: Se deve mostrar visualização
            target_fps: FPS desejado
            null_threshold: Proporção de features 500.0 para considerar amostra inválida
            confidence_threshold: Threshold mínimo de confiança para fazer predição
        """
        self.show_visualization = show_visualization
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Inicializar MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Carregar modelo de predição com preprocessadores
        self.predictor = ModelPredictor(
            model_path, #scaler_path, imputer_path,
            null_threshold=null_threshold,
            confidence_threshold=confidence_threshold
        )
        
        # Inicializar webcam
        self.cap = self._setup_camera()
        
        # Histórico de predições (para análise posterior)
        self.prediction_history = []
        
        # Contadores para estatísticas
        self.stats = {
            'total_frames': 0,
            'valid_predictions': 0,
            'invalid_too_many_nulls': 0,
            'invalid_low_confidence': 0,
            'class_counts': {i: 0 for i in range(1, 5)}  # Classes 1-4
        }
        
        print("Sistema inicializado com modelo de predição (Classes 1-4)!")
        #if scaler_path and imputer_path:
            #print("✓ Preprocessamento configurado corretamente com Scaler e Imputer do treino")
        #else:
            #print("⚠️ AVISO: Sem Scaler/Imputer - resultados podem ser incorretos!")
    
    def _setup_camera(self) -> cv2.VideoCapture:
        """Configura a webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PoseCaptureConfig.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PoseCaptureConfig.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, PoseCaptureConfig.CAMERA_FPS)
        
        if not cap.isOpened():
            raise RuntimeError("Erro: Não foi possível aceder à webcam!")
        
        return cap
    
    def _draw_landmarks(self, image, results):
        """Desenha todas as landmarks na imagem (face, pose e mãos)."""
        # Desenhar malha facial
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, 
                self.mp_holistic.FACEMESH_TESSELATION,
                None, 
                self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Desenhar pose corporal
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, 
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Desenhar mão esquerda
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        # Desenhar mão direita
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
    
    def _add_prediction_overlay(self, image, prediction: Dict, results):
        """
        Adiciona informação da predição na imagem.
        
        Args:
            image: Imagem onde adicionar o texto
            prediction: Resultado da predição do modelo
            results: Resultados do MediaPipe para mostrar estado de detecção
        """
        # Verificar se é uma predição válida
        if 'error' in prediction:
            # Mostrar erro se houver
            cv2.putText(
                image, f"Erro: {prediction['error']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
        elif not prediction.get('valid', False):
            # Mostrar razão da não-predição
            reason = prediction.get('reason', 'unknown')
            null_pct = prediction.get('null_proportion', 0) * 100
            
            cv2.putText(
                image, "SEM PREDICAO", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )
            
            if reason == 'too_many_nulls':
                cv2.putText(
                    image, f"Muitos valores ausentes ({null_pct:.0f}%)", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1
                )
            elif reason == 'low_confidence':
                confidence = prediction.get('confidence', 0) * 100
                best_guess = prediction.get('best_guess', '?')
                cv2.putText(
                    image, f"Confianca baixa: {confidence:.1f}%", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1
                )
                cv2.putText(
                    image, f"(Melhor palpite: Classe {best_guess})", 
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1
                )
        else:
            # Mostrar classe predita e confiança (CLASSES 1-4)
            class_name = prediction['class_name']
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence'] * 100  # Converter para percentagem
            null_pct = prediction.get('null_proportion', 0) * 100
            
            # Cor baseada na confiança
            if confidence > 80:
                color = (0, 255, 0)  # Verde
            elif confidence > 60:
                color = (0, 255, 255)  # Amarelo
            else:
                color = (0, 165, 255)  # Laranja
            
            # Texto principal com a predição
            cv2.putText(
                image, f"CLASSE {predicted_class}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
            )
            cv2.putText(
                image, f"Confianca: {confidence:.1f}%", 
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            
            # Mostrar barra de confiança
            bar_width = int(confidence * 2)  # Largura máxima 200 pixels
            cv2.rectangle(image, (10, 85), (10 + bar_width, 105), color, -1)
            cv2.rectangle(image, (10, 85), (210, 105), (255, 255, 255), 2)
            
            # Mostrar percentagem de valores ausentes
            cv2.putText(
                image, f"Valores ausentes: {null_pct:.0f}%", 
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Mostrar estado de detecção dos componentes
        y_pos = 155
        detections = [
            ("Face", results.face_landmarks is not None),
            ("Pose", results.pose_landmarks is not None),
            ("Mao Esq", results.left_hand_landmarks is not None),
            ("Mao Dir", results.right_hand_landmarks is not None)
        ]
        
        for name, detected in detections:
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = "OK" if detected else "X"
            cv2.putText(
                image, f"{name}: {status}", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            y_pos += 25
        
        # Adicionar nota sobre preprocessamento
        """if self.predictor.scaler is not None:
            cv2.putText(
                image, "Preprocessamento: OK", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
        else:
            cv2.putText(
                image, "Preprocessamento: BASICO", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1
            )"""
        cv2.putText(image, "Preprocessamento: Basico (0.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        
        # Mostrar estatísticas no canto superior direito
        stats_x = image.shape[1] - 200
        cv2.putText(
            image, "ESTATISTICAS", 
            (stats_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            image, f"Frames: {self.stats['total_frames']}", 
            (stats_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        cv2.putText(
            image, f"Validas: {self.stats['valid_predictions']}", 
            (stats_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        cv2.putText(
            image, f"Rejeitadas: {self.stats['invalid_too_many_nulls'] + self.stats['invalid_low_confidence']}", 
            (stats_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )
    
    def check_data_consistency(self, features_list: List[List[float]]):
        """Verifica a consistência dos dados ao longo do tempo."""
        if len(features_list) < 2:
            return
        
        print("\n=== DEBUG: Consistência dos Dados ===")
        
        # Comparar features consecutivas
        prev_features = np.array(features_list[-2])
        curr_features = np.array(features_list[-1])
        
        # Calcular diferença
        diff = np.abs(curr_features - prev_features)
        max_diff_idx = np.argmax(diff)
        
        print(f"Maior mudança: índice {max_diff_idx}, valor {diff[max_diff_idx]:.3f}")
        print(f"Mudanças > 0.1: {np.sum(diff > 0.1)} features")
        print(f"Mudanças > 0.5: {np.sum(diff > 0.5)} features")
        
        # Analisar estabilidade das coordenadas
        valid_mask = (prev_features != 500.0) & (curr_features != 500.0)
        if np.any(valid_mask):
            valid_diffs = diff[valid_mask]
            print(f"\nEstabilidade das coordenadas válidas:")
            print(f"  Média de mudança: {np.mean(valid_diffs):.4f}")
            print(f"  Máxima mudança: {np.max(valid_diffs):.4f}")
            print(f"  Desvio padrão: {np.std(valid_diffs):.4f}")
        
        # Verificar mudanças nos valores 500
        prev_500_count = (prev_features == 500.0).sum()
        curr_500_count = (curr_features == 500.0).sum()
        if prev_500_count != curr_500_count:
            print(f"\nMUDANÇA no número de valores 500.0: {prev_500_count} -> {curr_500_count}")
    
    def capture(self, debug_mode=True):
        """Executa o processo de captura com predição em tempo real."""
        start_time = time.time()
        last_frame_time = 0
        frame_count = 0
        features_history = []  # Para análise de consistência
        
        print("A iniciar captura com predição em tempo real (MODELO 1-4).")
        print("Pressione 'q' para parar.")
        if debug_mode:
            print("\nMODO DEBUG ATIVADO:")
            print("  - Pressione 'd' para debug detalhado do frame atual")
            print("  - Pressione 'f' para debug completo das features")
            print("  - Pressione 'p' para debug da predição")
            print("  - Pressione 'c' para verificar consistência dos dados")
            print("  - Pressione 's' para mostrar estatísticas detalhadas")
        
        print(f"\nCONFIGURAÇÕES DO MODELO:")
        print(f"  - Classes de saída: 1-4")
        print(f"  - Threshold de valores nulos: {self.predictor.null_threshold*100:.0f}%")
        print(f"  - Threshold de confiança: {self.predictor.confidence_threshold*100:.0f}%")
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=PoseCaptureConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=PoseCaptureConfig.MIN_TRACKING_CONFIDENCE,
            refine_face_landmarks=False,  # Ativar landmarks da face
            model_complexity=PoseCaptureConfig.MODEL_COMPLEXITY
        ) as holistic:
            
            while self.cap.isOpened():
                current_time = time.time()
                
                # Controlo de FPS
                if current_time - last_frame_time < self.frame_interval:
                    continue
                
                success, image = self.cap.read()
                if not success:
                    continue
                
                frame_count += 1
                self.stats['total_frames'] += 1
                
                # Processar imagem com MediaPipe
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # ========== PREDIÇÃO DO MODELO ==========
                # Extrair features para o modelo
                features = LandmarkProcessor.extract_features_for_model(results, {}, debug=False)
                
                # Debug limitado (a cada 30 frames)
                if debug_mode and frame_count % 30 == 0:
                    print(f"\n--- Frame {frame_count} ---")
                    print(f"Features shape: {len(features)}")
                    # Contar valores 500.0
                    num_500s = features.count(500.0)
                    print(f"Valores 500.0: {num_500s} ({num_500s/len(features)*100:.1f}%)")
                    # Mostrar resumo das detecções
                    detections = []
                    if results.face_landmarks: detections.append("Face")
                    if results.pose_landmarks: detections.append("Pose")
                    if results.right_hand_landmarks: detections.append("MãoDir")
                    if results.left_hand_landmarks: detections.append("MãoEsq")
                    print(f"Detecções ativas: {', '.join(detections) if detections else 'Nenhuma'}")
                
                # Guardar para análise de consistência
                features_history.append(features)
                if len(features_history) > 10:
                    features_history.pop(0)
                
                # Fazer predição
                prediction = self.predictor.predict(features, debug=False)
                
                # Atualizar estatísticas
                if prediction.get('valid', False):
                    self.stats['valid_predictions'] += 1
                    pred_class = prediction['predicted_class']
                    if pred_class in self.stats['class_counts']:
                        self.stats['class_counts'][pred_class] += 1
                else:
                    reason = prediction.get('reason', 'unknown')
                    if reason == 'too_many_nulls':
                        self.stats['invalid_too_many_nulls'] += 1
                    elif reason == 'low_confidence':
                        self.stats['invalid_low_confidence'] += 1
                
                # Guardar no histórico com timestamp
                self.prediction_history.append({
                    'timestamp': current_time,
                    'prediction': prediction,
                    'num_500s': features.count(500.0)
                })
                
                # ========== VISUALIZAÇÃO ==========
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                if self.show_visualization:
                    # Desenhar landmarks
                    self._draw_landmarks(image, results)
                    
                    # Adicionar informação da predição
                    self._add_prediction_overlay(image, prediction, results)
                    
                    # Mostrar tempo decorrido
                    elapsed = int(current_time - start_time)
                    minutes = elapsed // 60
                    seconds = elapsed % 60
                    cv2.putText(
                        image, f"Tempo: {minutes:02d}:{seconds:02d}", 
                        (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
                    
                    # Mostrar imagem
                    cv2.imshow('Pose com Predicao em Tempo Real (Classes 1-4)', image)
                
                last_frame_time = current_time
                
                # Verificar teclas pressionadas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('d') and debug_mode:
                    # Debug detalhado quando 'd' é pressionado
                    print("\n" + "="*80)
                    print("DEBUG DETALHADO DO FRAME ATUAL")
                    print("="*80)
                    print(f"Frame: {frame_count}")
                    print(f"Tempo: {current_time - start_time:.2f}s")
                    self.check_data_consistency(features_history)
                    print("="*80)
                elif key == ord('f') and debug_mode:
                    # Debug completo das features
                    print("\n" + "="*80)
                    print("DEBUG COMPLETO DAS FEATURES")
                    print("="*80)
                    _ = LandmarkProcessor.extract_features_for_model(results, {}, debug=True)
                    print("="*80)
                elif key == ord('p') and debug_mode:
                    # Debug da predição
                    print("\n" + "="*80)
                    print("DEBUG DA PREDIÇÃO")
                    print("="*80)
                    _ = self.predictor.predict(features, debug=True)
                    print("="*80)
                elif key == ord('c') and debug_mode:
                    # Verificar consistência
                    print("\n" + "="*80)
                    print("ANÁLISE DE CONSISTÊNCIA")
                    print("="*80)
                    if len(self.prediction_history) > 10:
                        # Analisar últimas 10 predições
                        recent_predictions = self.prediction_history[-10:]
                        
                        # Apenas predições válidas
                        valid_predictions = [p for p in recent_predictions if p['prediction'].get('valid', False)]
                        if valid_predictions:
                            classes = [p['prediction']['predicted_class'] for p in valid_predictions]
                            unique_classes = set(classes)
                            print(f"Classes únicas nas últimas {len(valid_predictions)} predições VÁLIDAS: {unique_classes}")
                            for cls in sorted(unique_classes):
                                count = classes.count(cls)
                                print(f"  Classe {cls}: {count} vezes ({count/len(classes)*100:.1f}%)")
                        
                        # Analisar predições inválidas
                        invalid_predictions = [p for p in recent_predictions if not p['prediction'].get('valid', False)]
                        if invalid_predictions:
                            print(f"\nPredições INVÁLIDAS: {len(invalid_predictions)}")
                            reasons = [p['prediction'].get('reason', 'unknown') for p in invalid_predictions]
                            for reason in set(reasons):
                                count = reasons.count(reason)
                                print(f"  {reason}: {count} vezes")
                        
                        # Verificar variação nos valores 500
                        num_500s_history = [p['num_500s'] for p in recent_predictions]
                        print(f"\nVariação de valores 500.0:")
                        print(f"  Min: {min(num_500s_history)}, Max: {max(num_500s_history)}")
                        print(f"  Média: {np.mean(num_500s_history):.1f}")
                        
                        # Análise de confiança (apenas válidas)
                        if valid_predictions:
                            confidences = [p['prediction']['confidence'] for p in valid_predictions]
                            print(f"\nConfiança das predições VÁLIDAS:")
                            print(f"  Min: {min(confidences)*100:.1f}%, Max: {max(confidences)*100:.1f}%")
                            print(f"  Média: {np.mean(confidences)*100:.1f}%")
                    print("="*80)
                elif key == ord('s') and debug_mode:
                    # Mostrar estatísticas detalhadas
                    print("\n" + "="*80)
                    print("ESTATÍSTICAS DETALHADAS DA SESSÃO")
                    print("="*80)
                    print(f"Total de frames: {self.stats['total_frames']}")
                    print(f"Predições válidas: {self.stats['valid_predictions']} ({self.stats['valid_predictions']/self.stats['total_frames']*100:.1f}%)")
                    print(f"Rejeitadas por muitos nulos: {self.stats['invalid_too_many_nulls']}")
                    print(f"Rejeitadas por baixa confiança: {self.stats['invalid_low_confidence']}")
                    
                    print(f"\nDistribuição das classes (apenas predições válidas):")
                    for cls in range(1, 5):
                        count = self.stats['class_counts'][cls]
                        if self.stats['valid_predictions'] > 0:
                            pct = count / self.stats['valid_predictions'] * 100
                        else:
                            pct = 0
                        print(f"  Classe {cls}: {count} ({pct:.1f}%)")
                    print("="*80)
        
        self._cleanup()
    
    def _cleanup(self):
        """Limpa recursos e mostra resumo."""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar resumo das predições
        print("\n" + "="*60)
        print("RESUMO DA SESSÃO - MODELO CLASSES 1-4")
        print("="*60)
        print(f"Total de frames processados: {self.stats['total_frames']}")
        print(f"Total de predições: {len(self.prediction_history)}")
        
        if self.prediction_history:
            # Separar predições válidas e inválidas
            valid_predictions = [p for p in self.prediction_history if p['prediction'].get('valid', False)]
            invalid_predictions = [p for p in self.prediction_history if not p['prediction'].get('valid', False)]
            
            print(f"\nPREDIÇÕES VÁLIDAS: {len(valid_predictions)} ({len(valid_predictions)/len(self.prediction_history)*100:.1f}%)")
            
            if valid_predictions:
                # Contar ocorrências de cada classe
                class_counts = {}
                confidence_by_class = {}
                for item in valid_predictions:
                    class_name = item['prediction']['class_name']
                    predicted_class = item['prediction']['predicted_class']
                    confidence = item['prediction']['confidence']
                    
                    class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                    if predicted_class not in confidence_by_class:
                        confidence_by_class[predicted_class] = []
                    confidence_by_class[predicted_class].append(confidence)
                
                print("\nDistribuição das classes detectadas:")
                for pred_class in sorted(class_counts.keys()):
                    count = class_counts[pred_class]
                    percentage = (count / len(valid_predictions)) * 100
                    avg_confidence = np.mean(confidence_by_class[pred_class]) * 100
                    print(f"  Classe {pred_class}: {count} ({percentage:.1f}%) - Confiança média: {avg_confidence:.1f}%")
                
                # Análise de transições entre classes
                if len(valid_predictions) > 1:
                    transitions = 0
                    for i in range(1, len(valid_predictions)):
                        if valid_predictions[i-1]['prediction']['predicted_class'] != valid_predictions[i]['prediction']['predicted_class']:
                            transitions += 1
                    print(f"\nTransições entre classes: {transitions}")
                    print(f"Estabilidade: {(1 - transitions/len(valid_predictions))*100:.1f}%")
            
            print(f"\nPREDIÇÕES INVÁLIDAS: {len(invalid_predictions)} ({len(invalid_predictions)/len(self.prediction_history)*100:.1f}%)")
            if invalid_predictions:
                # Analisar razões das rejeições
                rejection_reasons = {}
                for item in invalid_predictions:
                    reason = item['prediction'].get('reason', 'unknown')
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                
                print("Razões das rejeições:")
                for reason, count in rejection_reasons.items():
                    percentage = (count / len(invalid_predictions)) * 100
                    if reason == 'too_many_nulls':
                        print(f"  Muitos valores ausentes: {count} ({percentage:.1f}%)")
                    elif reason == 'low_confidence':
                        print(f"  Confiança baixa: {count} ({percentage:.1f}%)")
                    else:
                        print(f"  {reason}: {count} ({percentage:.1f}%)")
            
            # Análise dos valores 500
            all_500s = [p['num_500s'] for p in self.prediction_history]
            if all_500s:
                print(f"\nAnálise de valores 500.0 (landmarks não detectados):")
                print(f"  Média: {np.mean(all_500s):.1f} valores por frame")
                print(f"  Min: {min(all_500s)}, Max: {max(all_500s)}")
                print(f"  Percentagem média: {np.mean(all_500s)/1629*100:.1f}%")
        
        # Verificar se o preprocessamento estava correto
        """if self.predictor.scaler is None:
            print("\n⚠️ AVISO: A sessão foi executada SEM os preprocessadores corretos!")
            print("   As predições podem não ter sido precisas.")
            print("   Certifique-se de fornecer os caminhos para o Scaler e Imputer.")"""
        
        print("="*60)


# =============== FUNÇÃO PRINCIPAL ===============
def main():
    """Função principal."""
    try:
        # Caminho para o modelo treinado com coordenadas apenas (sem visibilidade)
        #MODEL_PATH = "data/output2/trained_model_4classes_corrected.pth"
        MODEL_PATH = "../data/output28/trained_model_coordinates_only.pth"
        SCALER_PATH = None  # Não usado neste modelo
        IMPUTER_PATH = None  # Não usado neste modelo

        print("\n" + "="*60)
        print("SISTEMA DE CAPTURA COM PREDIÇÃO - MODELO COORDENADAS ONLY (1-4)")
        print("="*60)
        print(f"Modelo: {MODEL_PATH}")
        print("Scaler: [NÃO USADO]")
        print("Imputer: [NÃO USADO]")
        print("="*60)

        # Verifica se o ficheiro do modelo existe
        if not os.path.exists(MODEL_PATH):
            print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
            return

        print("\n✅ Modelo encontrado!")

        # Criar sistema de captura com predição sem scaler/imputer
        capture_system = VideoCaptureWithPrediction(
            model_path=MODEL_PATH,
            #scaler_path=SCALER_PATH,
            #imputer_path=IMPUTER_PATH,
            show_visualization=True,
            target_fps=30.0,
            null_threshold=0.95,
            confidence_threshold=0.3
        )

        # Iniciar captura com debug ativado
        capture_system.capture(debug_mode=True)

    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo utilizador.")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("A encerrar programa...")

if __name__ == "__main__":
    main()
