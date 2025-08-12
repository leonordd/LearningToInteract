# 3realtime_prediction.py
# Sistema de Predições em Tempo Real com MediaPipe
# Carrega modelo treinado e faz inferências em tempo real

"""
Sistema de Predições em Tempo Real
Versão baseada no sistema de captura com predições de IA

Autor: Sistema de Predições
Data: 2025
Funcionalidades:
- Carregamento do modelo treinado
- Captura de landmarks em tempo real
- Predições instantâneas
- Visualização da predição atual
- Modo de debug com confiança
"""

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


# ===========================================================================================
# CONFIGURAÇÕES GLOBAIS
# ===========================================================================================

class PredictionConfig:
    """Configurações centralizadas do sistema de predição."""
    
    # Câmara
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 60
    
    # MediaPipe
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 0  # 0=rápido, 2=preciso
    
    # Landmarks
    NUM_POSE_LANDMARKS = 33
    NUM_HAND_LANDMARKS = 21
    NUM_FACE_LANDMARKS = 468
    DEFAULT_LANDMARK_VALUE = 0.0  # Valor para landmarks não detectados
    
    # Visualização
    SQUARE_SIZE = 3
    FACE_SQUARE_SIZE = 1
    SQUARE_COLOR = (255, 255, 255)  # Branco (BGR)
    
    # Predições
    CONFIDENCE_THRESHOLD = 0.3  # Limiar mínimo para mostrar predição
    SMOOTHING_WINDOW = 5  # Janela para suavização de predições
    PREDICTION_DISPLAY_TIME = 2000  # ms para mostrar predição
    
    # Debug
    SHOW_DEBUG_INFO = True
    FPS_UPDATE_INTERVAL = 30  # Atualizar FPS a cada 30 frames


# ===========================================================================================
# MODELO DE REDE NEURAL (DEVE SER IDÊNTICO AO TREINO)
# ===========================================================================================

class WeightedFlexibleModel(torch.nn.Module):
    """Modelo de rede neural flexível com pesos por classe."""
    
    def __init__(self, input_size, output_size, hidden_layers, class_weights=None):
        super().__init__()
        
        self.class_weights = class_weights
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Última camada sem ativação
        logits = self.layers[-1](x)
        return logits
    
    def predict_with_confidence(self, x):
        """Predição com cálculo de confiança."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            return predictions, confidences, probs


# ===========================================================================================
# CARREGADOR DE MODELO
# ===========================================================================================

class ModelLoader:
    """Classe para carregar o modelo treinado e suas configurações."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.model_info = None
        self.feature_names = None
        self.device = self._get_device()
        
        self._load_model()
    
    def _get_device(self):
        """Determina o dispositivo disponível."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Carrega o modelo e suas informações."""
        try:
            print(f"[MODEL] Carregando modelo de: {self.model_path}")
            
            # Carregar informações do modelo
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model_info = checkpoint
            
            # Extrair configurações
            input_size = checkpoint['input_size']
            output_size = checkpoint['output_size']
            hidden_layers = checkpoint['hidden_layers']
            
            # Criar modelo
            self.model = WeightedFlexibleModel(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=hidden_layers
            ).to(self.device)
            
            # Carregar pesos
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Extrair informações de features
            if 'feature_info' in checkpoint:
                self.feature_names = checkpoint['feature_info'].get('feature_order', [])
            
            print(f"[MODEL] ✅ Modelo carregado com sucesso!")
            print(f"[MODEL] Device: {self.device}")
            print(f"[MODEL] Input size: {input_size}")
            print(f"[MODEL] Output size: {output_size}")
            print(f"[MODEL] Hidden layers: {hidden_layers}")
            print(f"[MODEL] Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
            
            # Mostrar métricas se disponíveis
            if 'best_accuracy' in checkpoint:
                print(f"[MODEL] Accuracy treino: {checkpoint['best_accuracy']:.2f}%")
            
        except Exception as e:
            print(f"[MODEL] ❌ Erro ao carregar modelo: {e}")
            raise
    
    def get_model_info(self):
        """Retorna informações do modelo."""
        return {
            'input_size': self.model_info['input_size'],
            'output_size': self.model_info['output_size'],
            'feature_names': self.feature_names,
            'device': self.device,
            'preprocessing': self.model_info.get('preprocessing_details', {}),
            'best_accuracy': self.model_info.get('best_accuracy', 'N/A')
        }


# ===========================================================================================
# PROCESSADOR DE LANDMARKS PARA PREDIÇÃO
# ===========================================================================================

class LandmarkPreprocessor:
    """Processa landmarks do MediaPipe para formato de predição."""
    
    @staticmethod
    def extract_coordinates(landmarks, expected_count: int) -> List[float]:
        """Extrai coordenadas x,y,z de landmarks."""
        if landmarks:
            return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        else:
            return [PredictionConfig.DEFAULT_LANDMARK_VALUE] * (expected_count * 3)
    
    @staticmethod
    def create_feature_vector(results) -> np.ndarray:
        """Cria vetor de features a partir dos resultados do MediaPipe."""
        
        # Extrair coordenadas (mesma ordem que no treino)
        pose_coords = LandmarkPreprocessor.extract_coordinates(
            results.pose_landmarks.landmark if results.pose_landmarks else None,
            PredictionConfig.NUM_POSE_LANDMARKS
        )
        
        left_hand_coords = LandmarkPreprocessor.extract_coordinates(
            results.left_hand_landmarks.landmark if results.left_hand_landmarks else None,
            PredictionConfig.NUM_HAND_LANDMARKS
        )
        
        right_hand_coords = LandmarkPreprocessor.extract_coordinates(
            results.right_hand_landmarks.landmark if results.right_hand_landmarks else None,
            PredictionConfig.NUM_HAND_LANDMARKS
        )
        
        face_coords = LandmarkPreprocessor.extract_coordinates(
            results.face_landmarks.landmark if results.face_landmarks else None,
            PredictionConfig.NUM_FACE_LANDMARKS
        )
        
        # Combinar todas as coordenadas (mesma ordem que no treino)
        feature_vector = pose_coords + left_hand_coords + right_hand_coords + face_coords
        
        return np.array(feature_vector, dtype=np.float32)
    
    @staticmethod
    def get_detection_status(results) -> Dict[str, bool]:
        """Obtém status de detecção de landmarks."""
        return {
            "face": results.face_landmarks is not None,
            "pose": results.pose_landmarks is not None,
            "left_hand": results.left_hand_landmarks is not None,
            "right_hand": results.right_hand_landmarks is not None
        }


# ===========================================================================================
# SUAVIZAÇÃO DE PREDIÇÕES
# ===========================================================================================

class PredictionSmoother:
    """Classe para suavizar predições ao longo do tempo."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.prediction_history = []
        self.confidence_history = []
    
    def add_prediction(self, prediction: int, confidence: float):
        """Adiciona nova predição ao histórico."""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Manter apenas as últimas N predições
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_smoothed_prediction(self) -> Tuple[int, float]:
        """Retorna predição suavizada baseada no histórico."""
        if not self.prediction_history:
            return -1, 0.0
        
        # Usar moda para predição (mais frequente)
        unique_predictions, counts = np.unique(self.prediction_history, return_counts=True)
        most_common_idx = np.argmax(counts)
        smoothed_prediction = unique_predictions[most_common_idx]
        
        # Média das confianças para essa predição
        matching_confidences = [conf for pred, conf in zip(self.prediction_history, self.confidence_history) 
                               if pred == smoothed_prediction]
        smoothed_confidence = np.mean(matching_confidences) if matching_confidences else 0.0
        
        return int(smoothed_prediction), float(smoothed_confidence)
    
    def get_stability_score(self) -> float:
        """Retorna score de estabilidade das predições (0-1)."""
        if len(self.prediction_history) < 2:
            return 0.0
        
        # Calcular quantas predições são iguais à mais recente
        recent_prediction = self.prediction_history[-1]
        matches = sum(1 for pred in self.prediction_history if pred == recent_prediction)
        
        return matches / len(self.prediction_history)


# ===========================================================================================
# VISUALIZADOR PARA PREDIÇÕES
# ===========================================================================================

class PredictionVisualizer:
    """Classe para visualizar predições e informações na tela."""
    
    @staticmethod
    def draw_landmarks(image: np.ndarray, results):
        """Desenha landmarks como quadrados brancos."""
        h, w = image.shape[:2]
        
        # Landmarks do rosto (pequenos)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.rectangle(image, 
                            (x - PredictionConfig.FACE_SQUARE_SIZE, y - PredictionConfig.FACE_SQUARE_SIZE),
                            (x + PredictionConfig.FACE_SQUARE_SIZE, y + PredictionConfig.FACE_SQUARE_SIZE),
                            PredictionConfig.SQUARE_COLOR, -1)
        
        # Landmarks da pose, mãos (normais)
        for landmarks in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if landmarks:
                for landmark in landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.rectangle(image,
                                (x - PredictionConfig.SQUARE_SIZE, y - PredictionConfig.SQUARE_SIZE),
                                (x + PredictionConfig.SQUARE_SIZE, y + PredictionConfig.SQUARE_SIZE),
                                PredictionConfig.SQUARE_COLOR, -1)
    
    @staticmethod
    def add_prediction_overlay(image: np.ndarray, prediction: int, confidence: float, 
                             smoothed_prediction: int, smoothed_confidence: float,
                             stability: float, fps: float, detection_status: Dict[str, bool]):
        """Adiciona overlay com informações de predição."""
        h, w = image.shape[:2]
        
        # Cor baseada na confiança
        if smoothed_confidence > 0.7:
            confidence_color = (0, 255, 0)  # Verde
        elif smoothed_confidence > 0.4:
            confidence_color = (0, 255, 255)  # Amarelo
        else:
            confidence_color = (0, 0, 255)  # Vermelho
        
        # Criar overlay escuro para melhor legibilidade
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Predição principal (grande)
        main_text = f"Animacao: {smoothed_prediction + 1}"  # +1 para display (1-4 em vez de 0-3)
        cv2.putText(image, main_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, confidence_color, 2)
        
        # Confiança
        conf_text = f"Confianca: {smoothed_confidence:.2f}"
        cv2.putText(image, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
        
        # Estabilidade
        stability_text = f"Estabilidade: {stability:.2f}"
        cv2.putText(image, stability_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Status de detecção (canto superior direito)
        status_x = w - 200
        cv2.putText(image, "Deteccao:", (status_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status_symbols = []
        if detection_status["face"]: status_symbols.append("F")
        if detection_status["pose"]: status_symbols.append("P")
        if detection_status["left_hand"]: status_symbols.append("LH")
        if detection_status["right_hand"]: status_symbols.append("RH")
        
        status_text = " ".join(status_symbols) if status_symbols else "NENHUMA"
        status_color = (0, 255, 0) if len(status_symbols) >= 2 else (0, 255, 255) if status_symbols else (0, 0, 255)
        cv2.putText(image, status_text, (status_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # FPS (canto inferior direito)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Debug info (se ativado)
        if PredictionConfig.SHOW_DEBUG_INFO:
            debug_text = f"Raw: {prediction + 1} ({confidence:.2f})"  # +1 para display
            cv2.putText(image, debug_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ===========================================================================================
# SISTEMA PRINCIPAL DE PREDIÇÃO
# ===========================================================================================

class RealtimePredictionSystem:
    """Sistema principal de predições em tempo real."""
    
    def __init__(self, model_path: str):
        print("🎯 Inicializando Sistema de Predições em Tempo Real")
        print("=" * 60)
        
        # Carregar modelo
        self.model_loader = ModelLoader(model_path)
        self.model = self.model_loader.model
        self.device = self.model_loader.device
        
        # Componentes do sistema
        self.preprocessor = LandmarkPreprocessor()
        self.smoother = PredictionSmoother(window_size=PredictionConfig.SMOOTHING_WINDOW)
        self.visualizer = PredictionVisualizer()
        
        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        
        # Câmara
        self.cap = self._setup_camera()
        
        # Estatísticas
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = 0
        self.current_fps = 0
        
        self._print_initialization_summary()
    
    def _setup_camera(self) -> cv2.VideoCapture:
        """Configura webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PredictionConfig.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PredictionConfig.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, PredictionConfig.CAMERA_FPS)
        
        if not cap.isOpened():
            raise RuntimeError("Erro: Não foi possível aceder à webcam!")
        
        return cap
    
    def _print_initialization_summary(self):
        """Imprime resumo da inicialização."""
        model_info = self.model_loader.get_model_info()
        
        print(f"[INIT] ✅ Sistema inicializado com sucesso!")
        print(f"[INIT] Modelo: {model_info['input_size']} → {model_info['output_size']} classes")
        print(f"[INIT] Device: {model_info['device']}")
        print(f"[INIT] Accuracy treino: {model_info['best_accuracy']}")
        print(f"[INIT] Suavização: janela de {PredictionConfig.SMOOTHING_WINDOW} frames")
        print(f"[INIT] Limiar confiança: {PredictionConfig.CONFIDENCE_THRESHOLD}")
    
    def _update_fps(self):
        """Atualiza cálculo de FPS."""
        if self.frame_count % PredictionConfig.FPS_UPDATE_INTERVAL == 0:
            current_time = time.time()
            if self.last_fps_update > 0:
                time_diff = current_time - self.last_fps_update
                self.current_fps = PredictionConfig.FPS_UPDATE_INTERVAL / time_diff
            self.last_fps_update = current_time
    
    def run(self):
        """Executa sistema de predições em tempo real."""
        print(f"\n[PREDICTION] 🎬 Iniciando predições em tempo real")
        print("[PREDICTION] Pressione 'q' para sair, 's' para capturar screenshot")
        
        cv2.namedWindow('Predicoes em Tempo Real', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Predicoes em Tempo Real', PredictionConfig.CAMERA_WIDTH, PredictionConfig.CAMERA_HEIGHT)
        
        try:
            with self.mp_holistic.Holistic(
                min_detection_confidence=PredictionConfig.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=PredictionConfig.MIN_TRACKING_CONFIDENCE,
                refine_face_landmarks=False,
                model_complexity=PredictionConfig.MODEL_COMPLEXITY
            ) as holistic:
                
                while self.cap.isOpened():
                    success, image = self.cap.read()
                    if not success:
                        continue
                    
                    self.frame_count += 1
                    self._update_fps()
                    
                    # Processar MediaPipe
                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    # Obter status de detecção
                    detection_status = self.preprocessor.get_detection_status(results)
                    
                    # Fazer predição
                    prediction, confidence = self._make_prediction(results)
                    
                    # Adicionar à suavização
                    if prediction >= 0:  # Predição válida
                        self.smoother.add_prediction(prediction, confidence)
                    
                    # Obter predição suavizada
                    smoothed_pred, smoothed_conf = self.smoother.get_smoothed_prediction()
                    stability = self.smoother.get_stability_score()
                    
                    # Visualização
                    image.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Desenhar landmarks
                    self.visualizer.draw_landmarks(image, results)
                    
                    # Adicionar overlay de predição
                    display_pred = prediction if prediction >= 0 else -1
                    display_smoothed = smoothed_pred if smoothed_pred >= 0 else -1
                    
                    self.visualizer.add_prediction_overlay(
                        image, display_pred, confidence,
                        display_smoothed, smoothed_conf, stability,
                        self.current_fps, detection_status
                    )
                    
                    # Mostrar frame
                    cv2.imshow('Predicoes em Tempo Real', image)
                    
                    # Debug periódico
                    if self.frame_count % 60 == 0:  # A cada ~2 segundos
                        self._print_debug_info(detection_status, prediction, confidence, 
                                             smoothed_pred, smoothed_conf, stability)
                    
                    # Verificar teclas
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[PREDICTION] 🛑 Saindo...")
                        break
                    elif key == ord('s'):
                        self._save_screenshot(image)
        
        finally:
            self._cleanup()
    
    def _make_prediction(self, results) -> Tuple[int, float]:
        """Faz predição baseada nos landmarks detectados."""
        try:
            # Extrair features
            feature_vector = self.preprocessor.create_feature_vector(results)
            
            # Verificar se há landmarks suficientes
            non_zero_features = np.count_nonzero(feature_vector)
            total_features = len(feature_vector)
            detection_ratio = non_zero_features / total_features
            
            # Se muito poucas detecções, não fazer predição
            if detection_ratio < 0.1:  # Menos de 10% de landmarks detectados
                return -1, 0.0
            
            # Converter para tensor
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Fazer predição
            predictions, confidences, probs = self.model.predict_with_confidence(feature_tensor)
            
            prediction = predictions[0].item()
            confidence = confidences[0].item()
            
            # Aplicar limiar de confiança
            if confidence < PredictionConfig.CONFIDENCE_THRESHOLD:
                return -1, confidence
            
            return prediction, confidence
            
        except Exception as e:
            print(f"[ERROR] Erro na predição: {e}")
            return -1, 0.0
    
    def _print_debug_info(self, detection_status, prediction, confidence, 
                         smoothed_pred, smoothed_conf, stability):
        """Imprime informações de debug."""
        detected = [k for k, v in detection_status.items() if v]
        detected_str = "+".join(detected) if detected else "NONE"
        
        print(f"[DEBUG] Frame {self.frame_count}: "
              f"Det={detected_str} | "
              f"Pred={prediction + 1 if prediction >= 0 else 'N/A'}({confidence:.2f}) | "
              f"Smooth={smoothed_pred + 1 if smoothed_pred >= 0 else 'N/A'}({smoothed_conf:.2f}) | "
              f"Stability={stability:.2f} | FPS={self.current_fps:.1f}")
    
    def _save_screenshot(self, image):
        """Salva screenshot com timestamp."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_screenshot_{timestamp}.png"
        cv2.imwrite(filename, image)
        print(f"[SCREENSHOT] Salvo: {filename}")
    
    def _cleanup(self):
        """Limpa recursos."""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        print(f"\n[FINAL] 📊 Estatísticas da sessão:")
        print(f"[FINAL] Frames processados: {self.frame_count}")
        print(f"[FINAL] Tempo total: {total_time:.1f}s")
        print(f"[FINAL] FPS médio: {avg_fps:.1f}")
        print(f"[FINAL] 🔚 Sistema encerrado")


# ===========================================================================================
# FUNÇÃO PRINCIPAL
# ===========================================================================================

def main():
    """Função principal do programa."""
    try:
        print("🎯 Sistema de Predições em Tempo Real")
        print("=" * 60)
        
        # Caminho do modelo (ajustar conforme necessário)
        model_path = Path(__file__).resolve().parent / ".." / "data" / "dataset30" / "output30" / "trained_model_coordinates_only.pth"
        
        # Verificar se modelo existe
        if not model_path.exists():
            print(f"❌ Modelo não encontrado em: {model_path}")
            print("💡 Certifique-se de que o modelo foi treinado e salvo corretamente.")
            return
        
        print(f"[MAIN] Modelo encontrado: {model_path}")
        
        # Mostrar funcionalidades
        print(f"\n🎯 Funcionalidades Ativas:")
        print(f"⚡ - Predições em tempo real")
        print(f"🎬 - Captura de pose com MediaPipe")
        print(f"📊 - Suavização de predições")
        print(f"🔍 - Visualização de confiança")
        print(f"🎯 - Detecção de estabilidade")
        
        # Inicializar sistema
        print(f"\n[MAIN] Iniciando em 3 segundos...")
        time.sleep(3)
        
        system = RealtimePredictionSystem(str(model_path))
        system.run()
        
    except KeyboardInterrupt:
        print("\n[MAIN] 🛑 Interrompido pelo utilizador")
    except Exception as e:
        print(f"\n[ERROR] ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[MAIN] 🔚 Programa terminado")


if __name__ == "__main__":
    main()