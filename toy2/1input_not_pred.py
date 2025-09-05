# 3realtime_prediction_sockets.py
# Sistema de Predições em Tempo Real com MediaPipe + envio por sockets para Processing

"""
Sistema de Predições em Tempo Real
Versão robusta com carregamento seguro de checkpoints (PyTorch >= 2.6)
e servidor TCP que abre antes do load do modelo (evita "Connection refused").

Autor: Sistema de Predições
Data: 2025

Funcionalidades:
- Carregamento seguro do modelo treinado (compatível com "weights_only")
- Captura de landmarks em tempo real (MediaPipe Holistic)
- Predições com suavização
- Visualização overlay (OpenCV)
- Servidor TCP que envia a animação atual para um cliente (Processing)
- Fallback: se o modelo falhar, servidor continua vivo e envia 0
"""

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
import socket
import threading

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
    DEFAULT_LANDMARK_VALUE = 0.0  # Valor para landmarks não detetados

    # Visualização
    SQUARE_SIZE = 3
    FACE_SQUARE_SIZE = 1
    SQUARE_COLOR = (255, 255, 255)  # Branco (BGR)

    # Predições
    CONFIDENCE_THRESHOLD = 0.3  # Limiar mínimo para enviar animação != 0
    SMOOTHING_WINDOW = 5
    FPS_UPDATE_INTERVAL = 30

    # Rede
    HOST = '127.0.0.1'
    PORT = 60345
    SEND_HZ = 20.0  # taxa máxima de envio


# ===========================================================================================
# REDE: Servidor de sockets
# ===========================================================================================

class PredictionSocketServer:
    """Servidor TCP que envia a animação atual para um cliente (Processing)."""
    def __init__(self, host='127.0.0.1', port=60345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_conn = None
        self.client_addr = None
        self.running = False
        self.lock = threading.Lock()
        self.accept_thread = None

    def start(self):
        if self.running:
            return
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True

        def _accept():
            print(f"[NET] 📡 A ouvir em {self.host}:{self.port} (aguardar ligação do Processing)")
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    with self.lock:
                        if self.client_conn:
                            try:
                                self.client_conn.close()
                            except:
                                pass
                        self.client_conn, self.client_addr = conn, addr
                    print(f"[NET] ✅ Cliente ligado: {addr}")
                except OSError:
                    break

        self.accept_thread = threading.Thread(target=_accept, daemon=True)
        self.accept_thread.start()

    def send_line(self, text: str):
        """Envia uma linha terminada em '\\n'. Ignora se não houver cliente ligado."""
        with self.lock:
            conn = self.client_conn
        if not conn:
            return
        try:
            conn.sendall((text + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[NET] ⚠️ Ligação perdida; a aguardar novo cliente…")
            with self.lock:
                try:
                    conn.close()
                except:
                    pass
                self.client_conn = None
                self.client_addr = None

    def stop(self):
        self.running = False
        try:
            if self.server_socket:
                self.server_socket.close()
        except:
            pass
        with self.lock:
            if self.client_conn:
                try:
                    self.client_conn.close()
                except:
                    pass
            self.client_conn = None


# ===========================================================================================
# MODELO DE REDE NEURAL (DEVE SER IDÊNTICO AO TREINO)
# ===========================================================================================

class WeightedFlexibleModel(torch.nn.Module):
    """Modelo de rede neural flexível (idêntico ao usado no treino)."""
    def __init__(self, input_size, output_size, hidden_layers, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        logits = self.layers[-1](x)
        return logits

    def predict_with_confidence(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            return predictions, confidences, probs


# ===========================================================================================
# CARREGAMENTO SEGURO DO MODELO (PyTorch >= 2.6)
# ===========================================================================================

def load_state_dict_safely(path: Path, device: str):
    """Carrega o checkpoint aceitando formatos comuns e PyTorch>=2.6."""
    import numpy as np
    # permitir numpy scalar (erro: numpy.core.multiarray.scalar)
    try:
        torch.serialization.add_safe_globals([np.generic])
    except Exception:
        pass

    # 1) tentar com weights_only=True (seguro)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        print("[MODEL] weights_only=True falhou ->", e1)
        print("[MODEL] A tentar weights_only=False (usa apenas se confias no .pth)")
        # 2) fallback (atenção: pode executar código arbitrário durante unpickle!)
        ckpt = torch.load(path, map_location=device, weights_only=False)

    # normalizar para state_dict
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "net", "model"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v, ckpt
    # pode já ser o próprio state_dict
    return ckpt, {"input_size": None, "output_size": None, "hidden_layers": []}


class ModelLoader:
    """Carrega o modelo + info auxiliar a partir do checkpoint."""
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = None
        self.feature_names = None
        self.device = self._get_device()
        self._load_model()

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        print(f"[MODEL] Carregando modelo de: {self.model_path}")
        state_dict, raw_ckpt = load_state_dict_safely(self.model_path, self.device)

        # recuperar meta se existir
        input_size = raw_ckpt.get("input_size")
        output_size = raw_ckpt.get("output_size")
        hidden_layers = raw_ckpt.get("hidden_layers") or [256, 128]

        if input_size is None or output_size is None:
            # heurística: inferir pelo tamanho do state_dict última camada
            # (melhor é ter guardado no treino)
            # aqui só avisamos
            print("[MODEL] Aviso: input/output sizes não presentes no checkpoint; a usar defaults.")
            # tentar deduzir por chaves comuns
            last_w = None
            for k, v in state_dict.items():
                if k.endswith(".weight"):
                    last_w = v
            if last_w is not None and last_w.ndim == 2:
                output_size = last_w.shape[0]
            input_size = input_size or 33*3 + 21*3 + 21*3 + 468*3  # pose+hands+face coords

        self.model = WeightedFlexibleModel(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers
        ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[MODEL] Aviso - missing keys:", missing)
        if unexpected:
            print("[MODEL] Aviso - unexpected keys:", unexpected)

        self.model.eval()
        self.model_info = raw_ckpt
        self.feature_names = (raw_ckpt.get("feature_info", {}) or {}).get("feature_order", [])
        print(f"[MODEL] ✅ Modelo carregado. Device={self.device}  Input={input_size}  Output={output_size}  Hidden={hidden_layers}")


# ===========================================================================================
# PROCESSADOR DE LANDMARKS
# ===========================================================================================

class LandmarkPreprocessor:
    """Processa landmarks do MediaPipe para formato de predição."""
    @staticmethod
    def extract_coordinates(landmarks, expected_count: int) -> List[float]:
        if landmarks:
            return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        else:
            return [PredictionConfig.DEFAULT_LANDMARK_VALUE] * (expected_count * 3)

    @staticmethod
    def create_feature_vector(results) -> np.ndarray:
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
        feature_vector = pose_coords + left_hand_coords + right_hand_coords + face_coords
        return np.array(feature_vector, dtype=np.float32)

    @staticmethod
    def get_detection_status(results) -> Dict[str, bool]:
        return {
            "face": results.face_landmarks is not None,
            "pose": results.pose_landmarks is not None,
            "left_hand": results.left_hand_landmarks is not None,
            "right_hand": results.right_hand_landmarks is not None
        }


# ===========================================================================================
# SUAVIZAÇÃO
# ===========================================================================================

class PredictionSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.prediction_history = []
        self.confidence_history = []

    def add_prediction(self, prediction: int, confidence: float):
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)

    def get_smoothed_prediction(self) -> Tuple[int, float]:
        if not self.prediction_history:
            return -1, 0.0
        unique_predictions, counts = np.unique(self.prediction_history, return_counts=True)
        most_common_idx = int(np.argmax(counts))
        smoothed_prediction = int(unique_predictions[most_common_idx])
        matching_conf = [c for p, c in zip(self.prediction_history, self.confidence_history) if p == smoothed_prediction]
        smoothed_confidence = float(np.mean(matching_conf)) if matching_conf else 0.0
        return smoothed_prediction, smoothed_confidence

    def get_stability_score(self) -> float:
        if len(self.prediction_history) < 2:
            return 0.0
        recent = self.prediction_history[-1]
        matches = sum(1 for p in self.prediction_history if p == recent)
        return matches / len(self.prediction_history)


# ===========================================================================================
# VISUALIZAÇÃO
# ===========================================================================================

class PredictionVisualizer:
    @staticmethod
    def draw_landmarks(image: np.ndarray, results):
        h, w = image.shape[:2]
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.rectangle(image,
                              (x - PredictionConfig.FACE_SQUARE_SIZE, y - PredictionConfig.FACE_SQUARE_SIZE),
                              (x + PredictionConfig.FACE_SQUARE_SIZE, y + PredictionConfig.FACE_SQUARE_SIZE),
                              PredictionConfig.SQUARE_COLOR, -1)
        for landmarks in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.rectangle(image,
                                  (x - PredictionConfig.SQUARE_SIZE, y - PredictionConfig.SQUARE_SIZE),
                                  (x + PredictionConfig.SQUARE_SIZE, y + PredictionConfig.SQUARE_SIZE),
                                  PredictionConfig.SQUARE_COLOR, -1)

    @staticmethod
    def add_prediction_overlay(image: np.ndarray, prediction: int, confidence: float,
                               smoothed_prediction: int, smoothed_confidence: float,
                               stability: float, fps: float, detection_status: Dict[str, bool]):
        h, w = image.shape[:2]
        if smoothed_confidence > 0.7:
            confidence_color = (0, 255, 0)
        elif smoothed_confidence > 0.4:
            confidence_color = (0, 255, 255)
        else:
            confidence_color = (0, 0, 255)

        overlay = image.copy()
        cv2.rectangle(overlay, (0, image.shape[0] - 30), (300, image.shape[0] ), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        main_text = f"Anim: {smoothed_prediction + 1 if smoothed_prediction >= 0 else 0}"
        cv2.putText(image, main_text, (15, image.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, confidence_color, 2)

        conf_text = f"Confianca: {smoothed_confidence:.2f}"
        cv2.putText(image, conf_text, (100, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 2)

        #stability_text = f"Estabilidade: {stability:.2f}"
        #cv2.putText(image, stability_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        status_x = w - 200
        #cv2.putText(image, "Deteccao:", (status_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        flags = []
        if detection_status["face"]: flags.append("F")
        if detection_status["pose"]: flags.append("P")
        if detection_status["left_hand"]: flags.append("LH")
        if detection_status["right_hand"]: flags.append("RH")
        status_text = " ".join(flags) if flags else "NENHUMA"
        status_color = (0, 255, 0) if len(flags) >= 2 else (0, 255, 255) if flags else (0, 0, 255)
        #cv2.putText(image, status_text, (status_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        fps_text = f"FPS: {fps:.1f}"
        #cv2.putText(image, fps_text, (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ===========================================================================================
# SISTEMA PRINCIPAL
# ===========================================================================================

class RealtimePredictionSystem:
    """Sistema principal de predições em tempo real."""
    def __init__(self, model_path: str, net: PredictionSocketServer):
        print("🎯 Inicializando Sistema de Predições em Tempo Real")
        print("=" * 60)
        self.net = net  # servidor já started no main

        # Carregar modelo (com proteção)
        self.model = None
        self.device = "cpu"
        try:
            loader = ModelLoader(model_path)
            self.model = loader.model
            self.device = loader.device
        except Exception as e:
            print("[MODEL] ❌ Erro ao carregar modelo:", e)
            print("[MODEL] Fallback: servidor continua ativo e envia 0. (Pressiona Ctrl+C para sair)")

        self.preprocessor = LandmarkPreprocessor()
        self.smoother = PredictionSmoother(window_size=PredictionConfig.SMOOTHING_WINDOW)
        self.visualizer = PredictionVisualizer()

        self.mp_holistic = mp.solutions.holistic

        # Câmara
        self.cap = self._setup_camera()

        # Envio
        self._last_sent_anim = None
        self._last_send_time = 0.0
        self._send_interval = 1.0 / PredictionConfig.SEND_HZ

        # Métricas
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = 0
        self.current_fps = 0

        self._print_initialization_summary()

    def _setup_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PredictionConfig.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PredictionConfig.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, PredictionConfig.CAMERA_FPS)
        if not cap.isOpened():
            raise RuntimeError("Erro: Não foi possível aceder à webcam!")
        return cap

    def _print_initialization_summary(self):
        print(f"[INIT] ✅ Sistema inicializado (modelo={'OK' if self.model else 'N/A'})")
        print(f"[INIT] Socket em {PredictionConfig.HOST}:{PredictionConfig.PORT} (aguardar cliente)")
        print(f"[INIT] Suavização: janela={PredictionConfig.SMOOTHING_WINDOW}, Limiar={PredictionConfig.CONFIDENCE_THRESHOLD}")

    def _update_fps(self):
        if self.frame_count % PredictionConfig.FPS_UPDATE_INTERVAL == 0:
            current_time = time.time()
            if self.last_fps_update > 0:
                dt = current_time - self.last_fps_update
                if dt > 0:
                    self.current_fps = PredictionConfig.FPS_UPDATE_INTERVAL / dt
            self.last_fps_update = current_time

    def run(self):
        print("\n[PREDICTION] 🎬 Iniciando predições em tempo real")
        print("[PREDICTION] Pressione 'q' para sair, 's' para screenshot")

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

                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    detection_status = self.preprocessor.get_detection_status(results)

                    # Predição (se houver modelo)
                    prediction, confidence = self._make_prediction(results) if self.model else (-1, 0.0)
                    if prediction >= 0:
                        self.smoother.add_prediction(prediction, confidence)

                    smoothed_pred, smoothed_conf = self.smoother.get_smoothed_prediction()
                    stability = self.smoother.get_stability_score()

                    # --- Envio por socket ---
                    if smoothed_pred >= 0 and smoothed_conf >= PredictionConfig.CONFIDENCE_THRESHOLD:
                        anim_to_send = int(smoothed_pred) + 1  # 1..N
                    else:
                        anim_to_send = 0  # "idle/sem predição"

                    now = time.time()
                    if (anim_to_send != self._last_sent_anim) or (now - self._last_send_time >= self._send_interval):
                        self.net.send_line(str(anim_to_send))  # ou JSON
                        self._last_sent_anim = anim_to_send
                        self._last_send_time = now

                    # --- Visualização ---
                    image.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    self.visualizer.draw_landmarks(image, results)
                    display_pred = prediction if prediction >= 0 else -1
                    display_smoothed = smoothed_pred if smoothed_pred >= 0 else -1
                    self.visualizer.add_prediction_overlay(
                        image, display_pred, confidence,
                        display_smoothed, smoothed_conf, stability,
                        self.current_fps, detection_status
                    )
                    cv2.imshow('Predicoes em Tempo Real', image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[PREDICTION] 🛑 Saindo...")
                        break
                    elif key == ord('s'):
                        self._save_screenshot(image)

        finally:
            self._cleanup()

    def _make_prediction(self, results) -> Tuple[int, float]:
        try:
            feature_vector = self.preprocessor.create_feature_vector(results)
            # heuriśtica simples para casos sem deteção
            if np.count_nonzero(feature_vector) / len(feature_vector) < 0.1:
                return -1, 0.0
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            predictions, confidences, _ = self.model.predict_with_confidence(feature_tensor)
            p = int(predictions[0].item())
            c = float(confidences[0].item())
            if c < PredictionConfig.CONFIDENCE_THRESHOLD:
                return -1, c
            return p, c
        except Exception as e:
            print(f"[ERROR] Erro na predição: {e}")
            return -1, 0.0

    def _save_screenshot(self, image):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_screenshot_{timestamp}.png"
        cv2.imwrite(filename, image)
        print(f"[SCREENSHOT] Salvo: {filename}")

    def _cleanup(self):
        try:
            self.cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            self.net.stop()
        except:
            pass

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
    print("🎯 Sistema de Predições em Tempo Real")
    print("=" * 60)

    # Caminhos a alterar em cada teste
    dataset_folder = "dataset48"

    #Caminhos fixos
    #csv_file = "combinado.csv"  # Ficheiro combinado
    output_folder = "output"  # Pasta de saída para resultados

    # Caminho do modelo (ajusta consoante a tua estrutura)
    model_path = Path(__file__).resolve().parent / ".." / "data" / dataset_folder / output_folder / "trained_model.pth"

    if not model_path.exists():
        print(f"❌ Modelo não encontrado em: {model_path}")
        print("💡 O servidor TCP vai iniciar na mesma e enviar 0 (útil para testar a ligação).")

    # 1) Abre o servidor primeiro (Processing pode conectar já aqui)
    net = PredictionSocketServer(host=PredictionConfig.HOST, port=PredictionConfig.PORT)
    net.start()

    # 2) Se não existir modelo, fica num loop a enviar 0 até fechares
    if not model_path.exists():
        try:
            while True:
                net.send_line("0")
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            net.stop()
        return

    # 3) Inicializa o sistema completo (load modelo + camera + mediapipe)
    try:
        print(f"[MAIN] Modelo alvo: {model_path}")
        print("[MAIN] A iniciar sistema...")
        system = RealtimePredictionSystem(str(model_path), net)
        system.run()
    except KeyboardInterrupt:
        print("\n[MAIN] 🛑 Interrompido pelo utilizador")
    except Exception as e:
        print(f"\n[ERROR] ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        print("[MAIN] Fallback: a enviar 0 continuamente (servidor ativo). Ctrl+C para sair.")
        try:
            while True:
                net.send_line("0")
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    finally:
        try:
            net.stop()
        except:
            pass
        print("[MAIN] 🔚 Programa terminado")


if __name__ == "__main__":
    main()
