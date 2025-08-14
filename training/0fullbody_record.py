# Version learning_to_interact/2input/fullbody/fullbody_video_recording82_1.py
# Sistema de Captura e Análise de Pose Corporal com MediaPipe - VERSÃO CORRIGIDA
# Corrige o problema de sincronização entre tempo real e duração do vídeo
# adiciona o vídeo 

"""
Sistema de Captura e Análise de Pose Corporal com MediaPipe
Versão com Sincronização Temporal Precisa e Leitor de CSV

Autor: Sistema de Captura de Pose
Data: 2025
Funcionalidades:
- Captura de landmarks (pose, mãos, rosto) com MediaPipe
- Sincronização temporal de alta precisão
- Integração com vídeo de referência
- Leitura e apresentação de dados CSV em tempo real
- Detecção automática de drift temporal
- Gravação sincronizada de vídeo e dados CSV
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ===========================================================================================
# CONFIGURAÇÕES GLOBAIS
# ===========================================================================================

class Config:
    """Configurações centralizadas do sistema."""
    
    # Tempo e Performance
    MAX_RECORDING_TIME_MS = 300000  # 5 minutos
    DRIFT_CHECK_INTERVAL = 5.0      # Verificar drift a cada 5 segundos
    DRIFT_THRESHOLD = 10            # Frames de diferença para considerar drift
    
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
    DEFAULT_LANDMARK_VALUE = 500.0
    
    # Visualização
    SQUARE_SIZE = 3
    FACE_SQUARE_SIZE = 1
    SQUARE_COLOR = (255, 255, 255)  # Branco (BGR)
    
    # Debug
    DEBUG_FRAME_INTERVAL = 60  # Debug a cada 60 frames (~2s)


# ===========================================================================================
# UTILITÁRIOS DE TEMPO E SINCRONIZAÇÃO
# ===========================================================================================

class TimeSync:
    """Utilitários para sincronização temporal precisa."""
    
    def __init__(self):
        self.start_timestamp = time.time()
        self.start_perf_counter = time.perf_counter()
        self.last_drift_check = 0
        self.drift_corrections = 0
    
    def get_elapsed_seconds(self) -> float:
        """Retorna tempo decorrido em segundos com máxima precisão."""
        return time.perf_counter() - self.start_perf_counter
    
    def get_current_timestamp_ms(self) -> int:
        """Retorna timestamp absoluto em milissegundos."""
        elapsed = self.get_elapsed_seconds()
        return int((self.start_timestamp + elapsed) * 1000)
    
    def check_drift(self, frame_count: int) -> bool:
        """
        Verifica e reporta drift temporal.
        
        Args:
            frame_count: Número atual de frames
            
        Returns:
            True se drift foi detectado
        """
        elapsed = self.get_elapsed_seconds()
        
        if elapsed - self.last_drift_check >= Config.DRIFT_CHECK_INTERVAL:
            expected_frames = elapsed * 30  # Assumindo 30 FPS
            frame_drift = abs(frame_count - expected_frames)
            
            if frame_drift > Config.DRIFT_THRESHOLD:
                self.drift_corrections += 1
                print(f"[SYNC] ⚠️  Drift detectado: {frame_drift:.1f} frames aos {elapsed:.1f}s")
                self.last_drift_check = elapsed
                return True
            
            self.last_drift_check = elapsed
        
        return False
    
    def get_stats(self, frame_count: int) -> Dict[str, float]:
        """Retorna estatísticas de sincronização."""
        elapsed = self.get_elapsed_seconds()
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'elapsed_seconds': elapsed,
            'fps': fps,
            'drift_corrections': self.drift_corrections,
            'start_timestamp': self.start_timestamp
        }


# ===========================================================================================
# LEITOR DE DADOS CSV
# ===========================================================================================

class CSVDataReader:
    """Leitor inteligente de dados CSV com sincronização temporal."""
    
    def __init__(self, path_csv: str):
        self.path_csv = path_csv
        self.data = None
        self.is_loaded = False
        self.time_column = None
        self.max_time = 0
        self.fps_detected = 30
        
        # Arrays para busca rápida
        self.time_array = None
        self.animation_array = None
        
        if path_csv and os.path.exists(path_csv):
            self._load_csv()
    
    def _load_csv(self):
        """Carrega e processa o ficheiro CSV."""
        try:
            self.data = pd.read_csv(self.path_csv)
            print(f"[CSV] ✅ Carregado: {len(self.data)} linhas")
            print(f"[CSV] Colunas: {list(self.data.columns)}")
            
            if 'AnimacaoAtual' not in self.data.columns:
                print("[CSV] ❌ Coluna 'AnimacaoAtual' não encontrada!")
                return
            
            self._detect_time_column()
            self._prepare_lookup_arrays()
            self._print_summary()
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"[CSV] ❌ Erro ao carregar: {e}")
    
    def _detect_time_column(self):
        """Detecta coluna de tempo e calcula FPS."""
        time_columns = ['tempo', 'time', 'timestamp', 'segundos', 'Tempo', 'Time']
        
        for col in time_columns:
            if col in self.data.columns:
                self.time_column = col
                break
        
        if self.time_column is None:
            # Criar coluna baseada no índice
            estimated_duration = 300  # 5 minutos
            self.fps_detected = len(self.data) / estimated_duration
            self.fps_detected = np.clip(self.fps_detected, 10, 120)  # Limitar FPS
            
            self.data['tempo_calculado'] = self.data.index * (1/self.fps_detected)
            self.time_column = 'tempo_calculado'
        else:
            # Calcular FPS dos dados existentes
            if len(self.data) > 1:
                time_values = self.data[self.time_column].values
                time_diff = time_values[-1] - time_values[0]
                if time_diff > 0:
                    self.fps_detected = (len(self.data) - 1) / time_diff
        
        self.max_time = self.data[self.time_column].max()
    
    def _prepare_lookup_arrays(self):
        """Prepara arrays para busca binária rápida."""
        # Ordenar por tempo
        self.data = self.data.sort_values(by=self.time_column).reset_index(drop=True)
        
        # Criar arrays numpy para performance
        self.time_array = self.data[self.time_column].values
        self.animation_array = self.data['AnimacaoAtual'].values
    
    def _print_summary(self):
        """Imprime resumo das configurações CSV."""
        print(f"[CSV] Configuração:")
        print(f"[CSV]   - Coluna tempo: {self.time_column}")
        print(f"[CSV]   - FPS detectado: {self.fps_detected:.2f}")
        print(f"[CSV]   - Duração: {self.max_time:.2f}s")
        print(f"[CSV]   - Animações: {sorted(self.data['AnimacaoAtual'].unique())}")
    
    def get_animation_at_time(self, elapsed_seconds: float) -> Tuple[str, str, str]:
        """
        Obtém animação atual com informações de debug.
        
        Args:
            elapsed_seconds: Tempo decorrido
            
        Returns:
            Tuple (animação_original, animação_display, info_debug)
        """
        if not self.is_loaded:
            return "CSV não carregado", "CSV não carregado", "Erro"
        
        try:
            # Loop automático se necessário
            effective_time = elapsed_seconds % self.max_time if self.max_time > 0 else elapsed_seconds
            
            # Busca binária
            idx = np.searchsorted(self.time_array, effective_time, side='right') - 1
            idx = np.clip(idx, 0, len(self.animation_array) - 1)
            
            animation_original = str(self.animation_array[idx])
            csv_time = self.time_array[idx]
            
            # Converter animação para display (+1)
            animation_display = self._convert_animation_for_display(animation_original)
            
            # Info de debug
            time_diff = abs(csv_time - effective_time)
            loop_count = int(elapsed_seconds // self.max_time) if self.max_time > 0 else 0
            
            debug_info = f"csv:{csv_time:.2f}|diff:{time_diff:.3f}|loop:{loop_count}|orig:{animation_original}"
            
            return animation_original, animation_display, debug_info
            
        except Exception as e:
            return f"Erro: {str(e)}", f"Erro: {str(e)}", "Erro"
    
    def _convert_animation_for_display(self, animation_value: str) -> str:
        """
        Converte valor da animação para display (+1).
        
        Args:
            animation_value: Valor original da animação
            
        Returns:
            Valor convertido para display
        """
        try:
            # Tentar converter para número e adicionar 1
            if animation_value.isdigit():
                return str(int(animation_value) + 1)
            
            # Se for float
            try:
                float_val = float(animation_value)
                if float_val.is_integer():
                    return str(int(float_val) + 1)
                else:
                    return f"{float_val + 1:.1f}"
            except ValueError:
                pass
            
            # Se não conseguir converter, retornar original
            return animation_value
            
        except Exception:
            return animation_value


# ===========================================================================================
# GESTOR DE VÍDEO DE REFERÊNCIA
# ===========================================================================================

class VideoManager:
    """Gestor de vídeo de referência com sincronização temporal."""
    
    def __init__(self, path_video: str):
        self.path_video = path_video
        self.cap = None
        self.current_frame = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame_index = -1
        self.is_initialized = False
        
        if os.path.exists(path_video):
            self._initialize()
    
    def _initialize(self):
        """Inicializa o vídeo de referência."""
        try:
            self.cap = cv2.VideoCapture(self.path_video)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir: {self.path_video}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            print(f"[VIDEO] ✅ Carregado: {self.total_frames} frames, {self.fps:.2f} FPS, {duration:.2f}s")
            
            # Ler primeira frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_index = 0
            
            self.is_initialized = True
            
        except Exception as e:
            print(f"[VIDEO] ❌ Erro: {e}")
    
    def get_frame_at_time(self, elapsed_seconds: float) -> Optional[np.ndarray]:
        """
        Obtém frame do vídeo no tempo especificado.
        
        Args:
            elapsed_seconds: Tempo decorrido
            
        Returns:
            Frame do vídeo ou None
        """
        if not self.is_initialized or not self.cap:
            return None
        
        # Calcular frame alvo com loop
        target_frame = int(elapsed_seconds * self.fps) % self.total_frames
        
        # Só avançar se mudou de frame
        if target_frame != self.current_frame_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame
                self.current_frame_index = target_frame
        
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def release(self):
        """Liberta recursos."""
        if self.cap:
            self.cap.release()


# ===========================================================================================
# PROCESSADOR DE LANDMARKS
# ===========================================================================================

class LandmarkProcessor:
    """Processador de landmarks do MediaPipe."""
    
    @staticmethod
    def get_detection_status(results) -> Dict[str, int]:
        """Obtém status de detecção de landmarks."""
        return {
            "face": 1 if results.face_landmarks else 0,
            "pose": 1 if results.pose_landmarks else 0,
            "left_hand": 1 if results.left_hand_landmarks else 0,
            "right_hand": 1 if results.right_hand_landmarks else 0
        }
    
    @staticmethod
    def extract_coordinates(landmarks, expected_count: int) -> List[float]:
        """Extrai coordenadas x,y,z de landmarks."""
        if landmarks:
            return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        else:
            return [Config.DEFAULT_LANDMARK_VALUE] * (expected_count * 3)
    
    @staticmethod
    def create_csv_row(results, timestamp_ms: int, status: Dict[str, int]) -> List:
        """Cria linha completa para CSV SEM AnimacaoAtual."""
        row = [timestamp_ms]
        row.extend([status["face"], status["pose"], status["right_hand"], status["left_hand"]])
        # REMOVIDO: row.append(animation_original)  # NÃO incluir AnimacaoAtual no CSV de saída
        
        # Extrair coordenadas
        pose_coords = LandmarkProcessor.extract_coordinates(
            results.pose_landmarks.landmark if results.pose_landmarks else None,
            Config.NUM_POSE_LANDMARKS
        )
        
        left_hand_coords = LandmarkProcessor.extract_coordinates(
            results.left_hand_landmarks.landmark if results.left_hand_landmarks else None,
            Config.NUM_HAND_LANDMARKS
        )
        
        right_hand_coords = LandmarkProcessor.extract_coordinates(
            results.right_hand_landmarks.landmark if results.right_hand_landmarks else None,
            Config.NUM_HAND_LANDMARKS
        )
        
        face_coords = LandmarkProcessor.extract_coordinates(
            results.face_landmarks.landmark if results.face_landmarks else None,
            Config.NUM_FACE_LANDMARKS
        )
        
        row.extend(pose_coords + left_hand_coords + right_hand_coords + face_coords)
        return row


# ===========================================================================================
# GESTOR DE FICHEIROS
# ===========================================================================================

class FileManager:
    """Gestor de ficheiros de saída."""
    
    @staticmethod
    def create_path_csv() -> Tuple[str, List[str]]:
        """Cria caminho e cabeçalho do CSV."""
        today = datetime.now().strftime('%Y_%m_%d')
        folder_path = os.path.join("../data", today)
        os.makedirs(folder_path, exist_ok=True)
        
        # Encontrar próxima versão
        version = 1
        while os.path.exists(os.path.join(folder_path, f'v{version}.csv')):
            version += 1
        
        path_csv = os.path.join(folder_path, f'v{version}.csv')
        header = FileManager._create_csv_header()
        
        return path_csv, header
    
    @staticmethod
    def _create_csv_header() -> List[str]:
        """Cria cabeçalho completo do CSV SEM AnimacaoAtual."""
        header = ['MillisSinceEpoch', 'Face', 'Pose', 'RightHand', 'LeftHand']
        # REMOVIDO: 'AnimacaoAtual' do cabeçalho
        
        # Landmarks da pose
        for i in range(Config.NUM_POSE_LANDMARKS):
            header.extend([f'xp{i+1}', f'yp{i+1}', f'zp{i+1}'])
        
        # Landmarks das mãos
        for i in range(Config.NUM_HAND_LANDMARKS):
            header.extend([f'xlh{i+1}', f'ylh{i+1}', f'zlh{i+1}'])
        
        for i in range(Config.NUM_HAND_LANDMARKS):
            header.extend([f'xrh{i+1}', f'yrh{i+1}', f'zrh{i+1}'])
        
        # Landmarks do rosto
        for i in range(Config.NUM_FACE_LANDMARKS):
            header.extend([f'xfm{i+1}', f'yfm{i+1}', f'zfm{i+1}'])
        
        return header
    
    @staticmethod
    def create_path_video() -> str:
        """Cria caminho do vídeo."""
        today = datetime.now().strftime('%Y_%m_%d')
        folder_path = os.path.join("../data", today)
        os.makedirs(folder_path, exist_ok=True)
        
        version = 1
        while os.path.exists(os.path.join(folder_path, f'v{version}.mp4')):
            version += 1
        
        return os.path.join(folder_path, f'v{version}.mp4')


# ===========================================================================================
# VISUALIZADOR
# ===========================================================================================

class Visualizer:
    """Gestor de visualização e interface."""
    
    @staticmethod
    def draw_landmarks(image: np.ndarray, results):
        """Desenha landmarks como quadrados brancos."""
        h, w = image.shape[:2]
        
        # Landmarks do rosto (pequenos)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.rectangle(image, 
                            (x - Config.FACE_SQUARE_SIZE, y - Config.FACE_SQUARE_SIZE),
                            (x + Config.FACE_SQUARE_SIZE, y + Config.FACE_SQUARE_SIZE),
                            Config.SQUARE_COLOR, -1)
        
        # Landmarks da pose, mãos (normais)
        for landmarks in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if landmarks:
                for landmark in landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.rectangle(image,
                                (x - Config.SQUARE_SIZE, y - Config.SQUARE_SIZE),
                                (x + Config.SQUARE_SIZE, y + Config.SQUARE_SIZE),
                                Config.SQUARE_COLOR, -1)
    
    @staticmethod
    def add_info_overlay(image: np.ndarray, frame_count: int, elapsed_ms: float, 
                        animation_display: str, debug_info: str, has_video: bool):
        """Adiciona overlay com informações (mostra animação +1)."""
        elapsed_s = elapsed_ms / 1000
        max_s = Config.MAX_RECORDING_TIME_MS / 1000
        fps = frame_count / elapsed_s if elapsed_s > 0 else 0
        
        # Overlay principal
        overlay = image.copy()
        cv2.rectangle(overlay, (0, image.shape[0] - 30), (300, image.shape[0] ), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Animação atual (destaque) - usar valor +1 para display
        animation_name = animation_display.split(' (')[0] if animation_display else "N/A"
        cv2.putText(image, f"Anim: {animation_name}",
                   (15, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Tempo
        cv2.putText(image, f"Tempo: {elapsed_s:.2f}s / {max_s:.0f}s",
                   (100, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Debug de sincronização
        #cv2.putText(image, f"Sync: {debug_info}",
                   #(15, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FPS
        #cv2.putText(image, f"FPS: {fps:.1f}",
                   #(15, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Indicador de vídeo
        #if has_video:
            #cv2.putText(image, "REF", (image.shape[1] - 50, 25),
                       #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    @staticmethod
    def create_split_screen(camera_frame: np.ndarray, video_frame: Optional[np.ndarray]) -> np.ndarray:
        """Cria tela dividida câmara + vídeo."""
        camera_h, camera_w = camera_frame.shape[:2]
        target_width = 1280
        half_width = target_width // 2
        
        # Redimensionar câmara
        camera_scale = half_width / camera_w
        camera_new_h = int(camera_h * camera_scale)
        camera_resized = cv2.resize(camera_frame, (half_width, camera_new_h))
        
        # Preparar vídeo
        if video_frame is not None:
            video_h, video_w = video_frame.shape[:2]
            video_scale = half_width / video_w
            video_new_h = int(video_h * video_scale)
            video_resized = cv2.resize(video_frame, (half_width, video_new_h))
            final_h = max(camera_new_h, video_new_h)
        else:
            final_h = camera_new_h
            video_resized = np.zeros((final_h, half_width, 3), dtype=np.uint8)
            cv2.putText(video_resized, "Sem video", (half_width//4, final_h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ajustar alturas
        if camera_new_h < final_h:
            padding = (final_h - camera_new_h) // 2
            camera_padded = np.zeros((final_h, half_width, 3), dtype=np.uint8)
            camera_padded[padding:padding + camera_new_h, :] = camera_resized
        else:
            camera_padded = camera_resized
        
        if video_frame is not None and video_new_h < final_h:
            padding = (final_h - video_new_h) // 2
            video_padded = np.zeros((final_h, half_width, 3), dtype=np.uint8)
            video_padded[padding:padding + video_new_h, :] = video_resized
        else:
            video_padded = video_resized
        
        # Combinar
        combined = np.hstack([camera_padded, video_padded])
        cv2.line(combined, (half_width, 0), (half_width, final_h), (255, 255, 255), 2)
        
        return combined


# ===========================================================================================
# SISTEMA PRINCIPAL DE CAPTURA
# ===========================================================================================

class PoseCaptureSystem:
    """Sistema principal de captura com sincronização temporal."""
    
    def __init__(self, path_video: Optional[str] = None, path_csv: Optional[str] = None, 
                 split_screen: bool = True):
        # Componentes principais
        self.time_sync = TimeSync()
        self.csv_reader = CSVDataReader(path_csv) if path_csv else None
        self.video_manager = VideoManager(path_video) if path_video else None
        self.split_screen = split_screen
        
        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        
        # Câmara
        self.cap = self._setup_camera()
        
        # Ficheiros de saída
        self.path_csv, self.csv_header = FileManager.create_path_csv()
        self.path_video = FileManager.create_path_video()
        
        # Dados
        self.frames = []
        self.csv_data = []
        
        self._print_initialization_summary()
    
    def _setup_camera(self) -> cv2.VideoCapture:
        """Configura webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        if not cap.isOpened():
            raise RuntimeError("Erro: Não foi possível aceder à webcam!")
        
        return cap
    
    def _print_initialization_summary(self):
        """Imprime resumo da inicialização."""
        print("[INIT] 🎯 Sistema de Captura Inicializado")
        print(f"[INIT] CSV: {self.path_csv}")
        print(f"[INIT] Vídeo: {self.path_video}")
        
        if self.csv_reader and self.csv_reader.is_loaded:
            print("[INIT] ✅ Dados CSV carregados")
        
        if self.video_manager and self.video_manager.is_initialized:
            print("[INIT] ✅ Vídeo de referência carregado")
        
        if self.split_screen and self.video_manager:
            print("[INIT] 📺 Modo: Tela dividida")
        else:
            print("[INIT] 🪟 Modo: Janelas separadas")
    
    def run(self):
        """Executa captura principal."""
        print(f"\n[CAPTURE] 🎬 Iniciando captura de {Config.MAX_RECORDING_TIME_MS/1000:.0f}s")
        print("[CAPTURE] Pressione 'q' para parar")
        
        self._setup_windows()
        
        frame_count = 0
        
        try:
            with self.mp_holistic.Holistic(
                min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
                refine_face_landmarks=False,
                model_complexity=Config.MODEL_COMPLEXITY
            ) as holistic:
                
                while self.cap.isOpened():
                    success, image = self.cap.read()
                    if not success:
                        continue
                    
                    # Tempo atual
                    elapsed_s = self.time_sync.get_elapsed_seconds()
                    elapsed_ms = elapsed_s * 1000
                    timestamp_ms = self.time_sync.get_current_timestamp_ms()
                    frame_count += 1
                    
                    # Verificar tempo limite
                    if elapsed_ms >= Config.MAX_RECORDING_TIME_MS:
                        print("[CAPTURE] ⏰ Tempo máximo atingido")
                        break
                    
                    # Verificar drift
                    self.time_sync.check_drift(frame_count)
                    
                    # Obter animação atual APENAS para display (não para CSV)
                    animation_original, animation_display, debug_info = "N/A", "N/A", ""
                    if self.csv_reader and self.csv_reader.is_loaded:
                        animation_original, animation_display, debug_info = self.csv_reader.get_animation_at_time(elapsed_s)
                    
                    # Processar landmarks
                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    # Recolher dados SEM incluir AnimacaoAtual
                    status = LandmarkProcessor.get_detection_status(results)
                    csv_row = LandmarkProcessor.create_csv_row(results, timestamp_ms, status)  # REMOVIDO animation_original
                    self.csv_data.append(csv_row)
                    
                    # Debug periódico (mostrar valor +1 apenas para display)
                    if frame_count % Config.DEBUG_FRAME_INTERVAL == 0:
                        fps = frame_count / elapsed_s if elapsed_s > 0 else 0
                        display_anim = animation_display.split(' (')[0] if animation_display else "N/A"
                        print(f"[DEBUG] Frame {frame_count}: "
                            f"F{status['face']}P{status['pose']}L{status['left_hand']}R{status['right_hand']} "
                            f"T={elapsed_s:.2f}s FPS={fps:.1f} Anim={display_anim}")
                    
                    # Visualização (mostrar valor +1 apenas na interface)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    Visualizer.draw_landmarks(image, results)
                    Visualizer.add_info_overlay(image, frame_count, elapsed_ms, animation_display, debug_info,
                                            self.video_manager is not None)
                    
                    self._display_frames(image, elapsed_s)
                    self.frames.append(image.copy())
                    
                    # Verificar teclas
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[CAPTURE] 🛑 Parado pelo utilizador")
                        break
        
        finally:
            self._cleanup_and_save()
    
    def _setup_windows(self):
        """Configura janelas de visualização."""
        if self.split_screen and self.video_manager:
            cv2.namedWindow('Captura - Camara + Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Captura - Camara + Video', 1280, 600)
        else:
            cv2.namedWindow('Captura de Pose', cv2.WINDOW_NORMAL)
            if self.video_manager:
                cv2.namedWindow('Video de Referencia', cv2.WINDOW_NORMAL)
                cv2.moveWindow('Video de Referencia', 700, 100)
    
    def _display_frames(self, camera_frame: np.ndarray, elapsed_s: float):
        """Exibe frames na interface."""
        if self.split_screen and self.video_manager:
            # Tela dividida
            video_frame = self.video_manager.get_frame_at_time(elapsed_s)
            combined = Visualizer.create_split_screen(camera_frame, video_frame)
            cv2.imshow('Captura - Camara + Video', combined)
        else:
            # Janelas separadas
            cv2.imshow('Captura de Pose', camera_frame)
            if self.video_manager:
                video_frame = self.video_manager.get_frame_at_time(elapsed_s)
                if video_frame is not None:
                    cv2.imshow('Video de Referencia', video_frame)
    
    def _cleanup_and_save(self):
        """Limpa recursos e guarda ficheiros."""
        # Fechar recursos
        self.cap.release()
        if self.video_manager:
            self.video_manager.release()
        cv2.destroyAllWindows()
        
        # Obter estatísticas
        stats = self.time_sync.get_stats(len(self.frames))
        
        print(f"\n[SAVE] 💾 A guardar {len(self.frames)} frames...")
        
        # Guardar CSV
        self._save_csv(stats)
        
        # Guardar vídeo
        self._save_video(stats)
        
        # Relatório final
        self._print_final_report(stats)
    
    def _save_csv(self, stats: Dict[str, float]):
        """Guarda ficheiro CSV apenas com cabeçalho e dados."""
        with open(self.path_csv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.csv_header)
            writer.writerows(self.csv_data)
    
    def _save_video(self, stats: Dict[str, float]):
        """Guarda vídeo com FPS otimizado."""
        # Calcular FPS ideal
        target_fps = min(max(stats["fps"], 15), 60)
        if abs(target_fps - 30) > 15:
            target_fps = 30
        
        # Gravar vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.path_video, fourcc, target_fps,
                                (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
        
        for frame in self.frames:
            writer.write(frame)
        writer.release()
    
    def _print_final_report(self, stats: Dict[str, float]):
        """Imprime relatório final."""
        video_duration = len(self.frames) / 30  # Assumindo 30 FPS
        time_accuracy = abs(stats["elapsed_seconds"] - video_duration)
        
        if time_accuracy < 0.1:
            quality = "EXCELENTE"
        elif time_accuracy < 0.5:
            quality = "BOA"
        else:
            quality = "ACEITÁVEL"
        
        print("\n" + "="*70)
        print("[FINAL] 🎯 CAPTURA CONCLUÍDA")
        print("="*70)
        print(f"[FINAL] ⏱️  Duração real: {stats['elapsed_seconds']:.3f}s")
        print(f"[FINAL] 📹 Frames capturadas: {len(self.frames)}")
        print(f"[FINAL] 🎬 FPS médio: {stats['fps']:.2f}")
        print(f"[FINAL] 📐 Precisão temporal: ±{time_accuracy:.3f}s ({quality})")
        print(f"[FINAL] 🔧 Correções drift: {stats['drift_corrections']}")
        print(f"[FINAL] 📊 CSV: {self.path_csv}")
        print(f"[FINAL] 🎥 Vídeo: {self.path_video}")
        
        if self.csv_reader and self.csv_reader.is_loaded:
            print(f"[FINAL] 🔗 Sincronização CSV: ATIVA")
        
        if quality == "EXCELENTE":
            print("[FINAL] ✅ Sincronização temporal excelente!")
        elif time_accuracy > 1.0:
            print("[FINAL] ⚠️  AVISO: Discrepância temporal elevada!")
        
        print("="*70)


# ===========================================================================================
# FUNÇÃO PRINCIPAL
# ===========================================================================================

def main():
    """Função principal do programa."""
    try:
        print("🎯 Sistema de Captura de Pose com Sincronização Temporal")
        print("="*70)
        
        # Caminhos dos ficheiros
        #folder = "../data/dataset30"
        folder_video_and_csv = "../data/0base"
        video = "video1.mp4"
        csv = "dados_teclas1.csv"

        path_video = Path(__file__).resolve().parent / folder_video_and_csv / video
        path_csv = Path(__file__).resolve().parent / folder_video_and_csv / csv
        
        # Verificar ficheiros
        video_exists = os.path.exists(path_video)
        csv_exists = os.path.exists(path_csv)
        
        print(f"[MAIN] Vídeo: {'✅' if video_exists else '❌'} {path_video}")
        print(f"[MAIN] CSV: {'✅' if csv_exists else '❌'} {path_csv}")
        
        # Validar CSV
        if not csv_exists:
            print("\n⚠️  Ficheiro CSV não encontrado!")
            print("Certifique-se que existe com coluna 'AnimacaoAtual'")
            
            if input("Continuar sem CSV? (s/N): ").strip().lower() != 's':
                return
            path_csv = None
        
        # Validar vídeo
        if not video_exists:
            print("⚠️  Vídeo de referência não encontrado!")
            path_video = None
        
        # Escolher modo de visualização
        split_screen = True
        if path_video:
            print("\n[MAIN] Modo de visualização:")
            print("1 - Tela dividida (recomendado)")
            print("2 - Janelas separadas")
            
            choice = input("Escolha (1-2, padrão=1): ").strip()
            split_screen = choice != "2"
        
        # Mostrar funcionalidades ativas
        print("\n🎯 Funcionalidades Ativas:")
        print("⏱️  - Sincronização temporal de alta precisão")
        print("🔄 - Detecção automática de drift")
        print("📊 - Monitorização FPS em tempo real")
        if path_csv:
            print("📋 - Integração com dados CSV")
        if path_video:
            print("🎬 - Vídeo de referência sincronizado")
        
        # Inicializar e executar
        print("\n[MAIN] Iniciando em 3 segundos...")
        time.sleep(3)
        
        system = PoseCaptureSystem(path_video, path_csv, split_screen)
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