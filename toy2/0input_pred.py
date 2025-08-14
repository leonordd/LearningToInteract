import socket
import json
import time
import cv2
import mediapipe as mp
import math
import numpy as np
import threading

HOST = '127.0.0.1'
PORT = 60345

# Criar socket e permitir reusar a porta
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Variáveis globais para dados de animação recebidos
current_animation = 0
animation_progress = 0.0
is_transitioning = False
target_animation = 0

# Buffer ultra robusto para acumular dados
class RobustJSONReceiver:
    def __init__(self):
        self.buffer = ""
        self.messages_processed = 0
        self.errors_count = 0
        
    def add_data(self, data_bytes):
        """Adicionar dados recebidos ao buffer"""
        try:
            self.buffer += data_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return []
        
        return self.extract_complete_messages()
    
    def extract_complete_messages(self):
        """Extrair mensagens JSON completas do buffer"""
        messages = []
        
        while True:
            start_idx = self.buffer.find('{')
            if start_idx == -1:
                self.buffer = ""
                break
            
            if start_idx > 0:
                self.buffer = self.buffer[start_idx:]
            
            brace_count = 0
            end_idx = -1
            
            for i, char in enumerate(self.buffer):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                break
            
            json_str = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:].lstrip()
            
            if json_str:
                messages.append(json_str)
                self.messages_processed += 1
        
        return messages

# Instância global do receptor
json_receiver = RobustJSONReceiver()

try:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[Python] À escuta em {HOST}:{PORT}...")
except OSError as e:
    print(f"[Python] Erro ao criar socket: {e}")
    exit(1)

def esperar_conexao():
    global conn, addr
    while True:
        try:
            server_socket.settimeout(1.0)
            conn, addr = server_socket.accept()
            print(f"[Python] Conectado por {addr[0]}:{addr[1]}")
            return
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[Python] Erro ao aceitar conexão: {e}")
            time.sleep(1)

def enviar_dados(dados):
    global conn
    if conn:
        try:
            mensagem = json.dumps(dados).encode('utf-8')
            conn.sendall(mensagem + b'\n')
        except (BrokenPipeError, ConnectionResetError):
            print("[Python] Conexão perdida. A aguardar nova ligação...")
            conn.close()
            esperar_conexao()
        except Exception as e:
            print(f"[Python] Erro ao enviar dados: {e}")

def receber_dados_thread():
    global conn, current_animation, animation_progress, is_transitioning, target_animation, json_receiver
    
    while True:
        try:
            if conn:
                conn.settimeout(0.1)
                data = conn.recv(4096)
                
                if data:
                    complete_messages = json_receiver.add_data(data)
                    
                    for message in complete_messages:
                        try:
                            received_data = json.loads(message)
                            processar_dados_recebidos(received_data)
                        except json.JSONDecodeError as e:
                            json_receiver.errors_count += 1
                
        except socket.timeout:
            continue
        except (ConnectionResetError, BrokenPipeError):
            print("[Python] Conexão perdida no thread de recepção")
            break
        except Exception as e:
            print(f"[Python] Erro no thread de recepção: {e}")
            time.sleep(0.1)

def processar_dados_recebidos(data):
    global current_animation, animation_progress, is_transitioning, target_animation
    
    try:
        tipo = data.get('tipo', '')
        
        if tipo == 'animation_update':
            current_animation = data.get('currentAnimation', 0)
            animation_progress = data.get('animationProgress', 0.0)
            is_transitioning = data.get('isTransitioning', False)
            target_animation = data.get('targetAnimation', 0)
                
        elif tipo == 'manual_animation_change':
            new_anim = data.get('newAnimation', 0)
            prev_anim = data.get('previousAnimation', 0)
            print(f"[Python] Mudança manual: {prev_anim} → {new_anim}")
            
        elif tipo == 'test_message':
            message = data.get('message', 'sem mensagem')
            print(f"[Python] Teste recebido: {message}")
            
        elif tipo == 'disconnect':
            print("[Python] Processing enviou sinal de desconexão")
            
    except Exception as e:
        print(f"[Python] Erro ao processar dados recebidos: {e}")

def calcular_distancia(results, parte_corpo, n_ponto1, n_ponto2):
    parte_mapa = {
        'mao_direita': results.right_hand_landmarks,
        'mao_esquerda': results.left_hand_landmarks,
        'corpo': results.pose_landmarks,
        'face': results.face_landmarks
    }

    pontos = parte_mapa.get(parte_corpo)
    if not pontos:
        return None

    try:
        p1 = pontos.landmark[n_ponto1]
        p2 = pontos.landmark[n_ponto2]
        distancia = math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (p1.z - p2.z) ** 2
        )

        dados = {
            "parte": parte_corpo,
            "ponto1": n_ponto1,
            "ponto2": n_ponto2,
            "distancia": distancia,
            "tipo": "distancia"
        }
        enviar_dados(dados)
        return distancia

    except IndexError:
        return None

def desenhar_cubo(image, x, y, size=10, ang=0):
    # Cor branca para todos os elementos
    color = (255, 255, 255)
    
    pts = np.array([
        [x + size * math.cos(ang), y + size * math.sin(ang)],
        [x + size * math.cos(ang + math.pi / 2), y + size * math.sin(ang + math.pi / 2)],
        [x + size * math.cos(ang + math.pi), y + size * math.sin(ang + math.pi)],
        [x + size * math.cos(ang + 3 * math.pi / 2), y + size * math.sin(ang + 3 * math.pi / 2)],
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    cv2.fillPoly(image, [pts], color=color)

# Iniciar sistema de visão
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

# Esperar até o Processing se ligar
conn = None
esperar_conexao()

# Iniciar thread para receber dados
receive_thread = threading.Thread(target=receber_dados_thread, daemon=True)
receive_thread.start()

print("[Python] Thread de recepção iniciada")

def mapear_valor(valor, min_entrada, max_entrada, min_saida, max_saida):
    # Garante que o valor está dentro dos limites
    valor = max(min_entrada, min(max_entrada, valor))
    
    # Fórmula de mapeamento linear
    valor_mapeado = (valor - min_entrada) / (max_entrada - min_entrada) * (max_saida - min_saida) + min_saida
    return valor_mapeado

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        angulo_rotacao = time.time() % (2 * math.pi)

        if results.right_hand_landmarks:
            p4 = results.right_hand_landmarks.landmark[4]
            p8 = results.right_hand_landmarks.landmark[8]
            x1, y1 = int(p4.x * w), int(p4.y * h)
            x2, y2 = int(p8.x * w), int(p8.y * h)

            # Linha e cubos brancos
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            desenhar_cubo(image, x1, y1, ang=angulo_rotacao)
            desenhar_cubo(image, x2, y2, ang=angulo_rotacao)

            distancia = calcular_distancia(results, 'mao_direita', 4, 8)
            if distancia:
                valor_key_transformado = mapear_valor(distancia, 0.05, 0.3, 0, 5)
                cv2.putText(image, f"Keyframes: {valor_key_transformado:.4f}", (x1 + 20, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if results.left_hand_landmarks:
            p4e = results.left_hand_landmarks.landmark[4]
            p8e = results.left_hand_landmarks.landmark[8]
            p12e = results.left_hand_landmarks.landmark[12]

            x1e, y1e = int(p4e.x * w), int(p4e.y * h)
            x2e, y2e = int(p8e.x * w), int(p8e.y * h)
            x3e, y3e = int(p12e.x * w), int(p12e.y * h)

            # Desenhar cubos brancos
            desenhar_cubo(image, x1e, y1e, ang=angulo_rotacao)
            desenhar_cubo(image, x2e, y2e, ang=angulo_rotacao)
            desenhar_cubo(image, x3e, y3e, ang=angulo_rotacao)

            # Distância da mão esquerda
            distancia_e = calcular_distancia(results, 'mao_esquerda', 4, 8)
            if distancia_e:
                valor_transformado = mapear_valor(distancia_e, 0.05, 0.3, 0, 5)
                cv2.putText(image, f"Transformar: {valor_transformado:.2f}", (x1e + 20, y1e - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Linha branca
            cv2.line(image, (x1e, y1e), (x2e, y2e), (255, 255, 255), 2)

            # ÂNGULO: entre ponto 4 e 12
            delta_x = p12e.x - p4e.x
            delta_y = p12e.y - p4e.y
            ang_rad = math.atan2(delta_y, delta_x)
            ang_deg = math.degrees(ang_rad)
            if ang_deg < 0:
                ang_deg += 360

            ##cv2.putText(image, f"animacao: {ang_deg:.1f}", (x1e, y1e - 40),
                        ##cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # NOVO: Mostrar número da animação perto do dedo 12
            # Incluir informação de transição se estiver em transição
            if is_transitioning:
                anim_text = f"Anim: {current_animation} -> {target_animation}"
            else:
                anim_text = f"Anim: {current_animation}"
                
            cv2.putText(image, anim_text, (x3e + 20, y3e ),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            dados_angulo = {
                "parte": "mao_esquerda",
                "ponto1": 4,
                "ponto2": 12,
                "angulo_horizontal": ang_deg,
                "angulo_vertical": 0.0,
                "tipo": "angulo"
            }
            enviar_dados(dados_angulo)

        cv2.imshow('MediaPipe Holistic - Modo de Jogo', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Libertar recursos
cap.release()
cv2.destroyAllWindows()
if conn:
    try:
        conn.close()
    except:
        pass
server_socket.close()
print("[Python] Programa terminado.")