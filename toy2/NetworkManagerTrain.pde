
// NetworkManager.pde – modo de treino: cliente TCP para receber dados do Python
// Compatível com servidor Python em 127.0.0.1:60345 a enviar 1 linha por mensagem
// Linha pode ser: um inteiro (animação) OU JSON com campos (anim, distanciaMaoDireita, distanciaMaoEsquerda, anguloMaoEsquerda, anguloVerticalMaoEsquerda)
import java.io.*;
import java.net.*;
import java.util.concurrent.atomic.AtomicBoolean;

class NetworkManagerTrain {
  private Socket socket;
  private BufferedReader reader;
  private PrintWriter writer;
  private Thread rxThread;
  private final AtomicBoolean connected = new AtomicBoolean(false);
  private final AtomicBoolean stopFlag = new AtomicBoolean(false);

  // estados recebidos - ALTERAÇÃO: valor padrão mudado para 10
  private volatile int currentAnimation = 10;
  private volatile float distanciaMaoDireita = -1;
  private volatile float distanciaMaoEsquerda = -1;
  private volatile float anguloMaoEsquerda = -1;
  private volatile float anguloVerticalMaoEsquerda = -1;
  private volatile boolean validMaoDireita = false;
  private volatile boolean validMaoEsquerda = false;
  private volatile boolean validAnguloEsquerda = false;

  // métricas simples
  private volatile long lastRxAtMs = -1;
  private volatile long messagesReceived = 0;

  // ===== API =====
  public boolean isConnected() { return connected.get(); }
  public int getCurrentAnimation() { return currentAnimation; }
  public NetworkData getNetworkData() {
    return new NetworkData(distanciaMaoDireita, distanciaMaoEsquerda, anguloMaoEsquerda, anguloVerticalMaoEsquerda,
                           validMaoDireita, validMaoEsquerda, validAnguloEsquerda);
  }
  public long getMsSinceLastRx() { return lastRxAtMs < 0 ? -1 : (millis() - lastRxAtMs); }

  // Conectar ao servidor Python (ex.: 127.0.0.1:60345)
  public synchronized void connect(String host, int port) {
    disconnect();
    println("[NetworkManagerTrain] Tentando conectar a " + host + ":" + port + "...");
    
    try {
      socket = new Socket();
      // Aumentar timeout para 5 segundos
      socket.connect(new InetSocketAddress(host, port), 5000);
      reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
      writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()), true);
      stopFlag.set(false);
      rxThread = new Thread(new Runnable() {
        public void run() { rxLoop(); }
      }, "Net-RX");
      rxThread.start();
      connected.set(true);
      println("[NetworkManagerTrain] ✓ CONECTADO com sucesso a " + host + ":" + port);
    } catch (java.net.ConnectException e) {
      println("[NetworkManagerTrain] ✗ FALHA: Servidor não está ouvindo em " + host + ":" + port);
      println("[NetworkManagerTrain] Verifique se o script Python está rodando e o servidor está ativo");
      disconnect();
    } catch (java.net.SocketTimeoutException e) {
      println("[NetworkManagerTrain] ✗ TIMEOUT: Servidor não respondeu em " + host + ":" + port);
      disconnect();
    } catch (Exception e) {
      println("[NetworkManagerTrain] ✗ ERRO ao conectar: " + e.getClass().getSimpleName() + " - " + e.getMessage());
      disconnect();
    }
  }

  public synchronized void disconnect() {
    stopFlag.set(true);
    connected.set(false);
    try { if (socket != null) socket.close(); } catch (Exception e) {}
    try { if (reader != null) reader.close(); } catch (Exception e) {}
    try { if (writer != null) writer.close(); } catch (Exception e) {}
    socket = null; reader = null; writer = null;
    
    // ADIÇÃO: Reset para valor padrão quando desconecta
    currentAnimation = 10;
    validMaoDireita = false;
    validMaoEsquerda = false;
    validAnguloEsquerda = false;
  }

  // Enviar JSON para o Python (opcional)
  public void sendCustomData(String type, JSONObject data) {
    if (!isConnected() || writer == null) return;
    try {
      JSONObject envelope = new JSONObject();
      envelope.setString("type", type);
      envelope.setJSONObject("data", data);
      writer.println(envelope.toString());
    } catch (Exception e) {
      println("[NetworkManager] Erro ao enviar: " + e.getMessage());
    }
  }

  // ===== Internals =====
  private void rxLoop() {
    try {
      String line;
      while (!stopFlag.get() && (line = reader.readLine()) != null) {
        line = line.trim();
        if (line.length() == 0) continue;
        messagesReceived++;
        lastRxAtMs = millis();
        
        // ADIÇÃO: Debug para ver o que está sendo recebido
        println("[NetworkManagerTrain] Recebido: " + line);
        
        // tentar inteiro rápido
        try {
          int receivedAnim = Integer.parseInt(line);
          // ALTERAÇÃO: Só atualiza se receber valor válido (diferente de 0)
          // Se quiser aceitar 0 como valor válido, remova esta condição
          if (receivedAnim != 0) {
            currentAnimation = receivedAnim;
            println("[NetworkManagerTrain] Animação atualizada para: " + currentAnimation);
          } else {
            // Opcional: manter valor atual ou usar valor padrão
            println("[NetworkManagerTrain] Recebido 0, mantendo valor atual: " + currentAnimation);
          }
          continue;
        } catch (Exception ignore) {}
        
        // senão tentar JSON
        try {
          JSONObject obj = parseJSONObject(line);
          if (obj == null) continue;
          if (obj.hasKey("anim")) {
            int jsonAnim = obj.getInt("anim");
            // Mesma lógica para JSON
            if (jsonAnim != 0) {
              currentAnimation = jsonAnim;
              println("[NetworkManagerTrain] Animação (JSON) atualizada para: " + currentAnimation);
            }
          }
          if (obj.hasKey("distanciaMaoDireita")) {
            distanciaMaoDireita = obj.getFloat("distanciaMaoDireita");
            validMaoDireita = true;
          }
          if (obj.hasKey("distanciaMaoEsquerda")) {
            distanciaMaoEsquerda = obj.getFloat("distanciaMaoEsquerda");
            validMaoEsquerda = true;
          }
          if (obj.hasKey("anguloMaoEsquerda")) {
            anguloMaoEsquerda = obj.getFloat("anguloMaoEsquerda");
            validAnguloEsquerda = true;
          }
          if (obj.hasKey("anguloVerticalMaoEsquerda")) {
            anguloVerticalMaoEsquerda = obj.getFloat("anguloVerticalMaoEsquerda");
          }
        } catch (Exception e) {
          println("[NetworkManager] Linha inválida: " + line);
        }
      }
    } catch (Exception e) {
      if (!stopFlag.get()) println("[NetworkManager] RX terminou: " + e.getMessage());
    } finally {
      connected.set(false);
      // Reset para valor padrão quando conexão termina
      currentAnimation = 10;
    }
  }
  
  // ADIÇÃO: Método para forçar reset do valor
  public void resetToDefault() {
    currentAnimation = 10;
    validMaoDireita = false;
    validMaoEsquerda = false;
    validAnguloEsquerda = false;
    println("[NetworkManagerTrain] Reset para valores padrão");
  }
}
