// version5 --  igual à keypoints_optimized4.pde 
// no entanto com comunicação bidirecional entre o processing e o python
import java.io.*;
import java.net.*;

class NetworkManager {
  private Socket socket;
  private BufferedReader reader;
  private PrintWriter writer;
  private boolean connected = false;
  
  // Estruturas de dados otimizadas para simultaneidade
  private volatile float distanciaMaoDireita = -1;
  private volatile float distanciaMaoEsquerda = -1;
  private volatile float anguloMaoEsquerda = -1;
  private volatile float anguloVerticalMaoEsquerda = -1;
  
  // NOVO: Controle ultra restritivo de envio de animação
  private int lastSentAnimation = -1;
  private float lastSentT = -1;
  private boolean lastSentTransitioning = false;
  private long lastAnimationSendTime = 0;
  private static final long ANIMATION_SEND_INTERVAL = 200; // Aumentado para 200ms
  private static final float MIN_T_CHANGE = 0.5; // Mudança mínima em t para enviar
  
  // Timestamps para controlar validade dos dados
  private long timestampMaoDireita = 0;
  private long timestampMaoEsquerda = 0;
  private long timestampAnguloEsquerda = 0;
  
  // Configurações de timeout
  private static final long DATA_TIMEOUT = 500;
  private static final long RECONNECT_INTERVAL = 1000;
  private long lastReconnectAttempt = 0;
  
  // Buffer para processar múltiplas mensagens por frame
  private ArrayList<String> messageBuffer = new ArrayList<String>();
  
  // Contadores para debug
  private int messagesSent = 0;
  private int messagesReceived = 0;
  private int messagesSuppressed = 0; // NOVO: Contar mensagens suprimidas

  public void beginConnection() {
    tryToConnect();
  }

  private void tryToConnect() {
    long currentTime = System.currentTimeMillis();
    if (!connected && (currentTime - lastReconnectAttempt > RECONNECT_INTERVAL)) {
      try {
        println("Attempting to connect...");
        socket = new Socket("127.0.0.1", 60345);
        socket.setTcpNoDelay(true);
        // NOVO: Configurar timeouts do socket
        socket.setSoTimeout(1000); // 1 segundo timeout para operações
        
        reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        writer = new PrintWriter(socket.getOutputStream(), false); // NOVO: autoFlush = false
        connected = true;
        println("Connected to server.");
        lastReconnectAttempt = currentTime;
        
        // Reset contadores
        messagesSent = 0;
        messagesReceived = 0;
        messagesSuppressed = 0;
        
      }
      catch (Exception e) {
        println("Connection failed: " + e.getMessage());
        lastReconnectAttempt = currentTime;
      }
    }
  }

  // NOVO: Método ultra restritivo para enviar dados de animação
  public void sendAnimationData(int currentAnimation, float t, boolean isTransitioning, int targetAnimation) {
    if (!connected || writer == null) return;
    
    long currentTime = System.currentTimeMillis();
    
    // Critérios MUITO mais restritivos para enviar
    boolean shouldSend = false;
    
    // 1. Mudança de animação (sempre enviar)
    if (currentAnimation != lastSentAnimation) {
      shouldSend = true;
    }
    // 2. Mudança significativa em t E tempo suficiente passou
    else if (abs(t - lastSentT) >= MIN_T_CHANGE && 
             (currentTime - lastAnimationSendTime > ANIMATION_SEND_INTERVAL)) {
      shouldSend = true;
    }
    // 3. Mudança no estado de transição
    else if (isTransitioning != lastSentTransitioning) {
      shouldSend = true;
    }
    // 4. Forçar envio ocasional (a cada 2 segundos)
    else if (currentTime - lastAnimationSendTime > 2000) {
      shouldSend = true;
    }
    
    if (shouldSend) {
      try {
        JSONObject animationData = new JSONObject();
        animationData.setInt("currentAnimation", currentAnimation);
        animationData.setFloat("animationProgress", t);
        animationData.setBoolean("isTransitioning", isTransitioning);
        animationData.setInt("targetAnimation", targetAnimation);
        animationData.setString("tipo", "animation_update");
        animationData.setLong("timestamp", currentTime);
        
        String message = animationData.toString();
        writer.println(message);
        writer.flush(); // NOVO: Flush explícito
        
        lastSentAnimation = currentAnimation;
        lastSentT = t;
        lastSentTransitioning = isTransitioning;
        lastAnimationSendTime = currentTime;
        messagesSent++;
        
        // Debug muito reduzido
        if (messagesSent % 20 == 0) {
          println("Sent " + messagesSent + " messages (suppressed " + messagesSuppressed + "). Current: " + currentAnimation);
        }
        
      } catch (Exception e) {
        println("Error sending animation data: " + e.getMessage());
        connected = false;
        invalidateAllData();
      }
    } else {
      messagesSuppressed++;
    }
  }
  
  public void sendCustomData(String type, JSONObject customData) {
    if (!connected || writer == null) return;
    
    try {
      customData.setString("tipo", type);
      customData.setLong("timestamp", System.currentTimeMillis());
      
      String message = customData.toString();
      writer.println(message);
      writer.flush();
      messagesSent++;
      
      println("Sent custom data: " + type);
      
    } catch (Exception e) {
      println("Error sending custom data: " + e.getMessage());
      connected = false;
      invalidateAllData();
    }
  }

  public void checkIncomingMessages() {
    if (connected) {
      try {
        messageBuffer.clear();
        
        // NOVO: Ler com timeout para evitar bloqueio
        long startTime = System.currentTimeMillis();
        while (reader.ready() && (System.currentTimeMillis() - startTime < 10)) { // Max 10ms lendo
          String receivedText = reader.readLine();
          if (receivedText != null && !receivedText.isEmpty()) {
            messageBuffer.add(receivedText);
            messagesReceived++;
          }
        }
        
        // Processar mensagens com limite
        int processed = 0;
        for (String message : messageBuffer) {
          if (processed >= 10) break; // Máximo 10 mensagens por frame
          processMessage(message);
          processed++;
        }
        
        cleanExpiredData();
        
      }
      catch (SocketTimeoutException e) {
        // Timeout normal do socket
      }
      catch (IOException e) {
        println("Connection lost: " + e.getMessage());
        connected = false;
        invalidateAllData();
      }
      catch (Exception e) {
        println("Error reading message: " + e.getMessage());
        connected = false;
        invalidateAllData();
      }
    } else {
      tryToConnect();
    }
  }
  
  private void processMessage(String message) {
    try {
      JSONObject data = parseJSONObject(message);
      if (data != null) {
        processReceivedData(data);
      }
    } catch (Exception e) {
      // Suprimir completamente erros de JSON por agora
      // Para debug, descomentar a linha abaixo:
      // if (frameCount % 600 == 0) println("JSON parse error count: " + (++jsonErrorCount));
    }
  }

  private void processReceivedData(JSONObject receivedData) {
    if (receivedData == null) return;
    
    try {
      long currentTime = System.currentTimeMillis();
      
      // Verificar se tem as chaves necessárias com segurança
      if (!receivedData.hasKey("parte") || !receivedData.hasKey("ponto1") || !receivedData.hasKey("ponto2")) {
        return;
      }
      
      String parte = receivedData.getString("parte");
      int ponto1 = receivedData.getInt("ponto1");
      int ponto2 = receivedData.getInt("ponto2");
      
      boolean isAngulo = receivedData.hasKey("tipo") && 
                        receivedData.getString("tipo").equals("angulo");
      
      if (isAngulo) {
        if (receivedData.hasKey("angulo_horizontal") && receivedData.hasKey("angulo_vertical")) {
          float anguloHorizontal = receivedData.getFloat("angulo_horizontal");
          float anguloVertical = receivedData.getFloat("angulo_vertical");
          
          if (parte.equals("mao_esquerda") && ponto1 == 4 && ponto2 == 12) {
            anguloMaoEsquerda = anguloHorizontal;
            anguloVerticalMaoEsquerda = anguloVertical;
            timestampAnguloEsquerda = currentTime;
          }
        }
        
      } else {
        if (receivedData.hasKey("distancia")) {
          float distancia = receivedData.getFloat("distancia");
          
          if (parte.equals("mao_direita") && ponto1 == 4 && ponto2 == 8) {
            distanciaMaoDireita = distancia;
            timestampMaoDireita = currentTime;
            
          } else if (parte.equals("mao_esquerda") && ponto1 == 4 && ponto2 == 8) {
            distanciaMaoEsquerda = distancia;
            timestampMaoEsquerda = currentTime;
          }
        }
      }
    } catch (Exception e) {
      // Silenciosamente ignorar erros de processamento
    }
  }
  
  private void cleanExpiredData() {
    long currentTime = System.currentTimeMillis();
    
    if (currentTime - timestampMaoDireita > DATA_TIMEOUT) {
      distanciaMaoDireita = -1;
    }
    if (currentTime - timestampMaoEsquerda > DATA_TIMEOUT) {
      distanciaMaoEsquerda = -1;
    }
    if (currentTime - timestampAnguloEsquerda > DATA_TIMEOUT) {
      anguloMaoEsquerda = -1;
    }
  }
  
  private void invalidateAllData() {
    distanciaMaoDireita = -1;
    distanciaMaoEsquerda = -1;
    anguloMaoEsquerda = -1;
    anguloVerticalMaoEsquerda = -1;
    timestampMaoDireita = 0;
    timestampMaoEsquerda = 0;
    timestampAnguloEsquerda = 0;
  }
  
  public void closeConnection() {
    try {
      if (writer != null) {
        JSONObject farewell = new JSONObject();
        farewell.setString("tipo", "disconnect");
        farewell.setLong("timestamp", System.currentTimeMillis());
        writer.println(farewell.toString());
        writer.flush();
        
        println("Sent disconnect message to Python");
        delay(100);
        
        writer.close();
      }
      if (reader != null) reader.close();
      if (socket != null) socket.close();
      connected = false;
      
      println("Connection closed. Stats - Sent: " + messagesSent + ", Received: " + messagesReceived + ", Suppressed: " + messagesSuppressed);
    } catch (Exception e) {
      println("Error closing connection: " + e.getMessage());
    }
  }

  // Getters (mantidos iguais)
  public float getDistanciaMaoDireita() {
    return distanciaMaoDireita;
  }
  
  public float getDistanciaMaoEsquerda() {
    return distanciaMaoEsquerda;
  }
  
  public float getAnguloMaoEsquerda() {
    return anguloMaoEsquerda;
  }
  
  public float getAnguloVerticalMaoEsquerda() {
    return anguloVerticalMaoEsquerda;
  }
  
  public boolean temDadosMaoDireita() {
    return distanciaMaoDireita > 0 && 
           (System.currentTimeMillis() - timestampMaoDireita < DATA_TIMEOUT);
  }
  
  public boolean temDadosMaoEsquerda() {
    return distanciaMaoEsquerda > 0 && 
           (System.currentTimeMillis() - timestampMaoEsquerda < DATA_TIMEOUT);
  }
  
  public boolean temAnguloMaoEsquerda() {
    return anguloMaoEsquerda >= 0 && 
           (System.currentTimeMillis() - timestampAnguloEsquerda < DATA_TIMEOUT);
  }
  
  public long getIdadeDadosMaoDireita() {
    return System.currentTimeMillis() - timestampMaoDireita;
  }
  
  public long getIdadeDadosMaoEsquerda() {
    return System.currentTimeMillis() - timestampMaoEsquerda;
  }
  
  public long getIdadeAnguloEsquerda() {
    return System.currentTimeMillis() - timestampAnguloEsquerda;
  }
  
  public NetworkData getAllData() {
    return new NetworkData(
      distanciaMaoDireita,
      distanciaMaoEsquerda, 
      anguloMaoEsquerda,
      anguloVerticalMaoEsquerda,
      temDadosMaoDireita(),
      temDadosMaoEsquerda(),
      temAnguloMaoEsquerda()
    );
  }

  public boolean isConnected() {
    return connected;
  }
  
  // NOVO: Estatísticas melhoradas
  public String getCommunicationStats() {
    return String.format("Sent: %d | Received: %d | Suppressed: %d", 
                        messagesSent, messagesReceived, messagesSuppressed);
  }
  
  public String getDebugInfo() {
    return String.format(
      "Connected: %s | Right: %.3f (%dms) | Left: %.3f (%dms) | Angle: %.1f° (%dms) | %s",
      connected,
      distanciaMaoDireita, getIdadeDadosMaoDireita(),
      distanciaMaoEsquerda, getIdadeDadosMaoEsquerda(),
      anguloMaoEsquerda, getIdadeAnguloEsquerda(),
      getCommunicationStats()
    );
  }
}
