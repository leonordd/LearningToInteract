import com.hamoid.*;

class CaptureManager {
  private FileManager fileManager;
  private VideoExport videoExport;
  private PrintWriter output;
  private boolean recording = false;
  private boolean timingInitialized = false;

  // Configurações de captura - FIXAS PARA 5 MINUTOS
  private final int DURACAO_FIXA_MS = 300000; // 5 minutos em millisegundos
  private final float VIDEO_FPS = 30.0; // FPS fixo para o vídeo final
  
  // NOVO: Modo de captura - captura TODOS os frames ou baseado em tempo
  private boolean captureAllFrames = true; // true = captura todos os frames renderizados
  
  // Configurações calculadas
  private int targetFrameCount;
  private int capturedFrameCount = 0;
  private int csvLineCount = 0;

  // Timing
  private int tempoInicial;
  private int startTime;
  private long lastFrameTime = 0;
  private long millisSinceEpoch;
  
  // NOVO: Tempo virtual para sincronização
  private float virtualTimeMs = 0;
  private float msPerFrame;

  // FFmpeg path
  private String ffmpegPath = "/opt/homebrew/bin/ffmpeg";

  private PApplet parent;

  public CaptureManager(FileManager fileManager, PApplet parent) {
    this.fileManager = fileManager;
    this.parent = parent;
  }

  public void inicializar() {
    calcularConfiguracoes();
    verificarFFmpeg();
    fileManager.configurarPastas();
    inicializarCSV();
    inicializarVideo();

    // Não forçar framerate específico para permitir captura de todos os frames
    // frameRate(60); // REMOVIDO - deixa o Processing rodar na velocidade que conseguir
    
    println("=== CAPTURA INICIALIZADA ===");
    println("MODO: Captura de TODOS os frames renderizados");
    println("Cada frame renderizado será gravado no vídeo");
  }

  public void inicializarTiming() {
    if (!timingInitialized) {
      configurarTiming();
      timingInitialized = true;
      println("=== TIMING INICIALIZADO ===");
      println("Tempo inicial definido: " + tempoInicial + " ms");
      println("Modo de captura: " + (captureAllFrames ? "TODOS os frames" : "Baseado em tempo"));
    }
  }

  private void calcularConfiguracoes() {
    // Calcula o número de frames baseado na duração fixa e FPS fixo
    float duracao_em_segundos = DURACAO_FIXA_MS / 1000.0;
    targetFrameCount = (int)(VIDEO_FPS * duracao_em_segundos);
    msPerFrame = DURACAO_FIXA_MS / (float)targetFrameCount;

    println("=== CONFIGURAÇÃO DE VÍDEO (5 MINUTOS FIXOS) ===");
    println("Duração do vídeo: " + duracao_em_segundos + " segundos (5 minutos)");
    println("FPS do vídeo: " + VIDEO_FPS);
    println("Total de frames necessários: " + targetFrameCount);
    println("Tempo por frame: " + nf(msPerFrame, 0, 2) + " ms");
    println("IMPORTANTE: Serão capturados " + targetFrameCount + " frames");
    println("independentemente do tempo real que levar!");
  }

  private void configurarTiming() {
    tempoInicial = millis();
    startTime = millis();
    lastFrameTime = millis();
    millisSinceEpoch = System.currentTimeMillis();
    virtualTimeMs = 0; // Inicializa tempo virtual
    println("Timing configurado - millis atual: " + millis());
  }

  private void verificarFFmpeg() {
    File ffmpegFile = new File(ffmpegPath);
    if (!ffmpegFile.exists()) {
      println("ERRO: FFmpeg não encontrado em: " + ffmpegPath);
    } else {
      println("FFmpeg encontrado em: " + ffmpegPath);
    }
  }

  private void inicializarCSV() {
    try {
      int csvCount = fileManager.getNextCsvCount();
      String csvPath = fileManager.getDailyFolder() + "/dados_teclas" + csvCount + ".csv";
      output = createWriter(csvPath);
      output.println("MillisSinceEpoch,LocalMillisProcessing,AnimacaoAtual,ValorMapeado,Valor,VirtualTime");
      println("Arquivo CSV criado com sucesso: " + csvPath);
    }
    catch (Exception e) {
      println("Erro ao criar CSV: " + e.getMessage());
    }
  }

  private void inicializarVideo() {
    try {
      int videoCount = fileManager.getNextVideoCount();
      String videoPath = fileManager.getDailyFolder() + "/video" + videoCount + ".mp4";

      videoExport = new VideoExport(parent);
      videoExport.setFfmpegPath(ffmpegPath);
      videoExport.setMovieFileName(videoPath);
      videoExport.setFrameRate(VIDEO_FPS); // FPS fixo para o vídeo final
      videoExport.setQuality(85, 0);

      videoExport.startMovie();
      recording = true;
      println("Gravação de vídeo iniciada!");
      println("O vídeo terá exatamente 5 minutos com " + VIDEO_FPS + " FPS");
      println("Serão capturados " + targetFrameCount + " frames no total");
    }
    catch (Exception e) {
      println("Erro ao inicializar vídeo: " + e.getMessage());
      recording = false;
    }
  }

  private void atualizarTiming() {
    if (timingInitialized) {
      millisSinceEpoch = System.currentTimeMillis();
      // Atualiza tempo virtual baseado nos frames capturados
      virtualTimeMs = capturedFrameCount * msPerFrame;
    }
  }

  public void salvarDadosCSV(int animacao, float valorMapeado, float t) {
    if (output != null && timingInitialized) {
      int currentMillis = millis() - tempoInicial;
      output.println(millisSinceEpoch + "," + currentMillis + "," + animacao + "," + 
                    nf(valorMapeado, 1, 3) + "," + t + "," + nf(virtualTimeMs, 0, 2));
      output.flush();
      csvLineCount++;
    }
  }

  private boolean deveCapturarFrame() {
    if (!timingInitialized) {
      return false;
    }
    
    // MODO CAPTURA TODOS OS FRAMES:
    // Captura SEMPRE até atingir o número alvo de frames
    if (captureAllFrames) {
      return capturedFrameCount < targetFrameCount;
    }
    
    // MODO BASEADO EM TEMPO (antigo - mantido como opção):
    else {
      int currentMillis = millis() - tempoInicial;
      if (capturedFrameCount >= targetFrameCount || currentMillis >= DURACAO_FIXA_MS) {
        return false;
      }
      
      float idealCaptureMoment = (capturedFrameCount * msPerFrame);
      return currentMillis >= idealCaptureMoment;
    }
  }

  private void capturarFrame() {
    if (recording && videoExport != null) {
      try {
        videoExport.saveFrame();
        capturedFrameCount++;
        
        // Atualiza tempo virtual
        virtualTimeMs = capturedFrameCount * msPerFrame;
        
        // Feedback mais frequente para acompanhar progresso
        if (capturedFrameCount % 50 == 0 || capturedFrameCount == targetFrameCount) {
          int tempoReal = millis() - tempoInicial;
          float percentFrames = (capturedFrameCount * 100.0 / targetFrameCount);
          float tempoVirtualSeg = virtualTimeMs / 1000.0;
          float tempoRealSeg = tempoReal / 1000.0;
          
          println("===== PROGRESSO =====");
          println("Frames: " + capturedFrameCount + " / " + targetFrameCount + 
                 " (" + nf(percentFrames, 1, 1) + "%)");
          println("Tempo virtual do vídeo: " + nf(tempoVirtualSeg, 1, 1) + " seg");
          println("Tempo real decorrido: " + nf(tempoRealSeg, 1, 1) + " seg");
          
          if (tempoRealSeg > 0) {
            float fpsReal = capturedFrameCount / tempoRealSeg;
            println("FPS real de captura: " + nf(fpsReal, 1, 1));
            
            if (fpsReal < 10) {
              println("AVISO: Captura lenta! Considere reduzir a complexidade visual.");
            }
          }
        }
      }
      catch (Exception e) {
        println("Erro ao salvar frame: " + e.getMessage());
      }
    }
  }

  public CaptureInfo getCaptureInfo() {
    int currentMillis = timingInitialized ? (millis() - tempoInicial) : 0;
    
    // Usa tempo virtual para mostrar progresso do vídeo
    int videoTimeMs = (int)virtualTimeMs;

    return new CaptureInfo(
      videoTimeMs, // Mostra tempo do VÍDEO, não tempo real
      DURACAO_FIXA_MS,
      capturedFrameCount,
      targetFrameCount,
      csvLineCount,
      recording,
      VIDEO_FPS,
      msPerFrame,
      fileManager.getDateFormatted()
    );
  }

  public boolean isRecording() {
    return recording;
  }

  public float getProgress() {
    // Progresso baseado nos frames capturados, não no tempo real
    return (float)capturedFrameCount / targetFrameCount;
  }

  public float getFrameProgress() {
    return (float)capturedFrameCount / targetFrameCount;
  }

  public void executarCaptura() {
    try {
      if (!timingInitialized) {
        return;
      }
      
      atualizarTiming();

      // SEMPRE tenta capturar se deve
      if (deveCapturarFrame()) {
        capturarFrame();
      }

      if (capturaCompleta()) {
        finalizar();
      }
    }
    catch (Exception e) {
      println("Erro na captura: " + e.getMessage());
    }
  }

  private boolean capturaCompleta() {
    if (!timingInitialized) return false;
    
    // Completa quando captura todos os frames necessários
    return capturedFrameCount >= targetFrameCount;
  }

  public void finalizar() {
    println("\n=== FINALIZANDO CAPTURA ===");
    
    // Informações finais
    int tempoRealTotal = timingInitialized ? (millis() - tempoInicial) : 0;
    float tempoRealSeg = tempoRealTotal / 1000.0;
    float tempoVideoSeg = (capturedFrameCount / VIDEO_FPS);
    
    println("Frames capturados: " + capturedFrameCount + " de " + targetFrameCount);
    println("Duração do vídeo: " + nf(tempoVideoSeg, 1, 2) + " segundos");
    println("Tempo real de captura: " + nf(tempoRealSeg, 1, 2) + " segundos");
    
    if (capturedFrameCount == targetFrameCount) {
      println("SUCESSO: Vídeo terá exatamente 5 minutos!");
    } else if (capturedFrameCount < targetFrameCount) {
      println("AVISO: Capturados apenas " + capturedFrameCount + " frames.");
      println("O vídeo terá " + nf(tempoVideoSeg, 1, 2) + " segundos ao invés de 300.");
    }

    if (output != null) {
      output.flush();
      output.close();
      println("CSV fechado com sucesso");
    }

    if (recording && videoExport != null) {
      videoExport.endMovie();
      println("Vídeo finalizado com sucesso");
      recording = false;
    }

    println("=== CAPTURA CONCLUÍDA ===\n");
    noLoop();
  }

  // Método para alternar entre modos de captura
  public void setCaptureMode(boolean captureAll) {
    captureAllFrames = captureAll;
    println("Modo de captura alterado para: " + 
           (captureAllFrames ? "TODOS os frames" : "Baseado em tempo"));
  }

  // Método para configuração personalizada
  public void configurarVideoPersonalizado(int novoTargetFrames, int novoMaxMillis) {
    // Ignora novoMaxMillis pois sempre será 5 minutos
    println("NOTA: Duração fixa em 5 minutos. Ajustando apenas número de frames.");
    
    if (novoTargetFrames > 0) {
      // Calcula o FPS necessário para ter 5 minutos com o número de frames desejado
      float novoFPS = novoTargetFrames / 300.0; // 300 segundos = 5 minutos
      println("Para " + novoTargetFrames + " frames em 5 minutos, FPS seria: " + nf(novoFPS, 1, 2));
      
      // Mantém o targetFrameCount conforme solicitado
      targetFrameCount = novoTargetFrames;
      msPerFrame = DURACAO_FIXA_MS / (float)targetFrameCount;
      
      println("Configuração ajustada:");
      println("- Frames alvo: " + targetFrameCount);
      println("- Tempo por frame: " + nf(msPerFrame, 0, 2) + " ms");
    }
  }
  
  public boolean isTimingInitialized() {
    return timingInitialized;
  }
  
  // NOVO: Método para obter tempo virtual (útil para sincronização)
  public float getVirtualTimeMs() {
    return virtualTimeMs;
  }
  
  // NOVO: Método para obter informações de performance
  public String getPerformanceInfo() {
    if (!timingInitialized) return "Não inicializado";
    
    int tempoReal = millis() - tempoInicial;
    if (tempoReal <= 0) return "Calculando...";
    
    float fpsReal = (capturedFrameCount * 1000.0) / tempoReal;
    float tempoEstimado = (targetFrameCount / fpsReal);
    
    return "FPS real: " + nf(fpsReal, 1, 1) + 
           " | Tempo estimado total: " + nf(tempoEstimado, 1, 1) + " seg";
  }
}
