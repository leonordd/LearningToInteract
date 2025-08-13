import com.hamoid.*;

class CaptureManager {
  private FileManager fileManager;
  private VideoExport videoExport;
  private PrintWriter output;
  private boolean recording = false;
  private boolean timingInitialized = false;

  // Configurações de captura - FIXAS PARA 5 MINUTOS
  private final int DURACAO_FIXA_MS = 300000; // 5 minutos em millisegundos
  private final float VIDEO_FPS = 24.0; // FPS do vídeo final
  
  // Configurações calculadas
  private int targetFrameCount;
  private int capturedFrameCount = 0;
  private int csvLineCount = 0;

  // Timing
  private int tempoInicial;
  private long millisSinceEpoch;
  
  // Controle de sincronização REAL
  private float msPerFrame;
  private float nextFrameTime = 0; // Quando capturar o próximo frame
  private boolean frameReadyToCapture = false;

  // FFmpeg path
  private String ffmpegPath = "/opt/homebrew/bin/ffmpeg";
  private PApplet parent;

  // Cache do último estado para sincronização
  private int lastAnimacao = -1;
  private float lastValorMapeado = -1;
  private float lastT = -1;

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
    
    println("=== CAPTURA COM SINCRONIZAÇÃO REAL ===");
    println("GARANTIA: 1 frame do vídeo = 1 linha do CSV");
    println("Dados salvos apenas quando frame é capturado");
  }

  public void inicializarTiming() {
    if (!timingInitialized) {
      configurarTiming();
      timingInitialized = true;
      nextFrameTime = msPerFrame; // Primeiro frame após msPerFrame
      println("=== TIMING INICIALIZADO ===");
      println("Próximo frame em: " + nextFrameTime + " ms");
    }
  }

  private void calcularConfiguracoes() {
    float duracao_em_segundos = DURACAO_FIXA_MS / 1000.0;
    targetFrameCount = (int)(VIDEO_FPS * duracao_em_segundos);
    msPerFrame = DURACAO_FIXA_MS / (float)targetFrameCount;

    println("=== CONFIGURAÇÃO SINCRONIZADA ===");
    println("Duração: " + duracao_em_segundos + " segundos");
    println("FPS: " + VIDEO_FPS);
    println("Total de frames/linhas: " + targetFrameCount);
    println("Intervalo por frame: " + nf(msPerFrame, 0, 2) + " ms");
    println("IMPORTANTE: CSV terá EXATAMENTE " + targetFrameCount + " linhas");
  }

  private void configurarTiming() {
    tempoInicial = millis();
    millisSinceEpoch = System.currentTimeMillis();
    println("Timing configurado - início em: " + tempoInicial);
  }

  private void verificarFFmpeg() {
    File ffmpegFile = new File(ffmpegPath);
    if (!ffmpegFile.exists()) {
      println("ERRO: FFmpeg não encontrado em: " + ffmpegPath);
    } else {
      println("FFmpeg encontrado: " + ffmpegPath);
    }
  }

  private void inicializarCSV() {
    try {
      int csvCount = fileManager.getNextCsvCount();
      String csvPath = fileManager.getDailyFolder() + "/dados_teclas" + csvCount + ".csv";
      output = createWriter(csvPath);
      output.println("MillisSinceEpoch,TempoVideo,AnimacaoAtual,ValorMapeado,Valor,FrameNumber");
      println("CSV criado: " + csvPath);
      println("NOTA: CSV terá " + targetFrameCount + " linhas (1 por frame)");
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
      videoExport.setFrameRate(VIDEO_FPS);
      videoExport.setQuality(70, 0); // Performance otimizada
      
      videoExport.startMovie();
      recording = true;
      println("Vídeo iniciado: " + videoPath);
    }
    catch (Exception e) {
      println("Erro ao inicializar vídeo: " + e.getMessage());
      recording = false;
    }
  }

  // MÉTODO PRINCIPAL: Atualiza dados a cada frame do programa
  public void atualizarDados(int animacao, float valorMapeado, float t) {
    if (!timingInitialized) return;
    
    // Atualizar cache com dados mais recentes
    lastAnimacao = animacao;
    lastValorMapeado = valorMapeado;
    lastT = t;
    
    // Verificar se é hora de capturar frame
    int tempoDecorrido = millis() - tempoInicial;
    
    if (tempoDecorrido >= nextFrameTime && capturedFrameCount < targetFrameCount) {
      capturarFrameSincronizado();
      
      // Calcular próximo momento de captura
      nextFrameTime += msPerFrame;
    }
    
    // Verificar se completou
    if (capturaCompleta()) {
      finalizar();
    }
  }

  // CAPTURA SINCRONIZADA: Frame + CSV simultaneamente
  private void capturarFrameSincronizado() {
    if (!recording || videoExport == null || output == null) return;
    
    try {
      // 1. CAPTURAR FRAME
      videoExport.saveFrame();
      capturedFrameCount++;
      
      // 2. SALVAR LINHA CSV COM OS MESMOS DADOS
      float tempoVideo = capturedFrameCount * msPerFrame;
      long timestampAtual = System.currentTimeMillis();
      
      output.println(timestampAtual + "," + 
                    nf(tempoVideo, 0, 2) + "," + 
                    lastAnimacao + "," + 
                    nf(lastValorMapeado, 1, 3) + "," + 
                    lastT + "," + 
                    capturedFrameCount);
      output.flush();
      csvLineCount++;
      
      // 3. FEEDBACK
      if (capturedFrameCount % 100 == 0 || capturedFrameCount == targetFrameCount) {
        int tempoReal = millis() - tempoInicial;
        float percentual = (capturedFrameCount * 100.0 / targetFrameCount);
        
        println("===== PROGRESSO SINCRONIZADO =====");
        println("Frame/CSV: " + capturedFrameCount + "/" + targetFrameCount + 
               " (" + nf(percentual, 1, 1) + "%)");
        println("Tempo vídeo: " + nf(tempoVideo/1000, 1, 1) + "s");
        println("Tempo real: " + nf(tempoReal/1000, 1, 1) + "s");
        println("FPS efetivo: " + nf(capturedFrameCount*1000.0/tempoReal, 1, 1));
        
        // VERIFICAÇÃO DE SINCRONIZAÇÃO
        if (capturedFrameCount == csvLineCount) {
          println("✓ SINCRONIZADO: " + capturedFrameCount + " frames = " + csvLineCount + " linhas");
        } else {
          println("⚠ ERRO DE SINCRONIZAÇÃO!");
        }
      }
      
    } catch (Exception e) {
      println("Erro na captura sincronizada: " + e.getMessage());
    }
  }

  private boolean capturaCompleta() {
    return capturedFrameCount >= targetFrameCount;
  }

  public void finalizar() {
    println("\n=== FINALIZAÇÃO ===");
    
    int tempoRealTotal = millis() - tempoInicial;
    float tempoRealSeg = tempoRealTotal / 1000.0;
    float duracacaoVideo = capturedFrameCount / VIDEO_FPS;
    
    // VERIFICAÇÕES FINAIS
    println("Tempo real execução: " + nf(tempoRealSeg, 1, 2) + "s");
    println("Frames capturados: " + capturedFrameCount);
    println("Linhas CSV: " + csvLineCount);
    println("Duração vídeo: " + nf(duracacaoVideo, 1, 2) + "s");
    
    // VERIFICAÇÃO CRÍTICA DE SINCRONIZAÇÃO
    if (capturedFrameCount == csvLineCount && capturedFrameCount == targetFrameCount) {
      println("✅ PERFEITO: " + capturedFrameCount + " frames = " + csvLineCount + " linhas = 5 minutos");
      println("✅ SINCRONIZAÇÃO GARANTIDA: Cada linha CSV corresponde a 1 frame do vídeo");
    } else {
      println("❌ ERRO DE SINCRONIZAÇÃO:");
      println("  Frames: " + capturedFrameCount);
      println("  CSV linhas: " + csvLineCount);
      println("  Target: " + targetFrameCount);
    }

    // Fechar arquivos
    if (output != null) {
      output.flush();
      output.close();
      println("✓ CSV finalizado");
    }

    if (recording && videoExport != null) {
      videoExport.endMovie();
      println("✓ Vídeo finalizado");
      recording = false;
    }

    println("=== CAPTURA CONCLUÍDA ===\n");
    noLoop();
  }

  // GETTERS E MÉTODOS DE COMPATIBILIDADE
  public CaptureInfo getCaptureInfo() {
    int currentTime = timingInitialized ? (millis() - tempoInicial) : 0;
    float tempoVideo = capturedFrameCount * msPerFrame;

    return new CaptureInfo(
      (int)tempoVideo,
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
    return (float)capturedFrameCount / targetFrameCount;
  }

  public boolean isTimingInitialized() {
    return timingInitialized;
  }

  public String getPerformanceInfo() {
    if (!timingInitialized) return "Não inicializado";
    
    int tempoReal = millis() - tempoInicial;
    if (tempoReal <= 0) return "Calculando...";
    
    float fpsReal = (capturedFrameCount * 1000.0) / tempoReal;
    float tempoVideo = capturedFrameCount * msPerFrame / 1000.0;
    
    return "Próximo frame: " + nf((nextFrameTime - tempoReal)/1000.0, 1, 1) + "s" +
           " | Video: " + nf(tempoVideo, 1, 1) + "s/300s" +
           " | Sync: " + capturedFrameCount + "/" + csvLineCount;
  }

  // MÉTODOS ANTIGOS MANTIDOS PARA COMPATIBILIDADE
  public void executarCaptura() {
    // Método vazio - usar atualizarDados() no lugar
  }

  public void executarCapturaSincronizada(int animacao, float valorMapeado, float t) {
    atualizarDados(animacao, valorMapeado, t);
  }

  public void salvarDadosCSV(int animacao, float valorMapeado, float t) {
    atualizarDados(animacao, valorMapeado, t);
  }
}
