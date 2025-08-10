import com.hamoid.*;

class CaptureManager {
  private FileManager fileManager;
  //private Toy toy;
  private VideoExport videoExport;
  private PrintWriter output;
  private boolean recording = false;
  private boolean timingInitialized = false; // NOVO: flag para controlar se o timing foi inicializado

  // Configurações de captura
  private int targetFrameCount = 5000;
  private int max_millis = 300000;
  private float calculatedVideoFrameRate;
  private float frameInterval;
  private int capturedFrameCount = 0;
  private int csvLineCount = 0;

  // Timing
  private int tempoInicial;
  private int startTime;
  private long lastFrameTime = 0;
  private long millisSinceEpoch;

  // FFmpeg path
  private String ffmpegPath = "/opt/homebrew/bin/ffmpeg";

  private PApplet parent;

  public CaptureManager(FileManager fileManager, PApplet parent) {
    this.fileManager = fileManager;
    //this.toy = toy;
    this.parent = parent;
  }

  public void inicializar() {
    calcularConfiguracoes();
    // REMOVIDO: configurarTiming(); - agora só é chamado quando necessário
    verificarFFmpeg();
    fileManager.configurarPastas();
    inicializarCSV();
    inicializarVideo();

    frameRate(60);
    println("=== CAPTURA INICIALIZADA ===");
  }

  // NOVO: Método público para inicializar timing quando o modo for selecionado
  public void inicializarTiming() {
    if (!timingInitialized) {
      configurarTiming();
      timingInitialized = true;
      println("=== TIMING INICIALIZADO ===");
      println("Tempo inicial definido: " + tempoInicial + " ms");
    }
  }

  private void calcularConfiguracoes() {
    float duracao_em_segundos = max_millis / 1000.0;
    calculatedVideoFrameRate = targetFrameCount / duracao_em_segundos;
    frameInterval = (float)max_millis / targetFrameCount;

    println("=== CONFIGURAÇÃO PERSONALIZADA DE VÍDEO ===");
    println("Duração do programa: " + duracao_em_segundos + " segundos");
    println("Frames desejados: " + targetFrameCount);
    println("FPS calculado automaticamente: " + nf(calculatedVideoFrameRate, 0, 2));
    println("Intervalo entre frames: " + nf(frameInterval, 0, 2) + " ms");
  }

  private void configurarTiming() {
    tempoInicial = millis();
    startTime = millis();
    lastFrameTime = millis();
    millisSinceEpoch = System.currentTimeMillis();
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
      //output.println("MillisSinceEpoch,LocalMillisProcessing,Valor");
      output.println("MillisSinceEpoch,LocalMillisProcessing,AnimacaoAtual,ValorMapeado,Valor");
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
      videoExport.setFrameRate(calculatedVideoFrameRate);
      videoExport.setQuality(85, 0);

      videoExport.startMovie();
      recording = true;
      println("Gravação de vídeo iniciada com sucesso!");
    }
    catch (Exception e) {
      println("Erro ao inicializar vídeo: " + e.getMessage());
      recording = false;
    }
  }

  private void atualizarTiming() {
    // MODIFICADO: só atualiza se o timing foi inicializado
    if (timingInitialized) {
      millisSinceEpoch = System.currentTimeMillis();
    }
  }

  public void salvarDadosCSV(int animacao, float valorMapeado, float t ) {
    // MODIFICADO: só salva dados se o timing foi inicializado
    if (output != null && timingInitialized) {
      int currentMillis = millis() - tempoInicial;
      output.println(millisSinceEpoch + "," + currentMillis + "," + animacao + "," + nf(valorMapeado, 1, 3) + "," + t);
      output.flush();
      csvLineCount++;
    }
  }

  private boolean deveCapturarFrame() {
    // MODIFICADO: só captura frames se o timing foi inicializado
    if (!timingInitialized || capturedFrameCount >= targetFrameCount) {
      return false;
    }
    
    int currentMillis = millis() - tempoInicial;
    float idealCaptureMoment = (capturedFrameCount * frameInterval);

    return currentMillis >= idealCaptureMoment ||
      (currentMillis >= max_millis && capturedFrameCount < targetFrameCount - 1);
  }

  private void capturarFrame() {
    if (recording && videoExport != null) {
      try {
        videoExport.saveFrame();
      }
      catch (Exception e) {
        println("Erro ao salvar frame: " + e.getMessage());
      }
    }

    capturedFrameCount++;

    if (capturedFrameCount % 100 == 0) {
      println("Frames capturados: " + capturedFrameCount + " / " + targetFrameCount);
    }
  }

  public CaptureInfo getCaptureInfo() {
    // MODIFICADO: retorna tempo 0 se timing não foi inicializado
    int currentMillis = timingInitialized ? (millis() - tempoInicial) : 0;

    return new CaptureInfo(
      currentMillis,
      max_millis,
      capturedFrameCount,
      targetFrameCount,
      csvLineCount,
      recording,
      calculatedVideoFrameRate,
      frameInterval,
      fileManager.getDateFormatted()
      );
  }

  public boolean isRecording() {
    return recording;
  }

  public float getProgress() {
    if (!timingInitialized) return 0;
    int currentMillis = millis() - tempoInicial;
    return (float)currentMillis / max_millis;
  }

  public float getFrameProgress() {
    return (float)capturedFrameCount / targetFrameCount;
  }

  public void executarCaptura() {
    try {
      // MODIFICADO: só executa captura se timing foi inicializado
      if (!timingInitialized) {
        return; // Sai early se timing não foi inicializado
      }
      
      atualizarTiming();

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
    
    int currentMillis = millis() - tempoInicial;
    return capturedFrameCount >= targetFrameCount || currentMillis >= max_millis;
  }

  private void finalizar() {
    println("Finalizando captura...");

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

    noLoop();
  }

  // Método para configuração personalizada
  public void configurarVideoPersonalizado(int novoTargetFrames, int novoMaxMillis) {
    targetFrameCount = novoTargetFrames;
    max_millis = novoMaxMillis;
    calcularConfiguracoes();
  }
  
  // NOVO: Método para verificar se timing foi inicializado
  public boolean isTimingInitialized() {
    return timingInitialized;
  }
}
