///version 4 inicializa o python através do processing
import javax.swing.JOptionPane;
NetworkManager network;
ArrayList<Circle> circles = new ArrayList<Circle>();
ArrayList<ArrayList<Keyframe[]>> allAnimationKeyframes = new ArrayList<ArrayList<Keyframe[]>>();
CaptureManager captureManager;
FileManager fileManager;
ModeManager modeManager;
DebugWindow debug;
PFont title;

// Sistema de cache mais inteligente
private NetworkData lastNetworkData = null;
private float lastT = -1;
private int lastCurrentAnimation = -1;
private boolean forceUpdate = true;

// Controlo de animação melhorado
private int currentAnimation = 0;
private int targetAnimation = 0; // Animação que queremos alcançar
private float animationTransition = 0; // 0 = animação atual, 1 = animação alvo
private boolean isTransitioning = false;

// Controlo de gestos com debounce
private boolean gestureActive = false;
private long lastGestureTime = 0;
private static final long GESTURE_DEBOUNCE = 300; // 300ms debounce

// Sistema de contagem de ocorrências de animações
int[] animationOccurrences;
int totalFrames = 0; // Contador total de frames processados
boolean countingEnabled = true; // Flag para ativar/desativar contagem
float centerX, centerY;

KeyframeInfo[][] allKeyframeData = {
  {
    new KeyframeInfo(0, "Anim0 - Centro", "Círculos no centro"),
    new KeyframeInfo(1, "Anim0 - Expansão", "Expansão central"),
    new KeyframeInfo(2, "Anim0 - Orbital", "Formação orbital")
  },
  {
    new KeyframeInfo(0, "Anim1 - Linha", "Círculos em linha"),
    new KeyframeInfo(1, "Anim1 - Curva", "Movimento curvado"),
    new KeyframeInfo(2, "Anim1 - Espiral", "Formação espiral"),
    new KeyframeInfo(3, "Anim1 - Final", "Posição final")
  },
  {
    new KeyframeInfo(0, "Anim2 - Grid", "Formação em grid"),
    new KeyframeInfo(1, "Anim2 - Ondas", "Movimento ondulatório"),
    new KeyframeInfo(2, "Anim2 - Orbital Avançado", "Rotação radial 360º"),
    new KeyframeInfo(3, "Anim2 - Retorno", "Retorno ao centro")
  },
  {
    new KeyframeInfo(0, "Anim3 - Spiral Orbit", "Círculos em órbita espiral"),
    new KeyframeInfo(1, "Anim3 - Loop", "Caminho em loop"),
    new KeyframeInfo(2, "Anim3 - Cluster", "Agrupamento central"),
    new KeyframeInfo(3, "Anim3 - Fim", "Finalização da animação")
  }
};

float r = 100, espaco;
int n_circles = 7; //5
int maxAnimation = 4;

float[][] allTransitions = {
  {0.0, 5.0, 9.0, 11.0}, // Animação 0
  {0.0, 2.0, 5.0, 8.0, 11.0}, // Animação 1
  {0.0, 3.0, 5.0, 7.0, 9.0, 11.0}, // Animação 2
  {0.0, 3.0, 6.0, 9.0, 11.0}           // Animação 2 → 6 keyframes (kfs[0] a kfs[5])
};

String [] animationNames = {"The natural symmetry", "Transcending", "Looking into the void", "Inception"};

float minD = 0.05, minDe = 0.05;
float maxD = 0.3, maxDe = 0.3;

boolean estavaNoModoJogo = false;
Process processoPython = null;

void setup() {
  size(800, 1000);
  surface.setLocation(width-800, height-1000);
  background(0);
  surface.setTitle("Learning To Interact");
  modeManager = new ModeManager();
  network = new NetworkManager();
  fileManager = new FileManager();
  captureManager = new CaptureManager(fileManager, this);
  debug = new DebugWindow();
  PApplet.runSketch(new String[] { "DebugWindow" }, debug);
  title = createFont("Boldonse/Boldonse-Regular.ttf", 50);
  centerX = width/2;
  centerY = height/2 - 50;

  // Inicializar array de contagem de animações
  animationOccurrences = new int[maxAnimation];

  // Configurar modo com base no input do user
  modeManager.configurarModo();

  // Inicializar baseado no modo selecionado
  if (modeManager.isModoCaptura()) {
    initializeAllAnimations(n_circles);
    captureManager.inicializar();
    //captureManager.configurarVideoPersonalizado(300000, 30.0); // 5 minutos (300000ms), 30 FPS
  } else if (modeManager.isModoJogo()) {
    network.beginConnection();
    initializeAllAnimations(n_circles);
    // (opcional) alterar frames e duração da gravação
  } else if (modeManager.isModoJogoComTreino()) {
    network.beginConnection();
    initializeAllAnimations(n_circles);
  }

}

void draw() {
  background(0);
  textFont(title);
  textSize(50);

  text(animationNames[currentAnimation].toUpperCase(), 25, height-40);
  clip(centerX, centerY, width-50, height-150);

  if (modeManager.isModoCaptura()) {
    //inicializar aoenas quando entrar no modo
    if (!captureManager.isTimingInitialized()) {
      captureManager.inicializarTiming();
      println("Timing inicializado para modo captura!");
    }
    // animação: transição entre keyframes com o mapeamento do mouseY. Ou seja mouseY, 0, height, inicio animação kfs = 0, fim animação kfs = max
    // Processar controlo de keyframes (mão direita)

    float t = map(mouseY, 0, height, 0.0, 11.0);
    updateAndDrawAnimation(t, mouseX, 0, width);
    updateAnimationCounter();

    int valor = round(map(t, 0.0, 11.0, 0, 8));

    captureManager.atualizarDados(currentAnimation, valor, t);
    updateDebugWindow_Captura(t, valor);

  } else if (modeManager.isModoJogo()) {

    if (!estavaNoModoJogo) {
      println("Entrou no modo jogo. A iniciar script Python...");
      iniciarScriptPython();
      estavaNoModoJogo = true;

      if (!captureManager.isTimingInitialized()) {
        captureManager.inicializarTiming();
        println("Timing inicializado para modo jogo!");
      }
    }

    NetworkData networkData = network.getAllData();
    boolean dataChanged = hasDataChanged(networkData);
    float td = 0; //transition esquerda
    float te = 0; //transition direita
    float e = 0;

    processAnimationControl(networkData);

    float[] transitions = allTransitions[currentAnimation];

    if (networkData.validMaoDireita && networkData.distanciaMaoDireita >= 0) {
      float d = networkData.distanciaMaoDireita;
      //float t;

      if (minD == maxD) {
        td = transitions[1];
      } else {
        td = map(d, minD, maxD, transitions[0], transitions[transitions.length - 1]);
      }

      td = constrain(td, transitions[0], transitions[transitions.length - 1]);
    }

    if (networkData.validMaoEsquerda && networkData.distanciaMaoEsquerda >= 0) {
      minDe = 0.05;
      maxDe = 0.3;
      e = networkData.distanciaMaoEsquerda;
    }

    //print(te);
    updateAndDrawAnimation(td, e, minDe, maxDe);
    updateAnimationCounter();

    lastNetworkData = networkData;
    lastT = td;
    forceUpdate = false;

    network.checkIncomingMessages();

    if (network.isConnected()) { //se a network estiver connected guardar dados e dar update na janela de debug
      // Enviar informações para a janela de debug
      network.sendAnimationData(currentAnimation, td, isTransitioning, targetAnimation);
      updateDebugWindow_Jogo(networkData);

      int valor = round(map(td, 0.0, 11.0, 0, 8));
      captureManager.atualizarDados(currentAnimation, valor, td);

      //captureManager.executarCaptura(); //captura apenas o ecrã
    }
  } else if (modeManager.isModoJogoComTreino()) {
    print("A");
    
    //iniciar script de python com path "1input_not_pred.py"
    //iniciarScriptPythonTreino();
    //receber valor da animação (ou seja as classes predicted 1,2,3,4)
    //update currentAnimation para o valor da animação. Ou seja selecionar a animação consoante o (valor que vem da rede - 1)
    
  } else {
    if (estavaNoModoJogo) {
      println("Saiu do modo jogo. A terminar script Python...");
      terminarScriptPython();
      estavaNoModoJogo = false;
    }
  }

  noClip();
}

boolean hasDataChanged(NetworkData current) {
  if (lastNetworkData == null) return true;

  // Verificar mudanças significativas
  boolean animationChanged = (current.validAnguloEsquerda != lastNetworkData.validAnguloEsquerda) ||
    (current.validAnguloEsquerda &&
    abs(current.anguloMaoEsquerda - lastNetworkData.anguloMaoEsquerda) > 2);

  boolean keyframeChanged = (current.validMaoDireita != lastNetworkData.validMaoDireita) ||
    (current.validMaoDireita &&
    abs(current.distanciaMaoDireita - lastNetworkData.distanciaMaoDireita) > 0.01);

  return animationChanged || keyframeChanged || isTransitioning;
}

void processAnimationControl(NetworkData data) {
  long currentTime = System.currentTimeMillis();

  if (data.validAnguloEsquerda) {
    float angulo = data.anguloMaoEsquerda;
    boolean gestureDetected = (angulo >= 0 && angulo <= 50);

    // Debounce do gesto
    if (gestureDetected && !gestureActive &&
      (currentTime - lastGestureTime > GESTURE_DEBOUNCE)) {

      targetAnimation = (currentAnimation + 1) % maxAnimation;
      isTransitioning = true;
      animationTransition = 0;
      lastGestureTime = currentTime;

      println("Gesture detected! Transitioning to animation: " + targetAnimation);
    }

    gestureActive = gestureDetected;
  } else {
    gestureActive = false;
  }

  // Processar transição suave entre animações
  if (isTransitioning) {
    animationTransition += 0.05; // Velocidade da transição
    if (animationTransition >= 1.0) {
      currentAnimation = targetAnimation;
      isTransitioning = false;
      animationTransition = 0;
    }
  }
}

float processKeyframeControl(NetworkData data) {
  float[] transitions = allTransitions[currentAnimation];

  if (data.validMaoDireita && data.distanciaMaoDireita >= 0) {
    float d = data.distanciaMaoDireita;
    float t;

    if (minD == maxD) {
      t = transitions[1];
    } else {
      t = map(d, minD, maxD, transitions[0], transitions[transitions.length - 1]);
    }

    return constrain(t, transitions[0], transitions[transitions.length - 1]);
  } else {
    // Estado padrão (keyframe 0)
    return transitions[0];
  }
}

void updateAndDrawAnimation(float t, float inputValue, float inputMin, float inputMax) {
  resetMatrix();
  blendMode(ADD);

  //println("InputValue: " + inputValue);
  //println("InputMin: " + inputMin);
  //println("InputMax: " + inputMax);

  ArrayList<Keyframe[]> currentAnimationKeyframes = allAnimationKeyframes.get(currentAnimation);
  int currentKeyframeIndex = 0;

  for (int i = 0; i < circles.size(); i++) {
    Keyframe[] kfs = currentAnimationKeyframes.get(i);

    // Encontrar keyframes para interpolação
    Keyframe currentKf = null;
    Keyframe nextKf = null;

    for (int j = 0; j < kfs.length - 1; j++) {
      if (t >= kfs[j].time && t <= kfs[j + 1].time) {
        currentKf = kfs[j];
        nextKf = kfs[j + 1];
        if (i == 0) currentKeyframeIndex = j;
        break;
      }
    }

    if (currentKf == null) {
      currentKf = kfs[kfs.length - 1];
      nextKf = kfs[kfs.length - 1];
      if (i == 0) currentKeyframeIndex = kfs.length - 1;
    }

    // Interpolação suave
    float amt = 0;
    if (nextKf.time != currentKf.time) {
      amt = (t - currentKf.time) / (nextKf.time - currentKf.time);
      amt = constrain(amt, 0, 1);
      amt = smoothstep(amt); // Função de suavização personalizada
    }

    // Aplicar transformações
    Circle circle = circles.get(i);
    circle.x = lerp(currentKf.pos.x, nextKf.pos.x, amt);
    circle.y = lerp(currentKf.pos.y, nextKf.pos.y, amt);

    //NOVA TRANSIÇÃO
    if (currentAnimation == 0) {
      pushMatrix();
      resetMatrix();
      float distortion;

      if (inputValue == 0) {
        distortion = 0;
      } else {
        distortion = map(inputValue, inputMin, inputMax, -PI/4, PI/4); // shear angle
        distortion = constrain(distortion, -PI/4, PI/4);
      }
      //println("distortion: " + distortion);
      shearX(distortion);

      circle.setSize(lerp(currentKf.w, nextKf.w, amt), lerp(currentKf.h, nextKf.h, amt));
      circle.display();
      popMatrix();
    } else if (currentAnimation == 1) { //rotação
      float ang = map(inputValue, inputMin, inputMax, 0, TWO_PI);
      ang = constrain (ang, 0, TWO_PI);
      resetMatrix();
      pushMatrix();

      circle.setRotation(0);
      translate(centerX, centerY);
      rotate(ang);
      translate(-centerX, -centerY);

      circle.setSize(lerp(currentKf.w, nextKf.w, amt), lerp(currentKf.h, nextKf.h, amt));
      circle.display();
      popMatrix();
    } else if (currentAnimation == 2) { //aumentar e diminuir a animação (raio central), width e height
      pushMatrix();
      resetMatrix();
      float baseW = lerp(currentKf.w, nextKf.w, amt);
      float baseH = lerp(currentKf.h, nextKf.h, amt);

      // Controlado por mouseX em tempo real
      float scale = map(inputValue, inputMin, inputMax, 0.3, 2.0);
      scale = constrain(scale, 0.2, 1.5);

      circle.setSize(baseW * scale, baseH * scale);
      circle.setRotation(lerpAngle(currentKf.rotation, nextKf.rotation, amt));
      circle.display();
      popMatrix();
    } else if (currentAnimation == 3) { //aumenta o height apenas com o petala scale
      pushMatrix();
      resetMatrix();
      float baseW = lerp(currentKf.w, nextKf.w, amt);
      float baseH = lerp(currentKf.h, nextKf.h, amt);

      // Controlado por mouseX em tempo real
      float petalaScale = map(inputValue, inputMin, inputMax, 0.2, 1.5);
      petalaScale = constrain(petalaScale, 0.2, 1.5);

      circle.setSize(baseW, baseH * petalaScale);
      circle.setRotation(lerpAngle(currentKf.rotation, nextKf.rotation, amt));
      circle.display();
      popMatrix();
    }
  }


  blendMode(NORMAL);
}

void drawCircles() {
  blendMode(ADD);
  for (Circle circle : circles) {
    circle.display();
  }
  blendMode(NORMAL);
}

void drawOptimizedUI(NetworkData data) {
  // UI simplificada e otimizada
  fill(255);
  textSize(14);
  text("Animação: " + currentAnimation + (isTransitioning ? " → " + targetAnimation : ""), 20, 20);
  text("FPS: " + nf(frameRate, 0, 1), 20, 40);

  // Status das mãos com cores
  if (data.validMaoDireita) {
    fill(0, 255, 0);
    text("Mão Direita: " + nf(data.distanciaMaoDireita, 1, 3), 20, 60);
  } else {
    fill(255, 100, 100);
    text("Mão Direita: Inativa", 20, 60);
  }

  if (data.validAnguloEsquerda) {
    fill(0, 255, 0);
    text("Mão Esquerda: " + nf(data.anguloMaoEsquerda, 1, 1) + "°", 20, 80);
  } else {
    fill(255, 100, 100);
    text("Mão Esquerda: Inativa", 20, 80);
  }

  // Status da conexão
  fill(network.isConnected() ? color(0, 255, 0) : color(255, 0, 0));
  text(" " + (network.isConnected() ? "Conectado" : "Desconectado"), 20, 100);

  // Debug info (opcional)
  if (keyPressed && key == 'd') {
    fill(255, 255, 0);
    textSize(10);
    text(network.getDebugInfo(), 20, height - 20);
  }
}

// Função de suavização personalizada
float smoothstep(float x) {
  return x * x * (3.0 - 2.0 * x);
}

void initializeAllAnimations(int numCircles) {
  circles.clear();
  allAnimationKeyframes.clear();
  espaco = TWO_PI/numCircles;

  // Inicializar círculos
  for (int i = 0; i < numCircles; i++) {
    circles.add(new Circle(100, 100, 50, 50, color(255, 100, 0), color(255, 100, 255)));
  }

  // Criar keyframes (mesmo código anterior)
  for (int animIndex = 0; animIndex < maxAnimation; animIndex++) {
    ArrayList<Keyframe[]> animationKeyframes = new ArrayList<Keyframe[]>();

    for (int i = 0; i < numCircles; i++) {
      Keyframe[] kfs = new Keyframe[2];

      if (animIndex == 0) {
        kfs = new Keyframe[3];
        float prof = map(mouseX, 0, width, 0, TWO_PI);
        rotate(prof);

        kfs[0] = new Keyframe(allTransitions[animIndex][0], new PVector(centerX, centerY), 40, 40, 0);

        float baseAngle = espaco * i;
        float r1 = 80;
        kfs[1] = new Keyframe(allTransitions[animIndex][1], new PVector(centerX + r1 * cos(baseAngle), centerY + r1 * sin(baseAngle)), 100, 100, 0);

        float angle3 = baseAngle + PI/2;
        float r2 = 200;
        kfs[2] = new Keyframe(allTransitions[animIndex][2], new PVector(centerX + r2 * cos(angle3), centerY + r2 * sin(angle3)), 280, 280, 0);
      } else if (animIndex == 1) {
        kfs = new Keyframe[5];
        float angle3 = espaco * i + PI/2;
        kfs[0] = new Keyframe(allTransitions[animIndex][0], new PVector(centerX, centerY), 40, 40, 0);
        kfs[1] = new Keyframe(allTransitions[animIndex][1], new PVector(centerX + cos(angle3), centerY + sin(angle3)), 200, 100, 0);
        kfs[2] = new Keyframe(allTransitions[animIndex][2], new PVector(centerX + cos(angle3), centerY + i * 100), 200 + 100 * i, 100, 0);
        float verticalOffset, circleWidth;
        if (i == 0) {
          verticalOffset = 0;
          circleWidth = 200;
        } else {
          float mirrorDirection = ((i - 1) % 2 == 0) ? 1 : -1;
          float distance = ((i - 1) / 2 + 1) * 150;
          verticalOffset = distance * mirrorDirection;
          int symmetricPair = (i - 1) / 2;
          circleWidth = 400 + symmetricPair * 200;
        }
        kfs[3] = new Keyframe(allTransitions[animIndex][3], new PVector(centerX + cos(angle3), centerY + verticalOffset), circleWidth, 100, 0);
        kfs[4] = new Keyframe(allTransitions[animIndex][4], new PVector(centerX + cos(angle3), centerY - i * 100), 200 + 100 * i, 100, 0);
      } else if (animIndex == 2) {
        int numKfs = allTransitions[animIndex].length;
        kfs = new Keyframe[numKfs];

        float r1 = 80;
        float radius = r1 * i;
        float radiusOffset = r1;

        float angleStep = TWO_PI / numCircles; // distribuição circular
        float baseAngle = angleStep * i;

        float posX, posY;

        // Keyframe 0 (centro)
        kfs[0] = new Keyframe(allTransitions[animIndex][0], new PVector(centerX, centerY), 40, 40, 0);

        // Restantes keyframes com deslocamentos orbitais de PI/2
        for (int j = 1; j < numKfs; j++) {
          float deslocamento = HALF_PI * (j - 1); // j-1 porque o primeiro é estático no centro
          float angle = baseAngle + deslocamento;

          if (i % 2 == 0) {
            posX = centerX;
            posY = centerY;
          } else {
            posX = centerX + cos(angle) * radiusOffset;
            posY = centerY + sin(angle) * radiusOffset;
          }

          kfs[j] = new Keyframe(allTransitions[animIndex][j], new PVector(posX, posY), radius, radius, 0);
        }
      } else if (animIndex == 3) {
        int numKfs = allTransitions[animIndex].length;
        kfs = new Keyframe[numKfs];

        float r1 = 80;
        float radius = r1 * i;
        float radiusOffset = r1 * (1 + i * 0.2);
        float angleStep = TWO_PI / numCircles;
        float baseAngle = angleStep * i;

        kfs[0] = new Keyframe(allTransitions[animIndex][0], new PVector(centerX, centerY), 40, 40, 0);

        for (int j = 1; j < numKfs; j++) {
          float deslocamento = HALF_PI * j;
          float angle = baseAngle + deslocamento;

          float posX = centerX + cos(angle) * radiusOffset;
          float posY = centerY + sin(angle) * radiusOffset;

          kfs[j] = new Keyframe(allTransitions[animIndex][j], new PVector(posX, posY), radius, radius, angle);
        }
      }
      animationKeyframes.add(kfs);
    }
    allAnimationKeyframes.add(animationKeyframes);
  }
  forceUpdate = true;
}

void addCircle() {
  circles.add(new Circle(100, 100, 50, 50, color(0, 150, 255), color(255, 100, 0)));
  initializeAllAnimations(circles.size());
}

void removeCircle() {
  if (circles.size() > 1) {
    circles.remove(circles.size() - 1);
    initializeAllAnimations(circles.size());
  }
}

float lerpAngle(float a, float b, float amt) {
  float diff = b - a;
  while (diff > PI) diff -= TWO_PI;
  while (diff < -PI) diff += TWO_PI;
  return a + diff * amt;
}

// ==================== SISTEMA DE CONTAGEM DE ANIMAÇÕES ====================

//Incrementa o contador da animação atual
//Deve ser chamada a cada frame em draw()

void updateAnimationCounter() {
  if (countingEnabled && currentAnimation >= 0 && currentAnimation < maxAnimation) {
    animationOccurrences[currentAnimation]++;
    totalFrames++;
  }
}

// Retorna as estatísticas das animações
String getAnimationStats() {
  StringBuilder stats = new StringBuilder();
  stats.append("=== ESTATÍSTICAS DAS ANIMAÇÕES ===\n");
  stats.append("Total de frames: ").append(totalFrames).append("\n\n");

  for (int i = 0; i < maxAnimation; i++) {
    float percentage = totalFrames > 0 ? (animationOccurrences[i] * 100.0f / totalFrames) : 0;
    stats.append("Animação ").append(i).append(": ")
      .append(animationOccurrences[i]).append(" frames (")
      .append(nf(percentage, 1, 2)).append("%)\n");
  }

  return stats.toString();
}

//Imprime as estatísticas no console
void printAnimationStats() {
  println(getAnimationStats());
}

//Salva as estatísticas num arquivo de texto

void saveAnimationStats(String filename) {
  if (fileManager != null) {
    try {
      String[] lines = getAnimationStats().split("\n");
      saveStrings(filename, lines);
      println("Estatísticas salvas em: " + filename);
    }
    catch (Exception e) {
      println("Erro ao salvar estatísticas: " + e.getMessage());
    }
  } else {
    try {
      String[] lines = getAnimationStats().split("\n");
      saveStrings(filename, lines);
      println("Estatísticas salvas em: " + filename);
    }
    catch (Exception e) {
      println("Erro ao salvar estatísticas: " + e.getMessage());
    }
  }
}

//Reset dos contadores
void resetAnimationCounters() {
  for (int i = 0; i < maxAnimation; i++) {
    animationOccurrences[i] = 0;
  }
  totalFrames = 0;
  println("Contadores resetados!");
}

//Ativa/desativa a contagem
void toggleCounting() {
  countingEnabled = !countingEnabled;
  println("Contagem " + (countingEnabled ? "ativada" : "desativada"));
}


//Desenha as estatísticas na tela (UI)
void drawAnimationStats(float x, float y) {
  fill(255, 200); // Texto semi-transparente
  textSize(12);

  text("Total frames: " + totalFrames, x, y);

  for (int i = 0; i < maxAnimation; i++) {
    float percentage = totalFrames > 0 ? (animationOccurrences[i] * 100.0f / totalFrames) : 0;

    // Destacar a animação atual
    if (i == currentAnimation) {
      fill(0, 255, 0); // Verde para animação atual
    } else {
      fill(255, 200); // Branco semi-transparente
    }

    text("Anim " + i + ": " + animationOccurrences[i] + " (" + nf(percentage, 1, 1) + "%)",
      x, y + 20 + (i * 15));
  }
}

void updateDebugWindow_Captura(float t, int valor) {
  StringBuilder debugInfo = new StringBuilder();

  // Informações da animação atual
  debugInfo.append("=== MODO CAPTURA ===\n");
  debugInfo.append("Animação: ").append(animationNames[currentAnimation]).append("\n");
  debugInfo.append("Anim ID: ").append(currentAnimation + 1).append("\n");
  debugInfo.append("Tempo (t): ").append(nf(t, 1, 2)).append("\n");
  debugInfo.append("Valor mapeado: ").append(valor).append("\n");
  debugInfo.append("MouseY: ").append(mouseY).append("\n");
  debugInfo.append("MouseX: ").append(mouseX).append("\n\n");

  // NOVO: Status da conexão de rede
  debugInfo.append("=== COMUNICAÇÃO ===\n");
  if (network != null) {
    debugInfo.append("Status: ").append(network.isConnected() ? "✓ Conectado" : "✗ Desconectado").append("\n");
    debugInfo.append("Python: ").append(network.isConnected() ? "Recebendo dados" : "Sem comunicação").append("\n\n");
  }

  // Informações de captura (mantido igual)
  if (captureManager != null) {
    CaptureInfo captureInfo = captureManager.getCaptureInfo();

    debugInfo.append("=== CAPTURA DE VÍDEO ===\n");

    if (captureInfo.recording) {
      debugInfo.append("Status: ● A GRAVAR\n");
    } else {
      debugInfo.append("Status: ○ Parado\n");
    }

    float tempoSegundos = captureInfo.currentMillis / 1000.0;
    float tempoMaxSegundos = captureInfo.maxMillis / 1000.0;
    debugInfo.append("Tempo: ").append(nf(tempoSegundos, 1, 1))
      .append(" / ").append(nf(tempoMaxSegundos, 1, 1)).append(" s\n");

    debugInfo.append("Frames: ").append(captureInfo.capturedFrames)
      .append(" / ").append(captureInfo.targetFrames).append("\n");

    debugInfo.append("FPS: ").append(nf(captureInfo.frameRate, 1, 2)).append("\n");
    debugInfo.append("Intervalo: ").append(nf(captureInfo.frameInterval, 1, 1)).append(" ms\n");
    debugInfo.append("CSV linhas: ").append(captureInfo.csvLines).append("\n");
    debugInfo.append("Pasta: ").append(captureInfo.dateFormatted).append("\n\n");
  }

  // Estatísticas das animações (mantido igual)
  debugInfo.append("=== ESTATÍSTICAS ===\n");
  debugInfo.append("Total frames: ").append(totalFrames).append("\n");

  for (int i = 0; i < maxAnimation; i++) {
    float percentage = totalFrames > 0 ? (animationOccurrences[i] * 100.0f / totalFrames) : 0;
    String marker = (i == currentAnimation) ? " ◄" : "";
    debugInfo.append("Anim ").append(i).append(": ")
      .append(animationOccurrences[i]).append(" (")
      .append(nf(percentage, 1, 1)).append("%)").append(marker).append("\n");
  }

  debugInfo.append("\nFPS Real: ").append(nf(frameRate, 0, 1));
  debugInfo.append("\nFrame: ").append(frameCount);
  debugInfo.append("\n\nControles:\n[N] - Teste de comunicação\n[1-4] - Mudar animação");

  debug.setDebugText(debugInfo.toString());
}

void updateDebugWindow_Jogo(NetworkData data) {
  StringBuilder debugInfo = new StringBuilder();

  // Informações da animação atual
  debugInfo.append("=== MODO JOGO ===\n");
  debugInfo.append("Animação: ").append(animationNames[currentAnimation]).append("\n");
  debugInfo.append("Anim ID: ").append(currentAnimation);
  if (isTransitioning) {
    debugInfo.append(" → ").append(targetAnimation);
    debugInfo.append(" (").append(nf(animationTransition * 100, 1, 0)).append("%)");
  }
  debugInfo.append("\n\n");

  // Status da conexão - MODIFICADO para mostrar comunicação bidirecional
  debugInfo.append("=== COMUNICAÇÃO BIDIRECIONAL ===\n");
  if (network != null) {
    debugInfo.append("Status: ").append(network.isConnected() ? "✓ Conectado" : "✗ Desconectado").append("\n");
    debugInfo.append("Enviando: ").append(network.isConnected() ? "Dados de animação" : "Nada").append("\n");
    debugInfo.append("Recebendo: ").append(network.isConnected() ? "Dados das mãos" : "Nada").append("\n\n");
  }

  // Status das mãos (mantido igual)
  debugInfo.append("=== CONTROLES ===\n");
  if (data.validMaoDireita) {
    debugInfo.append("Mão Direita: ").append(nf(data.distanciaMaoDireita, 1, 3)).append(" ✓\n");
  } else {
    debugInfo.append("Mão Direita: Inativa ✗\n");
  }

  if (data.validAnguloEsquerda) {
    debugInfo.append("Mão Esquerda: ").append(nf(data.anguloMaoEsquerda, 1, 1)).append("° ✓\n");
    debugInfo.append("Gesto ativo: ").append(gestureActive ? "Sim" : "Não").append("\n");
  } else {
    debugInfo.append("Mão Esquerda: Inativa ✗\n");
  }
  debugInfo.append("\n");

  // Informações de captura (mantido igual)
  if (captureManager != null && captureManager.isRecording()) {
    CaptureInfo captureInfo = captureManager.getCaptureInfo();

    debugInfo.append("=== CAPTURA ===\n");
    debugInfo.append("A gravar \n");
    debugInfo.append("Frames: ").append(captureInfo.capturedFrames)
      .append(" / ").append(captureInfo.targetFrames).append("\n\n");
  }

  // Estatísticas das animações (mantido igual)
  debugInfo.append("=== ESTATÍSTICAS ===\n");
  debugInfo.append("Total frames: ").append(totalFrames).append("\n");

  for (int i = 0; i < maxAnimation; i++) {
    float percentage = totalFrames > 0 ? (animationOccurrences[i] * 100.0f / totalFrames) : 0;
    String marker = (i == currentAnimation) ? " ◄" : "";
    debugInfo.append("Anim ").append(i).append(": ")
      .append(animationOccurrences[i]).append(" (")
      .append(nf(percentage, 1, 1)).append("%)").append(marker).append("\n");
  }

  debugInfo.append("\nFPS: ").append(nf(frameRate, 0, 1));
  debugInfo.append("\nFrame: ").append(frameCount);
  debugInfo.append("\n\nControles:\n[N] - Teste de comunicação");

  debug.setDebugText(debugInfo.toString());
}

void iniciarScriptPython() {
  try {
    // Caminho absoluto ou relativo ao ficheiro .py
    String scriptPath = sketchPath("0input_pred.py"); // sketchPath("mediapipe_holistic.py");
    String pythonPath ="/Users/leonor/miniconda3/bin/python3";
    ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath);
    pb.redirectErrorStream(true); // junta stdout e stderr
    processoPython = pb.start();

    // Opcional: Ler o output do processo
    new Thread(() -> {
      try {
        BufferedReader reader = new BufferedReader(new InputStreamReader(processoPython.getInputStream()));
        String linha;
        while ((linha = reader.readLine()) != null) {
          println("[Python] " + linha);
        }
      }
      catch (IOException e) {
        e.printStackTrace();
      }
    }
    ).start();
  }
  catch (IOException e) {
    e.printStackTrace();
  }
}

void terminarScriptPython() {
  if (processoPython != null) {
    processoPython.destroy();
    processoPython = null;
    println("Script Python terminado.");
  }
}

void exit() {
  if (network != null && network.isConnected()) {
    println("Fechando conexão de rede...");
    network.closeConnection();
    delay(100); // Pequeno delay para garantir que a mensagem seja enviada
  }

  terminarScriptPython(); // fecha o processo
  super.exit(); // encerra o sketch
  exit();
}

// ==================== FIM DO SISTEMA DE CONTAGEM ====================

void sendAnimationChangeToNetwork(int newAnimation) {
  if (network != null && network.isConnected()) {
    // Criar dados customizados para mudança manual de animação
    JSONObject customData = new JSONObject();
    customData.setInt("newAnimation", newAnimation);
    customData.setInt("previousAnimation", currentAnimation);
    customData.setString("changeMethod", "keyboard");

    network.sendCustomData("manual_animation_change", customData);
    println("Enviado mudança de animação para Python: " + currentAnimation + " -> " + newAnimation);
  }
}

void keyPressed() {
  // transiçao entre animações com as teclas LEFT e RIGHT
  if (modeManager.isModoCaptura()) {
    int previousAnimation = currentAnimation;

    if (key == '1') {
      currentAnimation = 0;
    } else if (key == '2') {
      currentAnimation = 1;
    } else if (key == '3') {
      currentAnimation = 2;
    } else if (key == '4') {
      currentAnimation = 3;
    }
    //println(currentAnimation);

    if (currentAnimation != previousAnimation) {
      sendAnimationChangeToNetwork(currentAnimation);
    }
  }

  // ==================== CONTROLOS DO SISTEMA DE CONTAGEM ====================

  // 's' - Salvar estatísticas
  if (key == 's' || key == 'S') {
    String filename = "animation_stats_" + year() + month() + day() + "_" + hour() + minute() + second() + ".txt";
    saveAnimationStats(filename);
  }

  // 'p' - Imprimir estatísticas na consola
  if (key == 'p' || key == 'P') {
    printAnimationStats();
  }

  // 'c' - Reset dos contadores
  if (key == 'i' || key == 'I') {
    resetAnimationCounters();
  }

  // 't' - Toggle da contagem
  if (key == 't' || key == 'T') {
    toggleCounting();
  }

  if (key == 'a' || key == 'A') addCircle();
  if (key == 'r' || key == 'R') removeCircle();

  if (key == 'q' || key == ESC) {
    // Salvar estatísticas finais antes de sair
    printAnimationStats();
    String finalFilename = "final_animation_stats.txt";
    saveAnimationStats(finalFilename);

    key = 0;
    if (network != null && network.isConnected()) {
      network.closeConnection();
    }

    if (captureManager != null) {
      captureManager.finalizar();
    }
    exit();
  }

  if (key == 'c' || key == 'C') {
    saveFrame("../data/frame/captura-########.png");  // guarda como "captura-0001.png", etc.
    println("Frame capturada!");
  }

  if (key == 'n' || key == 'N') {
    if (network != null && network.isConnected()) {
      // Enviar dados de teste
      JSONObject testData = new JSONObject();
      testData.setString("message", "Teste de comunicação do Processing");
      testData.setInt("testNumber", frameCount);
      testData.setFloat("randomValue", random(0, 100));

      network.sendCustomData("test_message", testData);
      println("Enviado mensagem de teste para Python");
    } else {
      println("Rede não conectada - não é possível enviar teste");
    }
  }

  if (debug != null) {
    debug.addDebugLine("Tecla pressionada: " + key);
  }
}
