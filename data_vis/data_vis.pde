// Objetos para os vídeos
import processing.video.*;
VideoManager video1;
VideoManager video2;

// Variáveis originais mantidas
Table table;
float tecla, type, id, X, Y, Z, localMillis, value;
float maxMillis = 0;
float maxTecla = 0;
float maxType = 0;
float maxValue = 0;

// Variáveis para controlar o hover
boolean hoverOverPoint = false;
float hoverMillis = 0;
float lastHoverMillis = -1;

// Variáveis para carregamento progressivo
boolean csvLoaded = false;
boolean csvLoading = false;
int currentCSVRow = 0;
int batchSize = 50; // Carrega 50 linhas de cada vez
String[] csvLines;
int totalCSVRows = 0;

// Lista para armazenar dados carregados
ArrayList<DataPoint> dataPoints;
ArrayList<SkeletonData> skeletonData;

// Caminhos dos arquivos
String PATH_BALL = "dataset47/video1.mp4";
String PATH_FACE = "dataset47/v1.mp4";
String PATH_CSV = "data/dataset47/combinado.csv";

float video1duration, video2duration;
float dif;
float timelineDuration;

Table skeletonTable;
ArrayList<String> poseColumns;

// CONEXÕES CORRETAS DO MEDIAPIPE POSE (33 pontos)
// Baseadas na documentação oficial do MediaPipe
int[][] poseConnections = {
  // Face connections
  {0, 1}, {1, 2}, {2, 3}, {3, 7}, {0, 4}, {4, 5}, {5, 6}, {6, 8}, {9, 10},

  // Shoulder to shoulder
  {11, 12},

  // Right arm
  {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},

  // Left arm
  {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},

  // Torso
  {11, 23}, {12, 24}, {23, 24},

  // Right leg
  {23, 25}, {25, 27}, {27, 29}, {29, 31}, {27, 31},

  // Left leg
  {24, 26}, {26, 28}, {28, 30}, {30, 32}, {28, 32}
};

// Conexões das mãos (21 pontos cada - MediaPipe Hand)
int[][] handConnections = {
  // Polegar
  {1, 2}, {2, 3}, {3, 4},

  // Indicador
  {5, 6}, {6, 7}, {7, 8},

  // Médio
  {9, 10}, {10, 11}, {11, 12},

  // Anelar
  {13, 14}, {14, 15}, {15, 16},

  // Mindinho
  {17, 18}, {18, 19}, {19, 20},

  // Conexões da palma
  {0, 1}, {0, 5}, {0, 9}, {0, 13}, {0, 17},
  {5, 9}, {9, 13}, {13, 17}, {1, 5}
};

float millis_mil;

// Variáveis para atualizar instantaneamente os tempos dos vídeos após jump()
float lastJumpTime1 = -1;
float lastJumpTime2 = -1;
boolean jumped1 = false;
boolean jumped2 = false;

void setup() {
  size(1600, 950);
  //fullScreen();
  background(255);
  surface.setTitle("Modelo Visualização de Dados");

  // Inicializar listas
  dataPoints = new ArrayList<DataPoint>();
  skeletonData = new ArrayList<SkeletonData>();

  // Carregar os vídeos
  video1 = new VideoManager(this, PATH_BALL, color(0, 100, 255));
  video2 = new VideoManager(this, PATH_FACE, color(255, 100, 0));

  // Definir tamanhos dos frames na timeline
  video1.setFrameSize(800/6, 1000/6);
  video2.setFrameSize(640/2, 480/2);

  video1.read();
  video2.read();

  video1.play();
  video2.play();

  video1duration = video1.duration()*1000;
  video2duration = video2.duration()*1000;

  // Iniciar carregamento do CSV
  if (fileExists(PATH_CSV)) {
    startCSVLoading();
  }
}

void startCSVLoading() {
  // Carregar apenas o cabeçalho primeiro para obter informações básicas
  String[] lines = loadStrings(PATH_CSV);
  if (lines != null && lines.length > 0) {
    csvLines = lines;
    totalCSVRows = lines.length - 1; // -1 para o cabeçalho
    csvLoading = true;

    println("Iniciar carregamento de " + totalCSVRows + " linhas...");

    // Inicializar colunas do pose
    poseColumns = new ArrayList<String>();
    String header = lines[0];
    String[] columns = split(header, ',');

    for (String col : columns) {
      col = trim(col);
      if (col.startsWith("xp") || col.startsWith("yp") ||
        col.startsWith("xlh") || col.startsWith("ylh") ||
        col.startsWith("xrh") || col.startsWith("yrh") ||
        col.startsWith("xfm") || col.startsWith("yfm")) {
        poseColumns.add(col);
      }
    }
  }
}

// Função para carregar dados progressivamente no draw()
void loadCSVBatch() {
  if (!csvLoading || csvLoaded) return;

  String[] columns = split(csvLines[0], ',');
  int endRow = min(currentCSVRow + batchSize, totalCSVRows);

  for (int i = currentCSVRow; i < endRow; i++) {
    int lineIndex = i + 1; // +1 para pular o cabeçalho
    if (lineIndex < csvLines.length) {
      String[] values = split(csvLines[lineIndex], ',');

      try {
        // Encontrar índices das colunas importantes
        int millisIndex = findColumnIndex(columns, "TempoVideo");
        int valueIndex = findColumnIndex(columns, "AnimacaoAtual");

        if (millisIndex != -1 && valueIndex != -1) {
          float localMillis = float(trim(values[millisIndex]));
          float value = float(trim(values[valueIndex]));

          // Ajustar valores conforme o código original
          /*if (value == 5) value = 1.5;
          else if (value == 10) value = 2;*/

          // Adicionar ponto de dados
          dataPoints.add(new DataPoint(localMillis, value));

          // Atualizar máximos
          if (localMillis > maxMillis) maxMillis = localMillis;
          if (value > maxValue) maxValue = value;

          // Carregar dados do skeleton se disponíveis
          SkeletonData skelData = new SkeletonData(localMillis);
          for (String col : poseColumns) {
            int colIndex = findColumnIndex(columns, col);
            if (colIndex != -1 && colIndex < values.length) {
              try {
                float coord = float(trim(values[colIndex]));
                skelData.addCoordinate(col, coord);
              }
              catch (NumberFormatException e) {
                skelData.addCoordinate(col, 500f); // Valor padrão
              }
            }
          }
          skeletonData.add(skelData);
        }
      }
      catch (Exception e) {
        println("Erro ao processar linha " + lineIndex + ": " + e.getMessage());
      }
    }
  }

  currentCSVRow = endRow;

  // Verificar se terminou
  if (currentCSVRow >= totalCSVRows) {
    csvLoaded = true;
    csvLoading = false;

    // Calcular diferença e duração da timeline
    dif = video2duration - maxMillis;
    timelineDuration = maxMillis;

    println("Duração vídeo 1:" + video1duration);
    println("Duração vídeo 2:" + video2duration);
    println("CSV carregado completamente!");
    println("Total de pontos: " + dataPoints.size());
    println("Valor máximo de millis: " + maxMillis);
    println("Diferença entre vídeo 2 e maxMillis: " + dif + "ms");
    println("Duração da timeline: " + timelineDuration/1000 + "s");
    video2duration = video2duration + abs(dif);
    println("Duração vídeo 2:" + video2duration);
  } else {
    // Mostrar progresso
    float progress = (float)currentCSVRow / totalCSVRows * 100;
    if (frameCount % 30 == 0) { // Atualizar a cada 30 frames
      println("A carregar: " + nf(progress, 0, 1) + "% (" + currentCSVRow + "/" + totalCSVRows + ")");
    }
  }
}

// Função auxiliar para encontrar índice de coluna
int findColumnIndex(String[] columns, String columnName) {
  for (int i = 0; i < columns.length; i++) {
    if (trim(columns[i]).equals(columnName)) {
      return i;
    }
  }
  return -1;
}

// Função necessária para verificar se um arquivo existe
boolean fileExists(String path) {
  File file = new File(sketchPath(path));
  return file.exists();
}

// Função para calcular o tempo em segundos baseado na posição X do mouse
float calculateHoverTime(float mouseX) {
  float maxWidth = width-50;
  float minWidth = 50;

  // Mapear posição X do mouse para tempo em milissegundos
  float timeInMillis = map(mouseX, minWidth, maxWidth, 0, timelineDuration);

  // Converter para segundos e arredondar
  float timeInSeconds = timeInMillis / 1000.0;
  int roundedSeconds = round(timeInSeconds);

  // Converter de volta para milissegundos
  return roundedSeconds * 1000.0;
}

void draw() {
  // Carregar dados do CSV progressivamente
  if (csvLoading) {
    loadCSVBatch();
  }

  if (jumped1 && abs(video1.time() - lastJumpTime1) < 0.001) jumped1 = false;
  if (jumped2 && abs(video2.time() - lastJumpTime2) < 0.001) jumped2 = false;

  background(255);

  float posY_load = height/2;
  // Mostrar status do carregamento de dados
  if (csvLoading) {
    fill(75, 0, 167);
    textSize(40);
    text("MODELO DE VISUALIZAÇÃO DE DADOS", width/2, posY_load - 200);
    fill(0);
    textSize(16);
    textAlign(CENTER, CENTER);
    float progress = (float)currentCSVRow / totalCSVRows * 100;

    text("A carregar o CSV: " + nf(progress, 0, 1) + "%", width/2, posY_load - 40);
    text("Nº de linhas carregadas: " + currentCSVRow + "/" + totalCSVRows, width/2, posY_load - 20);

    // Barra de progresso
    stroke(0, 0, 0, 80);
    strokeWeight(1);
    noFill();
    rect(width/2 - 200, posY_load + 10, 400, 10, 100, 100, 100, 100);

    fill(45, 16, 225);
    noStroke();
    rect(width/2 - 200, posY_load + 10, map(progress, 0, 100, 0, 400), 10, 100, 100, 100, 100);

    return; // Não continuar até o CSV estar carregado
  }

  // Desenhar eixos (apenas se CSV estiver carregado)
  if (!csvLoaded) return;

  stroke(150);
  strokeWeight(0.5);

  video1.read();
  video2.read();

  float maxWidth = width-50;
  float minWidth = 50;
  float maxHeightCSVLine = height/2 - 80;
  float minHeightCSVLine = height/2 + 80;

  // Linhas verticais eixo xx (tempo) - agora de segundo em segundo
  stroke(200);
  strokeWeight(0.5);
  for (float i = 0; i <= timelineDuration+1000; i += 7500) { //
    float xRef = map(i, 0, timelineDuration, minWidth, maxWidth);
    line(xRef, height/2-80, xRef, height/2 + 100);

    fill(100);
    textSize(8);
    textAlign(CENTER, TOP);
    text(int(i)/1000 + "s", xRef, height/2 + 100);
  }

  // Linhas horizontais no eixo Y
  if (maxValue > 0) {
    for (int j = 0; j <= maxValue+0.02; j += 1) {
      float yRef = map(j, 0, maxValue, minHeightCSVLine, maxHeightCSVLine);
      stroke(200);
      line(50, yRef, width - 50, yRef);

      fill(100);
      textSize(10);
      textAlign(RIGHT, CENTER);
      text(j, 45, yRef);
    }
  }

  // Reset hover
  boolean prevHoverState = hoverOverPoint;
  hoverOverPoint = false;

  // Calcular tempo de hover baseado na posição X do mouse (arredondado para segundos)
  float calculatedHoverTime = calculateHoverTime(mouseX);

  // Verificar se o mouse está dentro da área da timeline
  if (mouseX >= minWidth && mouseX <= maxWidth && mouseY >= height/2-80 && mouseY <= height/2+100) {
    hoverOverPoint = true;
    hoverMillis = calculatedHoverTime;
  }

  // Desenhar pontos dos dados (usar ArrayList em vez de Table)
  /*for (DataPoint dp : dataPoints) {
    float localMillis_map = map(dp.localMillis, 0, timelineDuration, minWidth, maxWidth);
    float value_map = map(dp.value, 0, maxValue, minHeightCSVLine, maxHeightCSVLine);

    // Destacar pontos próximos do tempo de hover
    if (hoverOverPoint && abs(dp.localMillis - hoverMillis) < 300) { // 500ms de tolerância
      strokeWeight(5);
      stroke(255, 0, 0);

      float posText = mouseX > maxWidth-100 ? localMillis_map - 150 : localMillis_map + 100;

      fill(0);
      textSize(12);
      textAlign(LEFT, BOTTOM);
      text("Selected time: " + nf(hoverMillis/1000, 0, 0) + "s", posText, 120);
    } else {
      strokeWeight(3);
      stroke(255, 0, 255);
    }

    point(localMillis_map, value_map);
  }*/
  drawDataPointsWithLines(minWidth, maxWidth, minHeightCSVLine, maxHeightCSVLine);

  /*float videoPosY = height/2 - 225 - video1.frameHeight/2;
   float video2PosY = height/2 + 250 - video2.frameHeight/2;
   float video2PosX = mouseX - video2.frameWidth;
   float skelPosY = height/2 + 250 - video2.frameHeight/2;
   float skelPosX = mouseX - video2.frameHeight/2;*/


  float videoPosY = height/2 - 225 - video1.frameHeight/2;
  float video2PosY = height/2 + 250 - video2.frameHeight/2;
  float video2PosX = mouseX - video2.frameWidth;

  float video1Width =  video1.frameWidth;
  float video1Height = video1.frameHeight;
  float video1PosX = mouseX - video1Width/2;
  float video1PosY = height/2  - 250 - video1Height;

  float skelWidth = video2.frameWidth;
  float skelHeight = video2.frameHeight;
  float skelPosX = mouseX - video2.frameWidth/2;
  float skelPosY = height/2 + 280 - video2.frameHeight/2;

  // Se o rato está sobre a timeline, mostrar os frames correspondentes
  if (hoverOverPoint) {
    // Apenas atualizar os vídeos se o ponto de hover mudou
    if (lastHoverMillis != hoverMillis) {
      print("\nhoverMillis: " + hoverMillis);
      float timeInSeconds = hoverMillis / 1000.0;
      video1.jump(timeInSeconds);
      jumped1 = true;
      video2.jump(timeInSeconds);
      jumped2 = true;

      lastHoverMillis = hoverMillis;
    }

    // Desenhar linha vertical no ponto de hover
    float hoverX = map(hoverMillis, 0, timelineDuration, 50, width-50);
    stroke(0, 255, 0); // Linha verde
    strokeWeight(2);
    line(hoverX, 50, hoverX, height-50);

    // Adicionar indicador de segundo atual
    fill(0, 255, 0);
    textSize(14);
    textAlign(CENTER, BOTTOM);
    text(int(hoverMillis/1000) + "s", hoverX, 45);

    if (mouseX > maxWidth-video2.frameWidth) { //fim
      video1PosX = mouseX - video1Width;
      skelPosX = mouseX - video2.frameWidth;
    } else if (mouseX < minWidth + video2.frameWidth) { //inicio
      video1PosX = mouseX;
      skelPosX = mouseX;
    } else { //meio
      video1PosX = mouseX - video1Width/2;
      skelPosX = mouseX - video2.frameWidth/2;
    }

    //image(video2.videoFile, video2PosX, video2PosY, video2.frameWidth, video2.frameHeight);
    //drawFrameInfo(video2PosX, video2PosY, video2.frameWidth, video2.frameHeight,
    //"Video 2: " + nf(hoverMillis/1000.0, 0, 0) + "s", video2.highlightColor);
    image(video1.videoFile, video1PosX, video1PosY, video1Width, video1Height);
    drawFrameInfo(video1PosX, video1PosY, video1Width, video1Height, "Video 1: " + nf(hoverMillis/1000.0, 0, 0) + "s", video1.highlightColor);
    drawOptimizedSkeleton(hoverMillis, skelPosX, skelPosY, skelWidth, skelHeight);
  }

  // Mostrar informações sobre os vídeos
  fill(0);
  textSize(14);
  textAlign(LEFT, TOP);

  // Vídeo 1 (azul)
  fill(video1.highlightColor);
  text("Video 1: " + video1.filename, 60, 20);
  text("Time: " + nf(video1.time(), 0, 2) + "s / " + nf(video1.duration(), 0, 2) + "s", 60, 40);

  // Vídeo 2 (laranja)
  fill(video2.highlightColor);
  text("Video 2: " + video2.filename, 300, 20);
  text("Time: " + nf(video2.time(), 0, 2) + "s / " + nf(video2.duration(), 0, 2) + "s", 300, 40);

  // Adicionar instrução para o utilizador
  fill(0);
  textSize(12);
  textAlign(CENTER, BOTTOM);
  text("Passar o rato sobre a timeline para navegar de segundo em segundo", width/2, height - 10);

  drawLegend(width - 400, height - 50);
}

// Função para desenhar informações sobre o frame
void drawFrameInfo(float x, float y, float w, float h, String info, color c) {
  noFill();
  stroke(c);
  strokeWeight(1.5);
  rect(x, y, w, h);

  fill(c);
  textSize(12);
  textAlign(CENTER, BOTTOM);
  text(info, x + w/2, y + h + 20);
}

void drawOptimizedSkeleton(float time, float x, float y, float w, float h) {
  if (skeletonData.size() == 0) return;

  SkeletonData targetData = null;
  float minDiff = Float.MAX_VALUE;

  // Busca linear otimizada para encontrar o ponto mais próximo
  for (SkeletonData skelData : skeletonData) {
    float diff = abs(skelData.time - time);
    if (diff < minDiff) {
      minDiff = diff;
      targetData = skelData;
    }
  }

  if (targetData == null) return;

  // Desenhar frame de referência
  drawReferenceFrame(x, y, w, h);

  // POSE - Desenhar conexões
  stroke(0, 255, 0);
  strokeWeight(2);
  for (int[] conn : poseConnections) {
    try {
      int idx1 = conn[0];
      int idx2 = conn[1];

      String col1X = "xp" + (idx1 == 0 ? 1 : idx1);
      String col1Y = "yp" + (idx1 == 0 ? 1 : idx1);
      String col2X = "xp" + (idx2 == 0 ? 1 : idx2);
      String col2Y = "yp" + (idx2 == 0 ? 1 : idx2);

      float px1 = targetData.getCoordinate(col1X);
      float py1 = targetData.getCoordinate(col1Y);
      float px2 = targetData.getCoordinate(col2X);
      float py2 = targetData.getCoordinate(col2Y);

      if (isValidPoint(px1, py1) && isValidPoint(px2, py2)) {
        float sx1 = map(px1, 0, 1, x, x + w);
        float sy1 = map(py1, 0, 1, y, y + h);
        float sx2 = map(px2, 0, 1, x, x + w);
        float sy2 = map(py2, 0, 1, y, y + h);

        boolean point1Visible = isPointVisible(px1, py1);
        boolean point2Visible = isPointVisible(px2, py2);

        if (point1Visible && point2Visible) {
          stroke(0, 255, 0);
        } else {
          stroke(255, 255, 0);
        }

        line(sx1, sy1, sx2, sy2);
      }
    }
    catch (Exception e) {
      // Ignorar erros
    }
  }

  // POSE - Desenhar pontos
  noStroke();
  for (int i = 1; i <= 33; i++) {
    try {
      float px = targetData.getCoordinate("xp" + i);
      float py = targetData.getCoordinate("yp" + i);

      if (isValidPoint(px, py)) {
        float sx = map(px, 0, 1, x, x + w);
        float sy = map(py, 0, 1, y, y + h);

        if (isPointVisible(px, py)) {
          fill(0, 255, 0);
        } else {
          fill(255, 255, 0);
        }

        circle(sx, sy, 6);

        fill(255);
        textSize(8);
        textAlign(CENTER, CENTER);
        text(str(i), sx, sy);
      }
    }
    catch (Exception e) {
      // Continuar se a coluna não existir
    }
  }

  // Desenhar mãos
  drawOptimizedHand(targetData, "xlh", "ylh", x, y, w, h, color(255, 100, 100));
  drawOptimizedHand(targetData, "xrh", "yrh", x, y, w, h, color(100, 100, 255));
}

void drawOptimizedHand(SkeletonData data, String prefixX, String prefixY, float x, float y, float w, float h, color handColor) {
  // Conexões da mão
  stroke(handColor);
  strokeWeight(1.5);
  fill(255);

  for (int[] conn : handConnections) {
    try {
      int idx1 = conn[0] + 1;
      int idx2 = conn[1] + 1;

      float hx1 = data.getCoordinate(prefixX + idx1);
      float hy1 = data.getCoordinate(prefixY + idx1);
      float hx2 = data.getCoordinate(prefixX + idx2);
      float hy2 = data.getCoordinate(prefixY + idx2);

      if (isValidPoint(hx1, hy1) && isValidPoint(hx2, hy2)) {
        float sx1 = map(hx1, 0, 1, x, x + w);
        float sy1 = map(hy1, 0, 1, y, y + h);
        float sx2 = map(hx2, 0, 1, x, x + w);
        float sy2 = map(hy2, 0, 1, y, y + h);

        boolean point1Visible = isPointVisible(hx1, hy1);
        boolean point2Visible = isPointVisible(hx2, hy2);

        if (point1Visible && point2Visible) {
          stroke(handColor);
        } else {
          stroke(red(handColor), green(handColor), blue(handColor), 150);
        }

        line(sx1, sy1, sx2, sy2);
      }
    }
    catch (Exception e) {
      // Ignorar erros
    }
  }

  // Pontos da mão
  noStroke();
  for (int i = 1; i <= 21; i++) {
    try {
      float hx = data.getCoordinate(prefixX + i);
      float hy = data.getCoordinate(prefixY + i);

      if (isValidPoint(hx, hy)) {
        float sx = map(hx, 0, 1, x, x + w);
        float sy = map(hy, 0, 1, y, y + h);

        if (isPointVisible(hx, hy)) {
          fill(handColor);
        } else {
          fill(red(handColor), green(handColor), blue(handColor), 150);
        }

        circle(sx, sy, 4);
      }
    }
    catch (Exception e) {
      // Ignorar erros
    }
  }
}

// Função para desenhar frame de referência
void drawReferenceFrame(float x, float y, float w, float h) {
  // Frame externo (limites da imagem)
  stroke(100);
  strokeWeight(2);
  fill(255);
  rect(x, y, w, h);

  // Área "segura" onde os pontos são mais confiáveis (central 80%)
  stroke(200, 100, 100);
  strokeWeight(1);
  float safeMargin = 0.1; // 10% de margem
  rect(x + w * safeMargin, y + h * safeMargin,
    w * (1 - 2 * safeMargin), h * (1 - 2 * safeMargin));

  // Linhas de referência (terços)
  stroke(150);
  strokeWeight(0.5);
  // Verticais
  line(x + w/3, y, x + w/3, y + h);
  line(x + 2*w/3, y, x + 2*w/3, y + h);
  // Horizontais
  line(x, y + h/3, x + w, y + h/3);
  line(x, y + 2*h/3, x + w, y + 2*h/3);

  // Labels
  fill(100);
  textSize(10);
  textAlign(LEFT, TOP);
  text("Frame bounds", x + 5, y + 5);
  text("Safe area", x + w * safeMargin + 5, y + h * safeMargin + 5);
}

// Função para verificar se um ponto é válido (não é placeholder)
boolean isValidPoint(float x, float y) {
  return x != 500 && y != 500 && x >= 0 && x <= 1 && y >= 0 && y <= 1;
}

// Função para verificar se um ponto está na área "visível" (confiável)
boolean isPointVisible(float x, float y) {
  // Pontos na área central são considerados mais confiáveis
  // Margem de 10% em cada lado
  float margin = 0.1;
  return x > margin && x < (1 - margin) && y > margin && y < (1 - margin);
}

// Função para desenhar legenda das cores
void drawLegend(float x, float y) {
  textAlign(LEFT, TOP);
  textSize(12);
  float xx = 220;

  noStroke();
  // Pontos visíveis
  fill(0, 255, 0);
  circle(x, y, 10);
  fill(0);
  text("Pontos/conexões visíveis (confiáveis)", x + 15, y - 5);

  // Pontos estimados
  fill(255, 255, 0);
  circle(x, y + 20, 10);
  fill(0);
  text("Pontos/conexões estimados", x + 15, y + 15);

  // Frame bounds
  stroke(100);
  strokeWeight(2);
  noFill();
  rect(x + xx, y - 5, 16, 11);
  fill(0);
  text("Limites da imagem", x + 25 + xx, y - 5);

  // Safe area
  stroke(200, 100, 100);
  strokeWeight(1);
  noFill();
  rect(x + 2 + xx, y + 15, 16, 11);
  fill(0);
  text("Área segura (80% central)", x + 25 + xx, y + 15);
}

// Opção 1: Linhas simples conectando pontos consecutivos
void drawConnectedPoints(float minWidth, float maxWidth, float minHeightCSVLine, float maxHeightCSVLine) {
  if (dataPoints.size() < 2) return;
  
  stroke(100, 100, 255); // Cor azul para as linhas
  strokeWeight(0.1);
  
  for (int i = 0; i < dataPoints.size() - 1; i++) {
    DataPoint current = dataPoints.get(i);
    DataPoint next = dataPoints.get(i + 1);
    
    float x1 = map(current.localMillis, 0, timelineDuration, minWidth, maxWidth);
    float y1 = map(current.value, 0, maxValue, minHeightCSVLine, maxHeightCSVLine);
    
    float x2 = map(next.localMillis, 0, timelineDuration, minWidth, maxWidth);
    float y2 = map(next.value, 0, maxValue, minHeightCSVLine, maxHeightCSVLine);
    
    line(x1, y1, x2, y2);
  }
}

// Para adicionar ao seu método draw(), no local onde desenha os pontos:
void drawDataPointsWithLines(float minWidth, float maxWidth, float minHeightCSVLine, float maxHeightCSVLine) {
  // Primeiro desenhar as linhas (por baixo dos pontos)
  drawConnectedPoints(minWidth, maxWidth, minHeightCSVLine, maxHeightCSVLine); 
  // ou: drawColoredConnectedPoints(minWidth, maxWidth, minHeightCSVLine, maxHeightCSVLine);
  // ou: drawSmoothLine(minWidth, maxWidth, minHeightCSVLine, maxHeightCSVLine);
  
  // Depois desenhar os pontos (por cima das linhas)
  for (DataPoint dp : dataPoints) {
    float localMillis_map = map(dp.localMillis, 0, timelineDuration, minWidth, maxWidth);
    float value_map = map(dp.value, 0, maxValue, minHeightCSVLine, maxHeightCSVLine);

    // Destacar pontos próximos do tempo de hover
    if (hoverOverPoint && abs(dp.localMillis - hoverMillis) < 300) {
      strokeWeight(5);
      stroke(255, 0, 0);

      //float posText = mouseX > maxWidth-100 ? localMillis_map - 150 : localMillis_map + 100;

      //fill(0);
      //textSize(12);
      //textAlign(LEFT, BOTTOM);
      //text("Selected time: " + nf(hoverMillis/1000, 0, 0) + "s", posText, 120);
    } else {
      strokeWeight(3);
      stroke(255, 0, 255);
    }

    point(localMillis_map, value_map);
  }
}

void keyPressed(){
  if (key == 'g' || key == 'G') {
    saveFrame("data/frame-######.png");
    println("Frame guardada!");
  } else if (key == 'q' || key == 'Q') {
    println("Programa fechado!");
    exit(); 
  }
}
