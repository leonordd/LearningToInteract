class DebugWindow extends PApplet {
  ArrayList<String> debugLines = new ArrayList<String>();
  
  public void settings() {
    size(800, 300); // Aumentar ainda mais para acomodar info de captura
  }
  
  public void setup() {
    surface.setLocation(width+100, height+400);
    surface.setTitle("Debug & Capture Info");
    background(0);
    fill(255);
    textSize(11); // Texto ligeiramente menor para caber mais info
  }
  
  public void draw() {
  background(0);
  fill(255);
  textSize(20);

  float x = 20;                // posição inicial da 1ª coluna
  float yStart = 40;          // posição Y inicial
  float y = yStart;           
  float lineHeight = 20;      // altura entre linhas
  float maxHeight = height - 40;  // altura máxima antes de quebrar para nova coluna

  for (int i = 0; i < debugLines.size(); i++) {
    String line = debugLines.get(i);

    // Mudar de coluna se ultrapassar a altura máxima
    if (y > maxHeight) {
      x += 300; // distância horizontal entre colunas
      y = yStart;
    }

    // Colorir diferentes seções
    if (line.startsWith("===")) {
      fill(100, 200, 255); // Azul para cabeçalhos
    } else if (line.contains("●")) {
      fill(255, 100, 100); // Vermelho para indicadores de gravação
    } else if (line.contains("✓")) {
      fill(100, 255, 100); // Verde para status positivo
    } else if (line.contains("✗")) {
      fill(255, 150, 150); // Vermelho claro para status negativo
    } else if (line.contains("◄")) {
      fill(255, 255, 100); // Amarelo para animação atual
    } else {
      fill(255); // Branco para texto normal
    }

    text(line, x, y);
    y += lineHeight;
  }
}

  
  public void setDebugText(String txt) {
    debugLines.clear();
    String[] lines = txt.split("\n");
    for (String line : lines) {
      debugLines.add(line);
    }
  }
  
  public void addDebugLine(String line) {
    debugLines.add(line);
  }
  
  public void clearDebug() {
    debugLines.clear();
  }
}
