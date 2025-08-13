// Classe Circle otimizada
class Circle {
  float x, y, w, h;
  float rotation;
  color centerColor, edgeColor;

  // Cache para otimização
  private PGraphics gradientCache;
  private float lastW = -1, lastH = -1;
  private color lastCenterColor, lastEdgeColor;
  private boolean cacheValid = false;

  // Reduzir steps do gradiente para melhor performance
  private static final int GRADIENT_STEPS = 20; // Reduzido de 50 para 20

  Circle(float x, float y, float w, float h, color centerColor, color edgeColor) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.rotation = 0;
    this.centerColor = centerColor;
    this.edgeColor = edgeColor;
  }

  void updateFromKeyframe(Keyframe kf) {
    this.x = kf.pos.x;
    this.y = kf.pos.y;
    this.w = kf.w;
    this.h = kf.h;
    this.rotation = kf.rotation;
  }


  void display() {
    pushMatrix();

    translate(x, y);
    rotate(rotation);

    // Usar cache do gradiente se possível
    if (shouldUpdateCache()) {
      updateGradientCache();
    }

    if (gradientCache != null) {
      imageMode(CENTER);
      image(gradientCache, 0, 0);
    } else {
      // Fallback para gradiente simples se cache falhar
      drawSimpleGradient(0, 0, w, h, centerColor, edgeColor);
    }

    popMatrix();
  }

  private boolean shouldUpdateCache() {
    return !cacheValid ||
      abs(w - lastW) > 1 ||
      abs(h - lastH) > 1 ||
      centerColor != lastCenterColor ||
      edgeColor != lastEdgeColor;
  }

  private void updateGradientCache() {
    int cacheW = (int)(w * 2) + 10;
    int cacheH = (int)(h * 2) + 10;

    if (gradientCache == null ||
      gradientCache.width != cacheW ||
      gradientCache.height != cacheH) {
      gradientCache = createGraphics(cacheW, cacheH);
    }

    gradientCache.beginDraw();
    gradientCache.clear();
    gradientCache.noStroke();

    // Gradiente otimizado com menos steps
    for (int i = GRADIENT_STEPS; i > 0; i--) {
      float alpha = map(i, 0, GRADIENT_STEPS, 0, 1);
      float currentWidth = map(i, 0, GRADIENT_STEPS, 0, w);
      float currentHeight = map(i, 0, GRADIENT_STEPS, 0, h);

      color currentColor = lerpColor(edgeColor, centerColor, alpha);
      float transparency = map(i, 0, GRADIENT_STEPS, 50, 255);

      gradientCache.fill(red(currentColor), green(currentColor), blue(currentColor), transparency);
      gradientCache.ellipse(cacheW/2, cacheH/2, currentWidth * 2, currentHeight * 2);
    }

    gradientCache.endDraw();

    // Atualizar cache info
    lastW = w;
    lastH = h;
    lastCenterColor = centerColor;
    lastEdgeColor = edgeColor;
    cacheValid = true;
  }

  // Gradiente simples para fallback
  void drawSimpleGradient(float centerX, float centerY, float w, float h, color c1, color c2) {
    for (int i = 10; i > 0; i--) {
      float alpha = map(i, 0, 10, 0, 1);
      float currentWidth = map(i, 0, 10, 0, w);
      float currentHeight = map(i, 0, 10, 0, h);

      color currentColor = lerpColor(c2, c1, alpha);
      float transparency = map(i, 0, 10, 100, 255);

      fill(red(currentColor), green(currentColor), blue(currentColor), transparency);
      noStroke();
      ellipse(centerX, centerY, currentWidth * 2, currentHeight * 2);
    }
  }

  void changePos(float xx, float yy) {
    this.x = xx;
    this.y = yy;
  }

  void setRotation(float rot) {
    this.rotation = rot;
  }

  void setSize(float newW, float newH) {
    this.w = newW;
    this.h = newH;
  }
}
