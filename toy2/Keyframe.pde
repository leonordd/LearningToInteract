// Classe para representar keyframes com rotação
class Keyframe {
  float time;
  PVector pos;
  float w, h;
  float rotation; // Adicionar propriedade de rotação (em radianos)

  Keyframe(float time, PVector pos, float w, float h, float rotation) {
    this.time = time;
    this.pos = pos;
    this.w = w;
    this.h = h;
    this.rotation = rotation;
  }

  // Construtor sem rotação (para compatibilidade)
  Keyframe(float time, PVector pos, float w, float h) {
    this(time, pos, w, h, 0.0);
  }
}
