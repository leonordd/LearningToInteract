// Importar a biblioteca Video do Processing
import processing.video.*;

// Classe para gerenciar os vídeos
class VideoManager {
  Movie videoFile;
  ArrayList<PImage> capturedFrames = new ArrayList<PImage>();
  ArrayList<Float> frameTimes = new ArrayList<Float>();
  int frameCount = 0;
  String filename;
  color highlightColor;
  int frameWidth = 800; //120;  // Largura padrão de cada frame na timeline
  int frameHeight = 1000; //80;  // Altura padrão de cada frame na timeline
  float lastJumpTime = -1;  // Último tempo para o qual o vídeo saltou
  boolean isStable = true;  // Indica se o vídeo está estabilizado após um jump
  
  VideoManager(PApplet parent, String videoPath, color highlightColor) {
    videoFile = new Movie(parent, videoPath);
    this.filename = videoPath;
    this.highlightColor = highlightColor;
  }
  
  // Método para definir tamanho dos frames na timeline
  void setFrameSize(int width, int height) {
    this.frameWidth = width;
    this.frameHeight = height;
  }
  
  void play() {
    videoFile.play();
  }
  
  void pause() {
    videoFile.pause();
  }
  
  void jump(float time) {
    // Só faz o salto se for para um tempo diferente do atual
    // ou se o último salto foi há mais de 10 frames
    if (time != lastJumpTime) {
      videoFile.jump(time);
      lastJumpTime = time;
      isStable = false;  // Marcar o vídeo como instável após um salto
      
      // Para videos mais pesados, pode ser necessário pausar e dar play
      // para forçar o carregamento do frame correto
      videoFile.pause();
      videoFile.play();
      videoFile.pause();
    }
  }
  
  boolean isPlaying() {
    return videoFile.isPlaying();
  }
  
  float time() {
    return videoFile.time();
  }
  
  float duration() {
    return videoFile.duration();
  }
  
  void read() {
    if (videoFile.available()) {
      videoFile.read();
      if (!isStable && abs(videoFile.time() - lastJumpTime) < 0.1) {
        isStable = true;  // O vídeo está próximo do tempo desejado
      }
    }
  }
  
  void reset() {
    videoFile.jump(0);
    capturedFrames.clear();
    frameTimes.clear();
    frameCount = 0;
    lastJumpTime = -1;
    isStable = true;
  }
  
  // Método para verificar se o vídeo está estabilizado
  boolean isStabilized() {
    return isStable;
  }
}
