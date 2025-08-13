// Classe auxiliar para dados thread-safe
class NetworkData {
  public final float distanciaMaoDireita;
  public final float distanciaMaoEsquerda;
  public final float anguloMaoEsquerda;
  public final float anguloVerticalMaoEsquerda;
  public final boolean validMaoDireita;
  public final boolean validMaoEsquerda;
  public final boolean validAnguloEsquerda;
  
  NetworkData(float distDir, float distEsq, float angEsq, float angVertEsq,
              boolean validDir, boolean validEsq, boolean validAng) {
    this.distanciaMaoDireita = distDir;
    this.distanciaMaoEsquerda = distEsq;
    this.anguloMaoEsquerda = angEsq;
    this.anguloVerticalMaoEsquerda = angVertEsq;
    this.validMaoDireita = validDir;
    this.validMaoEsquerda = validEsq;
    this.validAnguloEsquerda = validAng;
  }
}
