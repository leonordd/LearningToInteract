class ModeManager {
  private int modo;

  public void configurarModo() {
    String input = JOptionPane.showInputDialog("Introduz \n1 para o modo de captura \n2 para modo de jogo \n3 para o modo de jogo com o treino");

    try {
      modo = int(input);
    }
    catch (Exception e) {
      println("Input inválido, a usar o modo 1");
      modo = 1;
    }

    switch(modo) {
    case 1:
      println("=== MODO 1 ATIVO ===");
      break;
    case 2:
      println("=== MODO 2 ATIVO ===");
      break;
    case 3:
      println("=== MODO 3 ATIVO ===");
      break;
    default:
      println("Modo inválido, a usar o modo 1 por default");
      modo = 1;
    }
  }

  public boolean isModoCaptura() {
    return modo == 1;
  }

  public boolean isModoJogo() {
    return modo == 2;
  }

  public boolean isModoJogoComTreino () {
    return modo == 3;
  }

  public int getModo() {
    return modo;
  }
}
