import java.text.SimpleDateFormat;
import java.util.Date;
import java.io.File;

class FileManager {
  private String baseDataFolder;
  private String dailyFolder;
  private SimpleDateFormat dateFormat;
  private String dateFormatted;
  private int videoCount = 1;
  private int csvCount = 1;
  
  public void configurarPastas() {
    try {
      dateFormat = new SimpleDateFormat("yyyy_MM_dd");
      dateFormatted = dateFormat.format(new Date());
      
      baseDataFolder = sketchPath("../data");
      File baseFolder = new File(baseDataFolder);
      if (!baseFolder.exists()) {
        baseFolder.mkdir();
        println("Pasta base de dados criada: " + baseDataFolder);
      }
      
      dailyFolder = baseDataFolder + "/" + dateFormatted;
      File dayFolder = new File(dailyFolder);
      if (!dayFolder.exists()) {
        dayFolder.mkdir();
        println("Pasta do dia criada: " + dailyFolder);
      } else {
        println("Pasta do dia já existe: " + dailyFolder);
      }
    } catch (Exception e) {
      println("Erro ao configurar pastas: " + e.getMessage());
      e.printStackTrace();
    }
  }
  
  public String getDailyFolder() {
    return dailyFolder;
  }
  
  public String getDateFormatted() {
    return dateFormatted;
  }
  
  public int getNextVideoCount() {
    while (new File(dailyFolder + "/video" + videoCount + ".mp4").exists()) {
      videoCount++;
    }
    return videoCount;
  }
  
  public int getNextCsvCount() {
    while (new File(dailyFolder + "/dados_teclas" + csvCount + ".csv").exists()) {
      csvCount++;
    }
    return csvCount;
  }
}
