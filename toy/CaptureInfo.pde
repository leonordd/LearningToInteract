class CaptureInfo {
  public int currentMillis;
  public int maxMillis;
  public int capturedFrames;
  public int targetFrames;
  public int csvLines;
  public boolean recording;
  public float frameRate;
  public float frameInterval;
  public String dateFormatted;

  public CaptureInfo(int currentMillis, int maxMillis, int capturedFrames, 
                     int targetFrames, int csvLines, boolean recording,
                     float frameRate, float frameInterval, String dateFormatted) {
    this.currentMillis = currentMillis;
    this.maxMillis = maxMillis;
    this.capturedFrames = capturedFrames;
    this.targetFrames = targetFrames;
    this.csvLines = csvLines;
    this.recording = recording;
    this.frameRate = frameRate;
    this.frameInterval = frameInterval;
    this.dateFormatted = dateFormatted;
  }
}
