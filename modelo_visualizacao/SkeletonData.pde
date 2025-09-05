class SkeletonData {
  float time;
  HashMap<String, Float> coordinates;
  
  SkeletonData(float t) {
    time = t;
    coordinates = new HashMap<String, Float>();
  }
  
  void addCoordinate(String key, float value) {
    coordinates.put(key, value);
  }
  
  float getCoordinate(String key) {
    Float val = coordinates.get(key);
    return val != null ? val : 500; // Valor padrão para dados inválidos
  }
}
