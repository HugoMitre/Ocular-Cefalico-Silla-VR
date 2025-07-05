// Código Arduino calibrado según las lecturas observadas
const int joystickX = A0;    // Pin analógico para el eje X
const int joystickY = A1;    // Pin analógico para el eje Y

// Calibración basada en las lecturas observadas
const int centerX = 516;     // Centro X observado
const int centerY = 503;     // Centro Y observado
const float deadzone = 5.0;  // Zona muerta reducida basada en las lecturas
const int responseDelay = 15; // Respuesta más rápida

// Variables para filtrado
float filteredX = centerX;
float filteredY = centerY;
const float filterFactor = 0.8;   // Aumentado para mayor suavidad

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Esperar por el puerto serial (Leonardo)
  }
}

void loop() {
  // Leer valores
  int rawX = analogRead(joystickX);
  int rawY = analogRead(joystickY);
  
  // Aplicar filtrado
  filteredX = (filterFactor * filteredX) + ((1 - filterFactor) * rawX);
  filteredY = (filterFactor * filteredY) + ((1 - filterFactor) * rawY);
  
  // Calcular desviación del centro calibrado
  float deltaX = filteredX - centerX;
  float deltaY = filteredY - centerY;
  
  // Calcular magnitud y ángulo
  float magnitude = sqrt(deltaX * deltaX + deltaY * deltaY);
  float angle = atan2(deltaY, deltaX) * 180.0 / PI;
  
  // Ajustar ángulo a rango 0-360
  if (angle < 0) angle += 360.0;
  
  // Normalizar magnitud con curva de respuesta suave
  float normalizedMagnitude;
  if (magnitude < deadzone) {
    normalizedMagnitude = 0;
  } else {
    // Mapear con curva exponencial suave
    normalizedMagnitude = map(magnitude, deadzone, 512, 0, 100);
    normalizedMagnitude = constrain(normalizedMagnitude, 0, 100);
    normalizedMagnitude = (normalizedMagnitude * normalizedMagnitude) / 100.0;
  }
  
  // Enviar solo los datos necesarios en el formato esperado por Unity
  Serial.print(angle, 1);          // Enviar el ángulo con un decimal
  Serial.print(",");                // Separador de coma
  Serial.println(normalizedMagnitude, 1);  // Enviar la magnitud normalizada con un decimal y terminar la línea

  delay(responseDelay);
}
