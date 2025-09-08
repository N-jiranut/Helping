#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "vivo_V25";
const char* password = "02_jiranut";

const char* serverName = "http://10.207.14.216:5000/right";

WiFiClient client;
HTTPClient http;

void connectToWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); 
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi!");
}

Adafruit_MPU6050 mpu;

const int FLEX_PIN_THUMB = 4;
const int FLEX_PIN_INDEX = 6;
const int FLEX_PIN_MIDDLE = 15;
const int FLEX_PIN_RING = 17;
const int FLEX_PIN_PINKY = 8;

int flexSensorValues[5];
const int FLEX_PINS[] = {FLEX_PIN_THUMB, FLEX_PIN_INDEX, FLEX_PIN_MIDDLE, FLEX_PIN_RING, FLEX_PIN_PINKY};

const int FLEX_MIN_STRAIGHT = 2000;
const int FLEX_MAX_BENT = 3500;

void setup() {
  Serial.begin(115200);
  delay(1000);

  connectToWiFi(); 
  Serial.println("Initializing I2C bus on GPIO 8 (SDA) and GPIO 9 (SCL)...");
  Wire.begin(1, 2);
  delay(100);
  Serial.println("Attempting to find MPU6050 chip...");
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip. Check wiring, power, and I2C address.");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found and initialized!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
    case MPU6050_RANGE_2_G: Serial.println("+-2G"); break;
    case MPU6050_RANGE_4_G: Serial.println("+-4G"); break;
    case MPU6050_RANGE_8_G: Serial.println("+-8G"); break;
    case MPU6050_RANGE_16_G: Serial.println("+-16G"); break;
  }

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
    case MPU6050_RANGE_250_DEG: Serial.println("+-250 deg/s"); break;
    case MPU6050_RANGE_500_DEG: Serial.println("+-500 deg/s"); break;
    case MPU6050_RANGE_1000_DEG: Serial.println("+-1000 deg/s"); break;
    case MPU6050_RANGE_2000_DEG: Serial.println("+-2000 deg/s"); break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
    case MPU6050_BAND_260_HZ: Serial.println("260 Hz"); break;
    case MPU6050_BAND_184_HZ: Serial.println("184 Hz"); break;
    case MPU6050_BAND_94_HZ: Serial.println("94 Hz"); break;
    case MPU6050_BAND_44_HZ: Serial.println("44 Hz"); break;
    case MPU6050_BAND_21_HZ: Serial.println("21 Hz"); break;
    case MPU6050_BAND_10_HZ: Serial.println("10 Hz"); break;
    case MPU6050_BAND_5_HZ: Serial.println("5 Hz"); break;
  }

  Serial.println("Flex Sensors initialized. Remember to calibrate FLEX_MIN_STRAIGHT and FLEX_MAX_BENT values.");
  Serial.println("------------------------------------");
}

unsigned long start = 0;
unsigned long del = 0;

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  for (int i = 0; i < 5; i++) {
    flexSensorValues[i] = analogRead(FLEX_PINS[i]);
    int bendPercentage = map(flexSensorValues[i], FLEX_MIN_STRAIGHT, FLEX_MAX_BENT, 0, 100);
    bendPercentage = constrain(bendPercentage, 0, 100);
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");
    String json = "{";
    json += "\"accel_x\":" + String(a.acceleration.x, 2) + ",";
    json += "\"accel_y\":" + String(a.acceleration.y, 2) + ",";
    json += "\"accel_z\":" + String(a.acceleration.z, 2) + ",";
    json += "\"gyro_x\":" + String(g.gyro.x, 2) + ",";
    json += "\"gyro_y\":" + String(g.gyro.y, 2) + ",";
    json += "\"gyro_z\":" + String(g.gyro.z, 2) + ",";
    json += "\"temperature\":" + String(temp.temperature, 2) + ",";
    json += "\"flex_raw_1\":" + String(flexSensorValues[0]) + ",";
    json += "\"flex_raw_2\":" + String(flexSensorValues[1]) + ",";
    json += "\"flex_raw_3\":" + String(flexSensorValues[2]) + ",";
    json += "\"flex_raw_4\":" + String(flexSensorValues[3]) + ",";
    json += "\"flex_raw_5\":" + String(flexSensorValues[4]);
    json += "}";
    int httpResponseCode = http.POST(json);
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Data sent to Google Sheets: " + response);
    } else {
      Serial.println("Error sending data. HTTP code: " + String(httpResponseCode));
    }
    http.end();
  } else {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectToWiFi();
  }
  delay(250);
}