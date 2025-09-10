#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

void setup(void) {
  // Start serial communication at a baud rate of 115200.
  // This is used to display the sensor data.
  Serial.begin(115200);
  // while (!Serial) {
  //   delay(10); // Wait for the serial port to open
  // }
  Serial.println("starting");
  // Initialize the MPU-6050 sensor.
  // The begin() function checks for the sensor and its I2C address.
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip, check wiring!");
    while (1) {
      Serial.println("fck");
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");
  
  // Set the ranges for the accelerometer and gyroscope.
  // These are optional and can be changed based on your needs.
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
  // Create a structure to hold the sensor event data.
  sensors_event_t a, g, temp;

  // Read the latest sensor data from the MPU-6050.
  mpu.getEvent(&a, &g, &temp);

  // Print the accelerometer data in meters per second squared (m/s^2).
  Serial.print("Accel X: ");
  Serial.print(a.acceleration.x);
  Serial.print(" Y: ");
  Serial.print(a.acceleration.y);
  Serial.print(" Z: ");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");

  // Print the gyroscope data in radians per second (rad/s).
  Serial.print("Gyro X: ");
  Serial.print(g.gyro.x);
  Serial.print(" Y: ");
  Serial.print(g.gyro.y);
  Serial.print(" Z: ");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");
  
  // Add a small delay between readings.
  delay(500);
}
