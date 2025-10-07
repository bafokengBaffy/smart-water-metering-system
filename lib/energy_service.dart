import 'dart:async';

class EnergyService {
  static int meterReading = 0; // Simulated energy consumption
  static int activeDevices = 5; // Simulated active devices
  static int targetUsage = 100; // Target energy consumption

  // Simulate updating energy consumption every second
  static void startEnergyUpdates(Function(int) onUpdate) {
    Timer.periodic(const Duration(seconds: 1), (timer) {
      meterReading += 1; // Increase the consumption over time

      if (meterReading % 10 == 0) {
        activeDevices =
            (activeDevices + 1) % 15 + 5; // Randomly update active devices
      }

      // Notify the listeners (usually the UI) about the updated data
      onUpdate(meterReading);
    });
  }

  // Method to get current energy consumption
  static int getEnergyConsumption() {
    return meterReading;
  }

  // Method to get the current active devices
  static int getActiveDevices() {
    return activeDevices;
  }

  // Method to get the target energy consumption
  static int getTargetUsage() {
    return targetUsage;
  }
}
