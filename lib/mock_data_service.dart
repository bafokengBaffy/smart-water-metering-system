import 'dart:math';

class MockDataService {
  Stream<Map<String, dynamic>> getWaterQualityStream() async* {
    while (true) {
      await Future.delayed(const Duration(seconds: 2));
      yield {
        'pH': 6.8 + Random().nextDouble() * 1.5,
        'chlorine': 1.5 + Random().nextDouble() * 3.0,
        'turbidity': 0.5 + Random().nextDouble() * 4.5,
      };
    }
  }
}
