import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';

class SensorData {
  final double flowRate;
  final double totalVolume;

  SensorData({
    required this.flowRate,
    required this.totalVolume,
  });

  factory SensorData.fromMap(Map<dynamic, dynamic> data) {
    return SensorData(
      flowRate: _parseDouble(data['flow_rate']),
      totalVolume: _parseDouble(data['total_volume']),
    );
  }

  static double _parseDouble(dynamic value) {
    try {
      if (value is double) return value;
      if (value is int) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0.0;
      return 0.0;
    } catch (e) {
      return 0.0;
    }
  }
}

class SensorService {
  late DatabaseReference _databaseRef;
  StreamSubscription<DatabaseEvent>? _subscription;
  final ValueNotifier<SensorData> currentReading = ValueNotifier(
    SensorData(flowRate: 0.0, totalVolume: 0.0),
  );
  final ValueNotifier<String?> errorState = ValueNotifier(null);

  void initialize() {
    _initializeFirebase();
    _startListening();
  }

  Future<void> _initializeFirebase() async {
    try {
      if (FirebaseAuth.instance.currentUser == null) {
        await FirebaseAuth.instance.signInAnonymously();
      }
    } catch (e) {
      errorState.value = 'Authentication error: $e';
      if (kDebugMode) print('Auth error: $e');
    }
  }

  void _startListening() {
    try {
      final database = FirebaseDatabase.instanceFor(
        app: FirebaseDatabase.instance.app,
        databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
      );

      _databaseRef = database.ref('sensor_readings');

      _subscription = _databaseRef.onValue.listen((event) {
        if (event.snapshot.value != null) {
          try {
            // Handle both single reading and list of readings
            if (event.snapshot.value is Map) {
              final data = event.snapshot.value as Map<dynamic, dynamic>;
              // Check if we have the expected fields
              if (data.containsKey('flow_rate') || data.containsKey('total_volume')) {
                currentReading.value = SensorData.fromMap(data);
                if (kDebugMode) {
                  print('Updated reading - Flow: ${currentReading.value.flowRate}, Volume: ${currentReading.value.totalVolume}');
                }
              } else {
                // Try to get the latest reading from a list of readings
                final lastReading = data.values.last as Map<dynamic, dynamic>?;
                if (lastReading != null) {
                  currentReading.value = SensorData.fromMap(lastReading);
                  if (kDebugMode) {
                    print('Updated from last reading - Flow: ${currentReading.value.flowRate}, Volume: ${currentReading.value.totalVolume}');
                  }
                }
              }
            }
          } catch (e) {
            errorState.value = 'Data parsing error: $e';
            if (kDebugMode) print('Data structure: ${event.snapshot.value}\nError: $e');
          }
        } else {
          if (kDebugMode) print('No data available in snapshot');
        }
      }, onError: (error) {
        errorState.value = 'Database error: ${error.toString()}';
        if (kDebugMode) print('DB error: $error');
      });
    } catch (e) {
      errorState.value = 'Listener setup error: ${e.toString()}';
      if (kDebugMode) print('Listener error: $e');
    }
  }

  void dispose() {
    _subscription?.cancel();
    currentReading.dispose();
    errorState.dispose();
  }
}