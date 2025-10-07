// ignore_for_file: deprecated_member_use, unused_field

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';

import 'home_screen.dart';
import 'payment_screen.dart';
import 'settings_screen.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with TickerProviderStateMixin {
  // Navigation
  int _selectedIndex = 1;

  // Data loading states
  bool _isLoading = true;
  bool _hasError = false;
  String _errorMessage = '';

  // Firebase references
  late DatabaseReference _databaseRef;
  late DatabaseReference _usageStatsRef;
  late StreamSubscription<DatabaseEvent> _databaseSubscription;

  // Real-time data
  double _currentFlowRate = 0.0;
  double _currentVolume = 0.0;
  double _currentPressure = 0.0; // Added pressure reading
  double _currentTemperature = 0.0; // Added temperature reading

  // Usage metrics
  double _dailyTotalUsage = 0.0;
  double _weeklyAverage = 0.0;
  double _monthlyAverage = 0.0;
  double _totalUsage = 0.0;
  double _peakFlow = 0.0;
  DateTime? _currentDay;
  double _previousTotalVolume = 0.0;

  // Time segments
  final List<String> _timeSegmentLabels = [
    'Night (12AM-6AM)',
    'Morning (6AM-9AM)',
    'Daytime (9AM-5PM)',
    'Evening (5PM-12AM)',
  ];
  List<double> _timeSegmentsUsage = [0.0, 0.0, 0.0, 0.0];

  // Historical data
  List<double> _weeklyData = List.filled(7, 0.0);
  final List<double> _monthlyData = List.filled(30, 0.0);
  List<double> _dailyPeakFlows = List.filled(7, 0.0);
  List<double> _lastWeekData = List.filled(7, 0.0); // For comparison

  // Cost and billing
  final double _tariffRate = 5.0; // Cost per cubic meter (adjust as needed)
  double _currentCost = 0.0;
  double _projectedMonthlyCost = 0.0;
  final DateTime _billingCycleStart = DateTime.now().subtract(
    const Duration(days: 15),
  );
  final DateTime _billingCycleEnd = DateTime.now().add(
    const Duration(days: 15),
  );

  // Environmental impact
  double _waterSaved = 0.0; // Compared to average household
  double _carbonReduction = 0.0; // In kg CO2 equivalent

  // AI and prediction system
  double _predictedUsage = 0.0;
  double _anomalyScore = 0.0;
  bool _isAnomaly = false;
  String _usagePattern = 'Normal';
  String _predictionConfidence = 'Medium';
  final List<double> _usageHistory = [];
  final int _maxHistoryLength = 100;
  Timer? _predictionTimer;
  Timer? _anomalyCheckTimer;
  final List<Map<String, dynamic>> _predictionHistory = [];
  final int _maxPredictionHistory = 20;

  // API Configuration
  static const String _mlApiUrl = 'http://10.11.13.217:5000'; // Local server

  // Visualization
  final List<FlSpot> _hourlyUsageSpots = [];
  final List<FlSpot> _predictionSpots = [];
  final List<Color> _gradientColors = [Colors.blue, Colors.lightBlue.shade300];

  // Animation
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  // UI Control
  int _selectedTimeFrame = 0; // 0: Day, 1: Week, 2: Month
  final List<String> _timeFrames = ['Day', 'Week', 'Month'];

  @override
  void initState() {
    super.initState();

    // Initialize animations
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..forward();

    _fadeAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );

    // Initialize data listeners
    _initializeDatabaseListener();
    _initializePredictionSystem();

    // Calculate initial values
    _calculateCosts();
    _calculateEnvironmentalImpact();
  }

  void _initializePredictionSystem() {
    // Main prediction timer (every 5 minutes)
    _predictionTimer = Timer.periodic(const Duration(minutes: 5), (timer) {
      if (_currentFlowRate > 0) {
        _getUsagePrediction();
      }
    });

    // More frequent anomaly checks (every 2 minutes)
    _anomalyCheckTimer = Timer.periodic(const Duration(minutes: 2), (timer) {
      if (_currentFlowRate > 0) {
        _checkForAnomalies();
      }
    });
  }

  Future<void> _getUsagePrediction() async {
    try {
      final currentHour = DateTime.now().hour;
      final currentWeekday = DateTime.now().weekday;

      // Prepare historical data (last 24 hours)
      final historicalData = _usageHistory.sublist(
        max(0, _usageHistory.length - 24),
        _usageHistory.length,
      );

      final response = await http.post(
        Uri.parse('$_mlApiUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'flow_rate': _currentFlowRate,
          'volume': _currentVolume,
          'pressure': _currentPressure,
          'temperature': _currentTemperature,
          'hour': currentHour,
          'weekday': currentWeekday,
          'historical_usage': historicalData,
          'weekly_average': _weeklyAverage,
          'monthly_average': _monthlyAverage,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _predictedUsage = data['predicted_usage']?.toDouble() ?? 0.0;
          _usagePattern = data['usage_pattern'] ?? 'Normal';
          _predictionConfidence = data['confidence'] ?? 'Medium';

          // Add to prediction history
          if (_predictionHistory.length >= _maxPredictionHistory) {
            _predictionHistory.removeAt(0);
          }
          _predictionHistory.add({
            'timestamp': DateTime.now(),
            'prediction': _predictedUsage,
            'confidence': _predictionConfidence,
          });

          // Update visualization data
          _updatePredictionVisualization();
        });
      }
    } catch (e) {
      debugPrint('Prediction error: $e');
      setState(() {
        _predictionConfidence = 'Low';
      });
    }
  }

  void _updatePredictionVisualization() {
    // Update hourly usage spots (last 12 hours)
    _hourlyUsageSpots.clear();
    final historyLength = min(_usageHistory.length, 12);
    for (int i = 0; i < historyLength; i++) {
      _hourlyUsageSpots.add(
        FlSpot(
          (12 - historyLength + i).toDouble(),
          _usageHistory[_usageHistory.length - historyLength + i],
        ),
      );
    }

    // Update prediction spots (next 6 hours)
    _predictionSpots.clear();
    for (int i = 1; i <= 6; i++) {
      _predictionSpots.add(
        FlSpot(
          (12 + i).toDouble(),
          _predictedUsage * (1 + 0.1 * i), // Simulated prediction with growth
        ),
      );
    }
  }

  Future<void> _checkForAnomalies() async {
    try {
      final response = await http.post(
        Uri.parse('$_mlApiUrl/detect_anomaly'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'current_flow': _currentFlowRate,
          'current_volume': _currentVolume,
          'current_pressure': _currentPressure,
          'historical_average': _weeklyAverage,
          'time_segment': _getTimeSegmentIndex(DateTime.now()),
          'daily_usage': _dailyTotalUsage,
          'hourly_trend': _usageHistory.sublist(
            max(0, _usageHistory.length - 6),
            _usageHistory.length,
          ),
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _anomalyScore = data['anomaly_score']?.toDouble() ?? 0.0;
          _isAnomaly = data['is_anomaly'] ?? false;
          if (_isAnomaly) {
            _showAnomalyAlert(data['confidence'] ?? 'high');
          }
        });
      }
    } catch (e) {
      debugPrint('Anomaly detection error: $e');
    }
  }

  void _showAnomalyAlert(String confidence) {
    final alertColor = confidence == 'high' ? Colors.red : Colors.orange;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.warning, color: Colors.white),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Potential water ${confidence == 'high' ? 'leak' : 'waste'} detected!',
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ],
        ),
        backgroundColor: alertColor,
        duration: const Duration(seconds: 10),
        action: SnackBarAction(
          label: 'DETAILS',
          textColor: Colors.white,
          onPressed: () => _showAnomalyDetailsDialog(),
        ),
      ),
    );
  }

  int _getTimeSegmentIndex(DateTime time) {
    final hour = time.hour;
    if (hour >= 0 && hour < 6) return 0;
    if (hour >= 6 && hour < 9) return 1;
    if (hour >= 9 && hour < 17) return 2;
    return 3;
  }

  Future<void> _initializeDatabaseListener() async {
    try {
      final database = FirebaseDatabase.instanceFor(
        app: Firebase.app(),
        databaseURL:
            'https://smart-water-metering-sys-default-rtdb.firebaseio.com',
      );

      _databaseRef = database.ref('sensor_readings');
      _usageStatsRef = database.ref('USAGE_STATISTICS');

      final sensorDataSnapshot = await _databaseRef.get();
      if (sensorDataSnapshot.exists) {
        final allData = sensorDataSnapshot.value as Map<dynamic, dynamic>;
        allData.forEach((key, value) {
          if (value is Map<dynamic, dynamic>) {
            _updateRealtimeData(value);
          }
        });
      }

      final totalUsageSnapshot =
          await _usageStatsRef.child('total_usage').get();
      if (totalUsageSnapshot.exists) {
        setState(
          () => _totalUsage = (totalUsageSnapshot.value as num).toDouble(),
        );
      }

      final weeklySnapshot = await _usageStatsRef.child('weekly_data').get();
      if (weeklySnapshot.exists && weeklySnapshot.value != null) {
        setState(
          () =>
              _weeklyData = List<double>.from(
                weeklySnapshot.value as List<dynamic>,
              ),
        );
      }

      final lastWeekSnapshot =
          await _usageStatsRef.child('last_week_data').get();
      if (lastWeekSnapshot.exists && lastWeekSnapshot.value != null) {
        setState(
          () =>
              _lastWeekData = List<double>.from(
                lastWeekSnapshot.value as List<dynamic>,
              ),
        );
      }

      final peaksSnapshot = await _usageStatsRef.child('daily_peaks').get();
      if (peaksSnapshot.exists && peaksSnapshot.value != null) {
        setState(
          () =>
              _dailyPeakFlows = List<double>.from(
                peaksSnapshot.value as List<dynamic>,
              ),
        );
      }

      _databaseSubscription = _databaseRef.onChildAdded.listen((
        DatabaseEvent event,
      ) {
        if (event.snapshot.value != null) {
          final data = event.snapshot.value as Map<dynamic, dynamic>;
          _updateRealtimeData(data);
        }
      }, onError: (error) => _handleError(error.toString()));

      setState(() => _isLoading = false);
    } catch (e) {
      _handleError(e.toString());
    }
  }

  void _updateRealtimeData(Map<dynamic, dynamic> data) {
    try {
      DateTime dateTime;
      final timestampRaw = data['timestamp'];

      if (timestampRaw is String) {
        try {
          dateTime = DateTime.parse(timestampRaw).toLocal();
        } catch (_) {
          debugPrint('Skipping invalid timestamp: $timestampRaw');
          return;
        }
      } else {
        final timestamp =
            (timestampRaw as num?)?.toInt() ??
            DateTime.now().millisecondsSinceEpoch ~/ 1000;
        dateTime =
            DateTime.fromMillisecondsSinceEpoch(
              timestamp * 1000,
              isUtc: true,
            ).toLocal();
      }

      if (_currentDay?.day != dateTime.day) {
        _timeSegmentsUsage = [0.0, 0.0, 0.0, 0.0];
        _peakFlow = 0.0;
        _previousTotalVolume = 0.0;
        _dailyTotalUsage = 0.0;
        _currentDay = dateTime;
      }

      final currentTotalVolume = _safeParseDouble(data['total_volume']) ?? 0.0;
      final flowRate = _safeParseDouble(data['Flow_rate']) ?? 0.0;
      final pressure = _safeParseDouble(data['pressure']) ?? 0.0;
      final temperature = _safeParseDouble(data['temperature']) ?? 0.0;

      double incrementalVolume = currentTotalVolume - _previousTotalVolume;
      incrementalVolume = incrementalVolume < 0 ? 0 : incrementalVolume;
      _previousTotalVolume = currentTotalVolume;

      if (_usageHistory.length >= _maxHistoryLength) {
        _usageHistory.removeAt(0);
      }
      _usageHistory.add(incrementalVolume);

      final segmentIndex = _getTimeSegmentIndex(dateTime);
      _timeSegmentsUsage[segmentIndex] += incrementalVolume;
      _dailyTotalUsage += incrementalVolume;

      _updateWeeklyData(incrementalVolume, dateTime);
      _updateMonthlyData(incrementalVolume, dateTime);

      if (flowRate > _peakFlow) _peakFlow = flowRate;

      if (mounted) {
        setState(() {
          _currentFlowRate = flowRate;
          _currentVolume = currentTotalVolume;
          _currentPressure = pressure;
          _currentTemperature = temperature;
        });

        // Recalculate derived values
        _calculateCosts();
        _calculateEnvironmentalImpact();
      }

      if (incrementalVolume > 5.0) {
        _getUsagePrediction();
      }
    } catch (e) {
      debugPrint('Error updating data: $e');
    }
  }

  void _calculateCosts() {
    // Calculate current cost (convert liters to cubic meters)
    _currentCost = (_currentVolume / 1000) * _tariffRate;

    // Calculate projected monthly cost
    final daysInMonth =
        DateTime(DateTime.now().year, DateTime.now().month + 1, 0).day;
    final daysSoFar = DateTime.now().day;
    final dailyAverage = _dailyTotalUsage / max(1, daysSoFar);
    _projectedMonthlyCost = (dailyAverage * daysInMonth / 1000) * _tariffRate;
  }

  void _calculateEnvironmentalImpact() {
    // Calculate water saved compared to average household (150L/person/day)
    const averageDailyUsage = 150 * 4; // Assuming 4-person household
    _waterSaved = max(0, averageDailyUsage - _dailyTotalUsage);

    // Calculate carbon reduction (0.0003 kg CO2 per liter of water saved)
    _carbonReduction = _waterSaved * 0.0003;
  }

  double? _safeParseDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    if (value is String) {
      try {
        return double.parse(value);
      } catch (_) {
        return null;
      }
    }
    return null;
  }

  void _updateWeeklyData(double volume, DateTime dateTime) {
    int weekday = dateTime.weekday - 1;
    if (weekday >= 0 && weekday < 7) {
      _weeklyData[weekday] += volume;
      if (_currentFlowRate > _dailyPeakFlows[weekday]) {
        _dailyPeakFlows[weekday] = _currentFlowRate;
      }
    }
  }

  void _updateMonthlyData(double volume, DateTime dateTime) {
    int dayOfMonth = dateTime.day - 1;
    if (dayOfMonth < _monthlyData.length) {
      _monthlyData[dayOfMonth] += volume;
    }
  }

  void _calculateAverages() {
    try {
      _weeklyAverage =
          _weeklyData.isNotEmpty
              ? _weeklyData.reduce((a, b) => a + b) / 7
              : 0.0;

      final now = DateTime.now();
      final daysInMonth = DateTime(now.year, now.month + 1, 0).day;

      final monthlySum =
          _monthlyData.isNotEmpty ? _monthlyData.reduce((a, b) => a + b) : 0.0;

      final daysToCalculate = min(now.day, daysInMonth);
      _monthlyAverage =
          daysToCalculate > 0 ? monthlySum / daysToCalculate : 0.0;
    } catch (e) {
      debugPrint('Error calculating averages: $e');
      _weeklyAverage = 0.0;
      _monthlyAverage = 0.0;
    }
  }

  void _handleError(String error) {
    debugPrint('Error: $error');
    if (mounted) {
      setState(() {
        _hasError = true;
        _errorMessage = error;
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $error'),
          duration: const Duration(seconds: 5),
          action: SnackBarAction(
            label: 'Retry',
            onPressed: _initializeDatabaseListener,
          ),
        ),
      );
    }
  }

  void _onItemTapped(int index) {
    setState(() => _selectedIndex = index);
    switch (index) {
      case 0:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const HomeScreen()),
        );
      case 2:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const PaymentScreen()),
        );
      case 3:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const SettingsScreen()),
        );
    }
  }

  // UI Components
  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Water Analytics',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue.shade800,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Real-time monitoring & insights',
                style: TextStyle(fontSize: 14, color: Colors.blue.shade600),
              ),
            ],
          ),
          Icon(Icons.water_drop, size: 36, color: Colors.blue.shade700),
        ],
      ),
    );
  }

  Widget _buildTimeFrameSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: List.generate(_timeFrames.length, (index) {
          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedTimeFrame = index;
              });
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color:
                    _selectedTimeFrame == index
                        ? Colors.blue
                        : Colors.transparent,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color:
                      _selectedTimeFrame == index
                          ? Colors.blue
                          : Colors.grey.shade300,
                ),
              ),
              child: Text(
                _timeFrames[index],
                style: TextStyle(
                  color:
                      _selectedTimeFrame == index
                          ? Colors.white
                          : Colors.grey.shade700,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildRealtimeMetrics() {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 1.5,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      children: [
        _MetricCard(
          title: 'Flow Rate',
          value: _currentFlowRate.toStringAsFixed(2),
          unit: 'L/min',
          icon: Icons.speed,
          color: Colors.blue,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Current Volume',
          value: _currentVolume.toStringAsFixed(2),
          unit: 'Liters',
          icon: Icons.water_drop,
          color: Colors.lightBlue,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Pressure',
          value: _currentPressure.toStringAsFixed(1),
          unit: 'psi',
          icon: Icons.compress,
          color: Colors.purple,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Temperature',
          value: _currentTemperature.toStringAsFixed(1),
          unit: '°C',
          icon: Icons.thermostat,
          color: Colors.orange,
          isLoading: _isLoading,
        ),
      ],
    );
  }

  Widget _buildUsageOverview() {
    _calculateAverages();
    final today = DateTime.now().weekday - 1;

    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 1.5,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      children: [
        _MetricCard(
          title: 'Today\'s Usage',
          value: _dailyTotalUsage.toStringAsFixed(1),
          unit: 'Liters',
          icon: Icons.today,
          color: Colors.blue,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Today\'s Peak',
          value: _dailyPeakFlows[today].toStringAsFixed(1),
          unit: 'L/min',
          icon: Icons.opacity,
          color: Colors.lightBlue,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Weekly Average',
          value: _weeklyAverage.toStringAsFixed(1),
          unit: 'L/day',
          icon: Icons.calendar_view_week,
          color: Colors.green,
          isLoading: _isLoading,
        ),
        _MetricCard(
          title: 'Monthly Average',
          value: _monthlyAverage.toStringAsFixed(1),
          unit: 'L/day',
          icon: Icons.calendar_month,
          color: Colors.orange,
          isLoading: _isLoading,
        ),
      ],
    );
  }

  Widget _buildCostInsights() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.attach_money, color: Colors.green.shade700),
                const SizedBox(width: 8),
                Text(
                  'Cost & Billing Insights',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.green.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Current Cost',
                  style: TextStyle(color: Colors.grey.shade700),
                ),
                Text(
                  '\$${_currentCost.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Projected Monthly',
                  style: TextStyle(color: Colors.grey.shade700),
                ),
                Text(
                  '\$${_projectedMonthlyCost.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value:
                  DateTime.now().difference(_billingCycleStart).inDays /
                  _billingCycleEnd.difference(_billingCycleStart).inDays,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(Colors.green.shade600),
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
            ),
            const SizedBox(height: 8),
            Text(
              'Billing Cycle: ${DateFormat('MMM d').format(_billingCycleStart)} - ${DateFormat('MMM d').format(_billingCycleEnd)}',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 16),
            Text(
              '💡 Tip: Shorter showers can save up to 20% of your water usage',
              style: TextStyle(
                fontSize: 12,
                fontStyle: FontStyle.italic,
                color: Colors.green.shade700,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEnvironmentalImpact() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.eco, color: Colors.green.shade700),
                const SizedBox(width: 8),
                Text(
                  'Environmental Impact',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.green.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Column(
                  children: [
                    Icon(Icons.water_drop, size: 30, color: Colors.blue),
                    const SizedBox(height: 8),
                    Text(
                      'Water Saved',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${_waterSaved.toStringAsFixed(1)} L',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Icon(Icons.cloud, size: 30, color: Colors.green),
                    const SizedBox(height: 8),
                    Text(
                      'CO₂ Reduction',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${_carbonReduction.toStringAsFixed(3)} kg',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Compared to average household usage',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAIIntegration() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.psychology, color: Colors.purple.shade700),
                const SizedBox(width: 8),
                Text(
                  'AI Water Insights',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.purple.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Column(
                  children: [
                    Icon(
                      _isAnomaly ? Icons.warning : Icons.check_circle,
                      size: 30,
                      color: _isAnomaly ? Colors.red : Colors.green,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Status',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _isAnomaly ? 'Alert' : 'Normal',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: _isAnomaly ? Colors.red : Colors.green,
                      ),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Icon(
                      Icons.trending_up,
                      size: 30,
                      color:
                          _predictedUsage > _weeklyAverage
                              ? Colors.orange
                              : Colors.green,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Prediction',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${_predictedUsage.toStringAsFixed(1)} L',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color:
                            _predictedUsage > _weeklyAverage
                                ? Colors.orange
                                : Colors.green,
                      ),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Icon(
                      _predictionConfidence == 'High'
                          ? Icons.verified
                          : _predictionConfidence == 'Medium'
                          ? Icons.help_outline
                          : Icons.error_outline,
                      size: 30,
                      color:
                          _predictionConfidence == 'High'
                              ? Colors.green
                              : _predictionConfidence == 'Medium'
                              ? Colors.orange
                              : Colors.red,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Confidence',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _predictionConfidence,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color:
                            _predictionConfidence == 'High'
                                ? Colors.green
                                : _predictionConfidence == 'Medium'
                                ? Colors.orange
                                : Colors.red,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (_isAnomaly)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning, color: Colors.red.shade700),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Unusual usage pattern detected. Possible leak or waste.',
                        style: TextStyle(color: Colors.red.shade700),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildWeeklyComparison() {
    final days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    final todayIndex = DateTime.now().weekday - 1;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Weekly Comparison (This Week vs Last Week)',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            SizedBox(
              height: 250,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  barGroups: List.generate(7, (index) {
                    final thisWeekValue = _weeklyData[index];
                    final lastWeekValue = _lastWeekData[index];

                    return BarChartGroupData(
                      x: index,
                      barRods: [
                        BarChartRodData(
                          toY: thisWeekValue,
                          color: Colors.blue,
                          width: 12,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        BarChartRodData(
                          toY: lastWeekValue,
                          color: Colors.grey.shade400,
                          width: 12,
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ],
                    );
                  }),
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          return Padding(
                            padding: const EdgeInsets.only(top: 8.0),
                            child: Text(
                              days[value.toInt()],
                              style: TextStyle(
                                color:
                                    value.toInt() == todayIndex
                                        ? Colors.blue
                                        : Colors.grey.shade700,
                                fontWeight:
                                    value.toInt() == todayIndex
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    leftTitles: const AxisTitles(
                      axisNameWidget: Text('Usage (L)'),
                      sideTitles: SideTitles(showTitles: true),
                    ),
                  ),
                  borderData: FlBorderData(show: true),
                  gridData: const FlGridData(show: true),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(width: 12, height: 12, color: Colors.blue),
                const SizedBox(width: 8),
                const Text('This Week'),
                const SizedBox(width: 20),
                Container(width: 12, height: 12, color: Colors.grey.shade400),
                const SizedBox(width: 8),
                const Text('Last Week'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSmartRecommendations() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.lightbulb_outline, color: Colors.amber.shade700),
                const SizedBox(width: 8),
                Text(
                  'Smart Recommendations',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.amber.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildRecommendationItem(
              Icons.plumbing,
              'Check for leaks',
              'Your usage pattern suggests a possible leak in the system',
              Colors.red.shade100,
            ),
            const SizedBox(height: 12),
            _buildRecommendationItem(
              Icons.schedule,
              'Optimize irrigation timing',
              'Water your garden early morning to reduce evaporation',
              Colors.green.shade100,
            ),
            const SizedBox(height: 12),
            _buildRecommendationItem(
              Icons.upgrade,
              'Consider smart valves',
              'Upgrade to smart valves for better control and savings',
              Colors.blue.shade100,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationItem(
    IconData icon,
    String title,
    String description,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(icon, size: 30),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Text(
                  description,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomNav() {
    return SafeArea(
      child: Container(
        height: 70,
        decoration: const BoxDecoration(
          border: Border(top: BorderSide(color: Colors.grey, width: 0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _NavButton(
              icon: Icons.home,
              label: "Home",
              isActive: _selectedIndex == 0,
              onTap: () => _onItemTapped(0),
            ),
            _NavButton(
              icon: Icons.analytics,
              label: "Stats",
              isActive: _selectedIndex == 1,
              onTap: () => _onItemTapped(1),
            ),
            _NavButton(
              icon: Icons.payment,
              label: "Pay",
              isActive: _selectedIndex == 2,
              onTap: () => _onItemTapped(2),
            ),
            _NavButton(
              icon: Icons.settings,
              label: "Settings",
              isActive: _selectedIndex == 3,
              onTap: () => _onItemTapped(3),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorScreen() {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 50),
            const SizedBox(height: 20),
            Text(
              'Error Loading Data',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 10),
            Text(
              _errorMessage,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _initializeDatabaseListener,
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  @override
  void dispose() {
    _databaseSubscription.cancel();
    _animationController.dispose();
    _predictionTimer?.cancel();
    _anomalyCheckTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 20),
              Text('Loading water usage data...'),
            ],
          ),
        ),
      );
    }

    if (_hasError) return _buildErrorScreen();

    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 80.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            const SizedBox(height: 20),
            _buildTimeFrameSelector(),
            const SizedBox(height: 20),
            Text(
              'Real-time Metrics',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 12),
            _buildRealtimeMetrics(),
            const SizedBox(height: 20),
            Text(
              'Usage Overview',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 12),
            _buildUsageOverview(),
            const SizedBox(height: 20),
            _buildCostInsights(),
            const SizedBox(height: 20),
            _buildEnvironmentalImpact(),
            const SizedBox(height: 20),
            _buildAIIntegration(),
            const SizedBox(height: 20),
            const SizedBox(height: 20),
            _buildWeeklyComparison(),
            const SizedBox(height: 20),
            _buildSmartRecommendations(),
            const SizedBox(height: 40), // Extra padding at the bottom
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  void _showAnomalyDetailsDialog() {}
}

class _MetricCard extends StatelessWidget {
  final String title;
  final String value;
  final String unit;
  final IconData icon;
  final Color color;
  final bool isLoading;

  const _MetricCard({
    required this.title,
    required this.value,
    required this.unit,
    required this.icon,
    required this.color,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: color, size: 20),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: Colors.grey.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            isLoading
                ? const LinearProgressIndicator()
                : Text(
                  value,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
            const SizedBox(height: 4),
            Text(
              unit,
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }
}

class _NavButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final VoidCallback onTap;

  const _NavButton({
    required this.icon,
    required this.label,
    this.isActive = false,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color:
              isActive
                  ? Colors.blueAccent.withOpacity(0.2)
                  : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isActive ? Colors.blueAccent : Colors.grey,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: isActive ? Colors.blueAccent : Colors.grey,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
