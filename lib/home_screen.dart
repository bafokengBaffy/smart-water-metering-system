// ignore_for_file: unused_import, unused_field, prefer_final_fields, deprecated_member_use

import 'dart:async';
import 'dart:math';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';

import 'analysis_screen.dart';
import 'payment_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  // Firebase references
  late DatabaseReference _databaseRef;
  late StreamSubscription<DatabaseEvent> _databaseSubscription;
  User? currentUser;

  // Animation controllers
  late AnimationController _waveController;
  late AnimationController _graphController;

  // Water usage metrics
  double meterReading = 0.0;
  double _previousMeterValue = 0.0;
  double _currentFlowRate = 0.0;
  double _peakFlow = 0.0;
  double _totalUsage = 0.0;
  double _dailyUsage = 0.0;
  double _weeklyAverage = 0.0;
  double _monthlyAverage = 0.0;
  List<double> _monthlyData = List.filled(30, 0.0);
  List<double> _dailyPeakFlows = List.filled(7, 0.0);
  List<double> _weeklyData = List.filled(7, 0.0);
  List<double> _hourlyUsage = List.filled(24, 0.0);

  // Billing cycle
  DateTime _billingCycleStart = DateTime.now().subtract(
    const Duration(days: 12),
  );
  DateTime _billingCycleEnd = DateTime.now().add(const Duration(days: 18));

  // Environmental impact
  double _waterSaved = 125.0; // Compared to average household
  double _carbonReduction = 0.0375; // In kg CO2
  int _communityRank = 42;

  // Meter status
  int _batteryLevel = 85;
  int _signalStrength = 4; // out of 5
  bool _sensorStatus = true; // true = OK, false = Needs Maintenance

  // User profile
  String _userName = "Thabo Moloi";
  String _userAddress = "123 Main St, Maseru";
  String _meterId = "SWM-728491";

  // Graph data
  static const int maxDataPoints = 30;
  List<FlSpot> _graphData = [];
  final List<double> _graphPoints = [];

  @override
  void initState() {
    super.initState();
    _initializeFirebase();
    _setupDatabaseListener();
    _initializeControllers();
  }

  Future<void> _initializeFirebase() async {
    try {
      final currentUser = FirebaseAuth.instance.currentUser;
      if (currentUser == null) {
        await FirebaseAuth.instance.signInAnonymously();
      }
      this.currentUser = FirebaseAuth.instance.currentUser;
    } catch (e) {
      _showError('Failed to initialize authentication');
    }
  }

  void _setupDatabaseListener() async {
    try {
      final database = FirebaseDatabase.instanceFor(
        app: Firebase.app(),
        databaseURL:
            'https://smart-water-metering-sys-default-rtdb.firebaseio.com',
      );

      _databaseRef = database.ref('sensor_readings');

      _databaseSubscription = _databaseRef
          .orderByKey()
          .limitToLast(1)
          .onChildAdded
          .listen(
            (DatabaseEvent event) {
              if (event.snapshot.value != null) {
                final data = event.snapshot.value as Map<dynamic, dynamic>;
                _updateWaterUsage(data);
              }
            },
            onError: (error) {
              _showError('Database listener error: $error');
            },
          );
    } catch (e) {
      _showError('Error setting up database listener: $e');
    }
  }

  void _updateWaterUsage(Map<dynamic, dynamic> data) {
    try {
      String volumeStr =
          data['total_volume']?.toString() ?? data['volume']?.toString() ?? '0';
      volumeStr = volumeStr.replaceAll(RegExp(r'[^0-9.]'), '');

      double totalVolume = double.tryParse(volumeStr) ?? 0.0;
      double flowRate =
          (data['Flow_rate'] ?? data['flow_rate'] ?? 0).toDouble();

      dynamic timestampValue = data['timestamp'];
      int timestamp =
          timestampValue is int
              ? timestampValue
              : timestampValue is String
              ? int.tryParse(timestampValue) ??
                  DateTime.now().millisecondsSinceEpoch ~/ 1000
              : DateTime.now().millisecondsSinceEpoch ~/ 1000;

      if (mounted) {
        setState(() {
          double delta = totalVolume - _previousMeterValue;
          if (delta > 0) {
            _totalUsage += delta;
            _dailyUsage += delta;
          }

          _previousMeterValue = meterReading;
          meterReading = totalVolume;
          _currentFlowRate = flowRate;

          if (flowRate > _peakFlow) {
            _peakFlow = flowRate;
          }

          _updateGraphData();
          _updateHourlyData(delta, timestamp);
          _updateWeeklyData(delta, timestamp);
          _updateMonthlyData(delta, timestamp);
          _calculateAverages();
        });
      }
    } catch (e) {
      _showError('Failed to update water usage display');
    }
  }

  void _updateGraphData() {
    if (_graphPoints.length >= maxDataPoints) {
      _graphPoints.removeAt(0);
    }
    _graphPoints.add(meterReading);

    _graphData =
        _graphPoints
            .asMap()
            .entries
            .map((e) => FlSpot(e.key.toDouble(), e.value))
            .toList();
  }

  void _updateHourlyData(double volume, int timestamp) {
    final dateTime = DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    final hour = dateTime.hour;

    if (hour < _hourlyUsage.length) {
      _hourlyUsage[hour] += volume;
    }
  }

  void _updateWeeklyData(double volume, int timestamp) {
    final dateTime = DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    final weekday = dateTime.weekday - 1;

    if (_weeklyData.isEmpty) {
      _weeklyData = List.filled(7, 0.0);
      _dailyPeakFlows = List.filled(7, 0.0);
    }

    _weeklyData[weekday] += volume;

    if (_currentFlowRate > _dailyPeakFlows[weekday]) {
      _dailyPeakFlows[weekday] = _currentFlowRate;
    }
  }

  void _updateMonthlyData(double volume, int timestamp) {
    final dateTime = DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    final dayOfMonth = dateTime.day - 1;
    if (dayOfMonth < _monthlyData.length) {
      _monthlyData[dayOfMonth] += volume;
    }
  }

  void _calculateAverages() {
    try {
      // Weekly average calculation
      _weeklyAverage =
          _weeklyData.isNotEmpty
              ? _weeklyData.reduce((a, b) => a + b) / 7
              : 0.0;

      // Monthly average calculation
      final now = DateTime.now();
      final daysInMonth = DateTime(now.year, now.month + 1, 0).day;
      final monthlySum =
          _monthlyData.isNotEmpty ? _monthlyData.reduce((a, b) => a + b) : 0.0;
      final daysToCalculate = min(now.day, daysInMonth);
      _monthlyAverage =
          daysToCalculate > 0 ? monthlySum / daysToCalculate : 0.0;
    } catch (e) {
      _weeklyAverage = 0.0;
      _monthlyAverage = 0.0;
    }
  }

  void _initializeControllers() {
    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    _graphController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..forward();

    for (int i = 0; i < 24; i++) {
      _graphPoints.add(0.0);
      _graphData.add(FlSpot(i.toDouble(), _graphPoints[i]));
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Water Management Dashboard'),
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.blue, Colors.lightBlue],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: () {
              setState(() {
                _dailyUsage = 0.0; // Reset daily usage for demo
              });
            },
          ),
          IconButton(
            icon: const Icon(Icons.analytics, color: Colors.white),
            onPressed:
                () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const AnalysisScreen(),
                  ),
                ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 80),
        child: Column(
          children: [
            _buildUsageSnapshot(),
            const SizedBox(height: 16),
            _buildQuickActions(),
            const SizedBox(height: 16),
            _buildVisualGraphs(),
            const SizedBox(height: 16),
            _buildEnvironmentalImpact(),
            const SizedBox(height: 16),
            _buildSmartInsights(),
            const SizedBox(height: 16),
            _buildMeterStatus(),
            const SizedBox(height: 16),
            _buildUserProfile(),
            const SizedBox(height: 16),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildUsageSnapshot() {
    final daysInCycle = _billingCycleEnd.difference(_billingCycleStart).inDays;
    final daysPassed = DateTime.now().difference(_billingCycleStart).inDays;
    final progress = daysPassed / daysInCycle;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            const Text(
              "Usage Snapshot",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildUsageMetric(
                  Icons.water_drop,
                  "Today's Usage",
                  "${_dailyUsage.toStringAsFixed(1)} L",
                  Colors.blue,
                ),
                _buildUsageMetric(
                  Icons.speed,
                  "Live Flow Rate",
                  "${_currentFlowRate.toStringAsFixed(1)} L/min",
                  Colors.green,
                ),
              ],
            ),
            const SizedBox(height: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      "Billing Cycle",
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    Text(
                      "Day $daysPassed of $daysInCycle",
                      style: const TextStyle(fontSize: 14),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                  value: progress,
                  backgroundColor: Colors.grey.shade200,
                  valueColor: AlwaysStoppedAnimation<Color>(
                    Colors.blue.shade700,
                  ),
                  minHeight: 8,
                  borderRadius: BorderRadius.circular(4),
                ),
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      DateFormat('MMM d').format(_billingCycleStart),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    Text(
                      DateFormat('MMM d').format(_billingCycleEnd),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildComparisonMetric("Vs Yesterday", "+5.2%", Colors.green),
                _buildComparisonMetric("Vs Last Week", "-2.8%", Colors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUsageMetric(
    IconData icon,
    String label,
    String value,
    Color color,
  ) {
    return Column(
      children: [
        Icon(icon, size: 30, color: color),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildComparisonMetric(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildQuickActions() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Quick Actions",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildActionButton(Icons.payment, "Pay Bill", () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const PaymentScreen(),
                    ),
                  );
                }),
                _buildActionButton(Icons.download, "Download Invoice", () {
                  _downloadInvoice();
                }),
                _buildActionButton(Icons.notifications, "Set Reminder", () {
                  _setReminder();
                }),
                _buildActionButton(Icons.report_problem, "Report Issue", () {
                  _reportIssue();
                }),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton(IconData icon, String label, VoidCallback onTap) {
    return Column(
      children: [
        IconButton(
          icon: Icon(icon),
          color: Colors.blue.shade700,
          onPressed: onTap,
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
        ),
      ],
    );
  }

  Widget _buildVisualGraphs() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Usage Visualization",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  gridData: const FlGridData(show: false),
                  titlesData: const FlTitlesData(show: false),
                  borderData: FlBorderData(show: false),
                  minX: 0,
                  maxX: 23,
                  minY: 0,
                  maxY:
                      _hourlyUsage.reduce(
                        (max, value) => value > max ? value : max,
                      ) *
                      1.2,
                  lineBarsData: [
                    LineChartBarData(
                      spots:
                          _hourlyUsage
                              .asMap()
                              .entries
                              .map((e) => FlSpot(e.key.toDouble(), e.value))
                              .toList(),
                      isCurved: true,
                      color: Colors.blue.shade700,
                      barWidth: 3,
                      belowBarData: BarAreaData(
                        show: true,
                        gradient: LinearGradient(
                          colors: [
                            Colors.blue.shade700.withOpacity(0.3),
                            Colors.blue.shade100.withOpacity(0.1),
                          ],
                        ),
                      ),
                      dotData: const FlDotData(show: false),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Center(
              child: Text(
                "Hourly Usage (Last 24 Hours)",
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildTimeFrameButton("Day", true),
                _buildTimeFrameButton("Week", false),
                _buildTimeFrameButton("Month", false),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimeFrameButton(String label, bool isSelected) {
    return ElevatedButton(
      onPressed: () {},
      style: ElevatedButton.styleFrom(
        backgroundColor:
            isSelected ? Colors.blue.shade700 : Colors.grey.shade200,
        foregroundColor: isSelected ? Colors.white : Colors.grey.shade700,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      child: Text(label),
    );
  }

  Widget _buildEnvironmentalImpact() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Environmental Impact",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildImpactMetric(
                  Icons.water_drop,
                  "Water Saved",
                  "${_waterSaved.toStringAsFixed(0)} L",
                  Colors.blue,
                ),
                _buildImpactMetric(
                  Icons.eco,
                  "CO₂ Reduction",
                  "${_carbonReduction.toStringAsFixed(3)} kg",
                  Colors.green,
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  const Icon(Icons.emoji_events, color: Colors.amber),
                  const SizedBox(width: 8),
                  const Text(
                    "Community Rank:",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(width: 4),
                  Text(
                    "#$_communityRank",
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.blue,
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    "in Water Conservation",
                    style: TextStyle(fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImpactMetric(
    IconData icon,
    String label,
    String value,
    Color color,
  ) {
    return Column(
      children: [
        Icon(icon, size: 30, color: color),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildSmartInsights() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Smart Insights",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            _buildInsightItem(
              Icons.tips_and_updates,
              "Personalized Tip",
              "Your shower usage increased by 3 minutes yesterday. Try to keep showers under 5 minutes to save water.",
              Colors.blue,
            ),
            const SizedBox(height: 12),
            _buildInsightItem(
              Icons.warning,
              "Anomaly Alert",
              "Unusual water flow detected at 2:30 AM. Check for possible leaks.",
              Colors.orange,
            ),
            const SizedBox(height: 12),
            _buildInsightItem(
              Icons.upgrade,
              "Recommended Upgrade",
              "Consider installing low-flow showerheads to reduce water usage by up to 40%.",
              Colors.green,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInsightItem(
    IconData icon,
    String title,
    String description,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(fontWeight: FontWeight.bold, color: color),
                ),
                const SizedBox(height: 4),
                Text(description, style: const TextStyle(fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMeterStatus() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Meter Health & Status",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatusMetric(
                  Icons.battery_charging_full,
                  "Battery",
                  "$_batteryLevel%",
                  _batteryLevel > 20 ? Colors.green : Colors.red,
                ),
                _buildStatusMetric(
                  Icons.signal_cellular_alt,
                  "Signal",
                  "$_signalStrength/5",
                  _signalStrength > 2 ? Colors.green : Colors.orange,
                ),
                _buildStatusMetric(
                  Icons.sensors,
                  "Sensor",
                  _sensorStatus ? "OK" : "Needs Check",
                  _sensorStatus ? Colors.green : Colors.red,
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (!_sensorStatus)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning, color: Colors.red),
                    const SizedBox(width: 8),
                    const Expanded(
                      child: Text(
                        "Sensor needs maintenance. Please contact support.",
                        style: TextStyle(color: Colors.red),
                      ),
                    ),
                    TextButton(onPressed: () {}, child: const Text("Contact")),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusMetric(
    IconData icon,
    String label,
    String value,
    Color color,
  ) {
    return Column(
      children: [
        Icon(icon, size: 30, color: color),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildUserProfile() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "User Profile",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            ListTile(
              leading: const CircleAvatar(child: Icon(Icons.person)),
              title: Text(_userName),
              subtitle: Text(_userAddress),
            ),
            const Divider(),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Meter ID: $_meterId",
                    style: TextStyle(fontSize: 14, color: Colors.grey.shade700),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "Notification Preferences: Enabled",
                    style: TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "Security: Standard",
                    style: TextStyle(fontSize: 14),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const SettingsScreen(),
                    ),
                  );
                },
                child: const Text("Manage Settings"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _downloadInvoice() {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text("Invoice download started")));
  }

  void _setReminder() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text("Set Reminder"),
            content: const Text("What would you like to be reminded about?"),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text("Cancel"),
              ),
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Reminder set successfully")),
                  );
                },
                child: const Text("Set"),
              ),
            ],
          ),
    );
  }

  void _reportIssue() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text("Report Issue"),
            content: const Text("Describe the issue you're experiencing:"),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text("Cancel"),
              ),
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text("Issue reported successfully"),
                    ),
                  );
                },
                child: const Text("Report"),
              ),
            ],
          ),
    );
  }

  Widget _buildBottomNav() {
    return SafeArea(
      child: Container(
        decoration: BoxDecoration(
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              spreadRadius: 2,
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
          child: BottomAppBar(
            height: 70,
            padding: EdgeInsets.zero,
            color: Colors.white,
            elevation: 8,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _NavButton(
                  icon: Icons.home,
                  label: "Home",
                  isActive: true,
                  onTap: () {},
                ),
                _NavButton(
                  icon: Icons.analytics,
                  label: "Stats",
                  onTap:
                      () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const AnalysisScreen(),
                        ),
                      ),
                ),
                _NavButton(
                  icon: Icons.payment,
                  label: "Pay",
                  onTap:
                      () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const PaymentScreen(),
                        ),
                      ),
                ),
                _NavButton(
                  icon: Icons.settings,
                  label: "Settings",
                  onTap:
                      () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SettingsScreen(),
                        ),
                      ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _waveController.dispose();
    _graphController.dispose();
    _databaseSubscription.cancel();
    super.dispose();
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
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color:
              isActive
                  ? Colors.blueAccent.withOpacity(0.2)
                  : Colors.transparent,
          borderRadius: BorderRadius.circular(10),
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
