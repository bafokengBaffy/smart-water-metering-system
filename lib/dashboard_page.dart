import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';

import 'dart:async'; // Import for Timer
import 'dart:math'; // Import for Random

void main() {
  runApp(
    MaterialApp(
      home: DashboardPage(),
      theme: ThemeData.light(),
      darkTheme: ThemeData.dark(),
      themeMode: ThemeMode.light,
    ),
  );
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  DashboardPageState createState() => DashboardPageState();
}

class DashboardPageState extends State<DashboardPage>
    with SingleTickerProviderStateMixin {
  // Simulated dynamic data
  List<FlSpot> _liveGraphData = [];
  List<Map<String, dynamic>> _recentAlerts = [];
  Map<String, double> _urbanUsage = {};
  String _currentWeather = "Sunny, 28°C";
  Timer? _dataUpdateTimer;
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 2),
    );
    _initializeData();
    _startDataUpdates();
  }

  void _initializeData() {
    // Initialize urban area data
    _urbanUsage = {
      "Thetsane": 1200.0,
      "Matala": 950.0,
      "Maseru West": 800.0,
      "Stadium Area": 600.0,
    };

    // Generate initial live graph data
    _liveGraphData = List.generate(
      24,
      (index) => FlSpot(
        index.toDouble(),
        (index < 8 ? index * 2 : 24 - index).toDouble(),
      ),
    );

    // Generate initial alerts
    _recentAlerts = List.generate(
      5,
      (index) => {
        'id': index,
        'message': 'System alert ${index + 1}',
        'timestamp': '${24 - index}h ago',
        'resolved': false,
      },
    );
  }

  void _startDataUpdates() {
    _dataUpdateTimer = Timer.periodic(Duration(seconds: 5), (timer) {
      setState(() {
        // Simulate data changes
        _updateLiveGraphData();
        _updateUrbanUsage();
        _updateAlerts();
        _updateWeather();
      });
    });
  }

  void _updateLiveGraphData() {
    final newData = List<double>.from(_liveGraphData.map((spot) => spot.y));
    newData.removeAt(0);
    newData.add(
      (newData.last + (Random().nextDouble() * 10 - 5)).clamp(0, 100),
    );

    _liveGraphData =
        newData
            .asMap()
            .entries
            .map((e) => FlSpot(e.key.toDouble(), e.value))
            .toList();
  }

  void _updateUrbanUsage() {
    _urbanUsage = _urbanUsage.map(
      (key, value) =>
          MapEntry(key, value * (0.95 + Random().nextDouble() * 0.1)),
    );
  }

  void _updateAlerts() {
    if (_recentAlerts.length > 10) _recentAlerts.removeAt(0);
    _recentAlerts.add({
      'id': _recentAlerts.length,
      'message': 'New system alert',
      'timestamp': 'Just now',
      'resolved': false,
    });
  }

  void _updateWeather() {
    final conditions = ['Sunny', 'Cloudy', 'Rainy', 'Windy'];
    _currentWeather =
        '${conditions[Random().nextInt(conditions.length)]}, '
        '${20 + Random().nextInt(15)}°C';
  }

  @override
  void dispose() {
    _dataUpdateTimer?.cancel();
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            SizedBox(height: 20),
            _buildUsageSection(),
            SizedBox(height: 20),
            _buildLiveMetrics(),
            SizedBox(height: 20),
            _buildWeatherCard(),
            SizedBox(height: 20),
            _buildRecentAlerts(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Text(
          'Admin Dashboard',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        )
        .animate(controller: _animationController)
        .fadeIn(duration: 500.ms)
        .slideY(begin: -0.2);
  }

  Widget _buildUsageSection() {
    return Card(
      elevation: 6,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Water Usage by Area',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            SizedBox(
              height: 250,
              child: BarChart(
                BarChartData(
                  barTouchData: BarTouchData(enabled: true),
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: true),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          final areas = _urbanUsage.keys.toList();
                          return Text(areas[value.toInt()]);
                        },
                      ),
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  barGroups:
                      _urbanUsage.entries.map((entry) {
                        final index = _urbanUsage.keys.toList().indexOf(
                          entry.key,
                        );
                        return BarChartGroupData(
                          x: index,
                          barRods: [
                            BarChartRodData(
                              toY: entry.value,
                              color: _getAreaColor(entry.key),
                              width: 22,
                            ),
                          ],
                        );
                      }).toList(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getAreaColor(String area) {
    const colors = [Colors.blue, Colors.green, Colors.orange, Colors.red];
    return colors[_urbanUsage.keys.toList().indexOf(area) % colors.length];
  }

  Widget _buildLiveMetrics() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            Text(
              'Real-time Consumption',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(show: false),
                  titlesData: FlTitlesData(show: false),
                  borderData: FlBorderData(show: false),
                  minX: 0,
                  maxX: 23,
                  lineBarsData: [
                    LineChartBarData(
                      spots: _liveGraphData,
                      isCurved: true,
                      color: Colors.blueAccent,
                      barWidth: 3,
                      belowBarData: BarAreaData(
                        show: true,
                        gradient: LinearGradient(
                          // ignore: deprecated_member_use
                          colors: [
                            Colors.blueAccent.withAlpha((0.3 * 255).toInt()),
                            Colors.blueAccent.withAlpha((0.1 * 255).toInt()),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWeatherCard() {
    return Card(
      elevation: 4,
      child: Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.orange.shade200, Colors.yellow.shade100],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Row(
          children: [
            Icon(Icons.wb_sunny, size: 40, color: Colors.orange),
            SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Current Weather',
                  style: TextStyle(color: Colors.deepOrange),
                ),
                Text(
                  _currentWeather,
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentAlerts() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Recent Alerts',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ).animate().fadeIn(delay: 300.ms),
        SizedBox(height: 12),
        ..._recentAlerts.reversed.map((alert) => _buildAlertTile(alert)),
      ],
    );
  }

  Widget _buildAlertTile(Map<String, dynamic> alert) {
    return Card(
      margin: EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: Icon(Icons.warning, color: Colors.red),
        title: Text(alert['message']),
        subtitle: Text(alert['timestamp']),
        trailing: Icon(Icons.arrow_forward),
      ),
    ).animate().fadeIn(delay: 100.ms);
  }
}
