// ignore_for_file: prefer_final_fields, deprecated_member_use

import 'package:demo/sensor_service.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';

import 'billing_management_page.dart';
import 'home_screen_admin.dart';
import 'main.dart';
import 'meter_management_page.dart';
import 'settings_screen_admin.dart';
import 'user_management_page.dart';

class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({super.key});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  int _currentIndex = 1;
  late SensorService _sensorService;
  double _dailyUsage = 1250.5;
  double _monthlyAverage = 1428.7;
  int _alertCount = 3;

  final Map<String, double> _usageDistribution = {
    'Residential': 45,
    'Commercial': 30,
    'Industrial': 25,
  };

  final List<FlSpot> _monthlyUsageData = [
    FlSpot(0, 1200),
    FlSpot(1, 1350),
    FlSpot(2, 1280),
    FlSpot(3, 1410),
    FlSpot(4, 1560),
    FlSpot(5, 1620),
    FlSpot(6, 1480),
  ];

  @override
  void initState() {
    super.initState();
    _sensorService = SensorService();
    _sensorService.initialize();
    _setupSensorListeners();
  }

  @override
  void dispose() {
    _sensorService.errorState.removeListener(_handleSensorError);
    _sensorService.dispose();
    super.dispose();
  }

  void _setupSensorListeners() {
    _sensorService.currentReading.addListener(_updateMetrics);
    _sensorService.errorState.addListener(_handleSensorError);
  }

  void _updateMetrics() {
    if (!mounted) return;

    setState(() {
      // Update metrics based on sensor data
      _dailyUsage = _sensorService.currentReading.value as double;
      // Add logic to update other metrics as needed
    });
  }

  void _handleSensorError() {
    if (!mounted) return;

    final error = _sensorService.errorState.value;
    if (error != null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(error)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analytics Dashboard'),
        backgroundColor: Colors.blueGrey[800],
        actions: [
          IconButton(icon: const Icon(Icons.notifications), onPressed: () {}),
          IconButton(
            icon: Icon(
              themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
            ),
            onPressed: () {
              themeProvider.toggleTheme(!themeProvider.isDarkMode);
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildMetricsRow(),
            const SizedBox(height: 20),
            _buildConsumptionChart(),
            const SizedBox(height: 20),
            _buildUsageDistributionChart(),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildMetricsRow() {
    return Row(
      children: [
        _buildMetricCard(
          'Daily Usage',
          '${_dailyUsage.toStringAsFixed(1)} L',
          Icons.water_drop,
          Colors.blue,
        ),
        const SizedBox(width: 10),
        _buildMetricCard(
          'Monthly Avg',
          '${_monthlyAverage.toStringAsFixed(1)} L',
          Icons.av_timer,
          Colors.green,
        ),
        const SizedBox(width: 10),
        _buildMetricCard(
          'Alerts',
          _alertCount.toString(),
          Icons.warning_rounded,
          Colors.orange,
        ),
      ],
    ).animate().fadeIn(delay: 100.ms);
  }

  Widget _buildMetricCard(
    String title,
    String value,
    IconData icon,
    Color color,
  ) {
    return Expanded(
      child: Card(
        elevation: 4,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Icon(icon, color: color, size: 32),
              const SizedBox(height: 8),
              Text(
                title,
                style: const TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildConsumptionChart() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Monthly Consumption Trend',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  gridData: const FlGridData(
                    show: true,
                    drawVerticalLine: false,
                  ),
                  titlesData: FlTitlesData(
                    leftTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: true),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            'Week ${value.toInt() + 1}',
                            style: const TextStyle(fontSize: 10),
                          );
                        },
                      ),
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  lineBarsData: [
                    LineChartBarData(
                      spots: _monthlyUsageData,
                      isCurved: true,
                      color: Colors.blueAccent,
                      barWidth: 3,
                      belowBarData: BarAreaData(
                        show: true,
                        color: Colors.blueAccent.withOpacity(0.1),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    ).animate().slideX(begin: 0.1);
  }

  Widget _buildUsageDistributionChart() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Usage Distribution',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: PieChart(
                PieChartData(
                  sections:
                      _usageDistribution.entries.map((entry) {
                        return PieChartSectionData(
                          color: _getSectionColor(entry.key),
                          value: entry.value,
                          title: '${entry.value}%',
                          radius: 30,
                          titleStyle: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                        );
                      }).toList(),
                ),
              ),
            ),
            const SizedBox(height: 16),
            ..._usageDistribution.keys.map((key) => _buildLegendItem(key)),
          ],
        ),
      ),
    ).animate().scale(delay: 200.ms);
  }

  Widget _buildLegendItem(String category) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(width: 12, height: 12, color: _getSectionColor(category)),
          const SizedBox(width: 8),
          Text(category, style: const TextStyle(fontSize: 14)),
        ],
      ),
    );
  }

  Color _getSectionColor(String category) {
    switch (category) {
      case 'Residential':
        return Colors.blueAccent;
      case 'Commercial':
        return Colors.green;
      case 'Industrial':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  Widget _buildBottomNav() {
    return BottomNavigationBar(
      currentIndex: _currentIndex,
      onTap: (index) {
        setState(() => _currentIndex = index);
        switch (index) {
          case 0:
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => const HomeScreenAdmin()),
            );
            break;
          case 1:
            // Already on analytics page
            break;
          case 2:
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => UserManagementPage()),
            );
            break;
          case 3:
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => BillingManagementPage()),
            );
            break;
          case 4:
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => MeterManagementPage()),
            );
            break;
          case 5:
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => const SettingsScreen()),
            );
            break;
        }
      },
      selectedItemColor: Colors.blueAccent,
      unselectedItemColor: Colors.grey,
      items: const [
        BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
        BottomNavigationBarItem(
          icon: Icon(Icons.analytics),
          label: 'Analytics',
        ),
        BottomNavigationBarItem(icon: Icon(Icons.people), label: 'Users'),
        BottomNavigationBarItem(icon: Icon(Icons.payment), label: 'Billing'),
        BottomNavigationBarItem(icon: Icon(Icons.speed), label: 'Meters'),
        BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Settings'),
      ],
    );
  }
}
