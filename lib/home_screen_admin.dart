import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:demo/analytics_page.dart';
import 'package:demo/billing_management_page.dart';
import 'package:demo/main.dart';
import 'package:demo/meter_management_page.dart';
//other screens
import 'package:demo/sensor_service.dart';
import 'package:demo/settings_screen_admin.dart';
import 'package:demo/user_management_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

class HomeScreenAdmin extends StatefulWidget {
  const HomeScreenAdmin({super.key});

  @override
  State<HomeScreenAdmin> createState() => _HomeScreenAdminState();
}

class _HomeScreenAdminState extends State<HomeScreenAdmin>
    with SingleTickerProviderStateMixin {
  // Weather API Configuration
  static const String _weatherApiKey = 'fa49971f87eebb578199a5a203e1c1b6';
  static const double _lat = -29.3167;
  static const double _lon = 27.4833;

  // State variables
  List<FlSpot> _liveGraphData = [];
  Map<String, double> _urbanUsage = {};
  Map<String, dynamic>? weatherData;
  bool isLoadingWeather = true;
  String? errorMessage;
  Timer? _dataUpdateTimer;
  late AnimationController _animationController;
  late Stream<QuerySnapshot<Map<String, dynamic>>> _commentsStream;
  late Stream<QuerySnapshot<Map<String, dynamic>>> _userActivityStream;
  User? currentUser;
  double _currentConsumption = 0.0;
  double _previousConsumption = 0.0;
  late SensorService _sensorService; // Add this line

  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    //_sensorService.initialize();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );
    _initializeData();
    _startDataUpdates();
    _fetchWeather();
    _initializeFirebase();

    // Initialize sensor service
    _sensorService = SensorService();
    _sensorService.initialize();

    // Listen for sensor updates
    _sensorService.currentReading.addListener(_updateFromSensor);
    _sensorService.errorState.addListener(_handleSensorError);
  }

  // Update the _updateFromSensor method
  void _updateFromSensor() {
    if (!mounted) return;
    final reading = _sensorService.currentReading.value;

    setState(() {
      _previousConsumption = _currentConsumption;
      // Changed from flowRate to totalVolume
      _currentConsumption = reading.totalVolume;
      _updateLiveGraphData();
      _updateUrbanUsage();
    });
  }

  void _handleSensorError() {
    final error = _sensorService.errorState.value;
    if (error != null) {
      // Log the error but don't show weather error
      if (kDebugMode) {
        print('Sensor error: $error');
      }
      // Only show UI error if it's not a known database event type error
      if (!error.toString().contains('Unknown event type')) {
        _handleWeatherError('Sensor error: $error');
      }
    }
  }

  void _initializeFirebase() async {
    currentUser = FirebaseAuth.instance.currentUser;
    if (currentUser == null) {
      await FirebaseAuth.instance.signInAnonymously();
      currentUser = FirebaseAuth.instance.currentUser;
    }
    _setupStreams();
  }

  void _setupStreams() {
    _commentsStream =
        FirebaseFirestore.instance
            .collection('comments')
            .orderBy('timestamp', descending: true)
            .limit(5)
            .snapshots();

    _userActivityStream =
        FirebaseFirestore.instance
            .collection('user_activity')
            .orderBy('timestamp', descending: true)
            .limit(5)
            .snapshots();
  }

  void _initializeData() {
    _urbanUsage = {
      "Thetsane": 1200.0,
      "Matala": 950.0,
      "Maseru West": 800.0,
      "Stadium Area": 600.0,
    };

    _liveGraphData = List.generate(
      24,
      (index) => FlSpot(
        index.toDouble(),
        (index < 8 ? index * 2 : 24 - index).toDouble(),
      ),
    );
  }

  Future<void> _fetchWeather() async {
    if (!mounted) return;

    setState(() {
      isLoadingWeather = true;
      errorMessage = null;
    });

    try {
      final response = await http.get(
        Uri.parse(
          'https://api.openweathermap.org/data/2.5/weather?lat=$_lat&lon=$_lon&appid=$_weatherApiKey&units=metric',
        ),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (mounted) {
          setState(() {
            weatherData = data;
            isLoadingWeather = false;
          });
        }
      } else {
        _handleWeatherError('Weather error: ${response.statusCode}');
      }
    } catch (e) {
      _handleWeatherError('Error: ${e.toString()}');
    }
  }

  void _handleWeatherError(String message) {
    if (!mounted) return;
    setState(() {
      errorMessage = message;
      isLoadingWeather = false;
    });
  }

  void _startDataUpdates() {
    _dataUpdateTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _fetchWeather();
      // Remove _updateConsumptionData() call since we're using sensor data
    });
  }

  void _updateLiveGraphData() {
    final newData = List<double>.from(_liveGraphData.map((spot) => spot.y));
    newData.removeAt(0);

    // Use actual flow rate from sensor instead of scaled consumption
    final currentFlow = _sensorService.currentReading.value.flowRate.toDouble();
    // Keep previous value if current flow is 0
    final valueToAdd =
        currentFlow > 0 ? currentFlow : (newData.isEmpty ? 0.0 : newData.last);
    newData.add(valueToAdd);

    _liveGraphData =
        newData
            .asMap()
            .entries
            .map((e) => FlSpot(e.key.toDouble(), e.value))
            .toList();
  }

  void _updateUrbanUsage() {
    final reading = _sensorService.currentReading.value;
    final totalFlow = reading.flowRate;

    // Distribute total flow across areas based on their relative proportions
    final totalProportion = _urbanUsage.values.reduce((a, b) => a + b);
    _urbanUsage = _urbanUsage.map(
      (key, value) => MapEntry(key, (value / totalProportion) * totalFlow),
    );
  }

  @override
  void dispose() {
    _dataUpdateTimer?.cancel();
    _animationController.dispose();
    _sensorService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final weather = weatherData?['weather']?[0];
    final main = weatherData?['main'];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin Dashboard'),
        backgroundColor: Colors.blueGrey[800],
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _fetchWeather),
          IconButton(
            icon: Icon(
              themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
            ),
            onPressed:
                () => themeProvider.toggleTheme(!themeProvider.isDarkMode),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            const SizedBox(height: 20),
            _buildRealTimeConsumption(),
            const SizedBox(height: 20),
            _buildUsageSection(),
            const SizedBox(height: 20),
            _buildLiveMetrics(),
            const SizedBox(height: 20),
            _buildWeatherCard(weather, main),
            const SizedBox(height: 20),
            _buildRecentActivity(),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
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

  Widget _buildRealTimeConsumption() {
    return Card(
      elevation: 6,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Real-time Water Consumption',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                final animatedValue =
                    Tween<double>(
                          begin: _previousConsumption,
                          end: _currentConsumption,
                        )
                        .animate(
                          CurvedAnimation(
                            parent: _animationController,
                            curve: Curves.easeOut,
                          ),
                        )
                        .value;

                return Column(
                  children: [
                    Text(
                      '${animatedValue.toStringAsFixed(1)} L/s',
                      style: TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        color: _getConsumptionColor(animatedValue),
                      ),
                    ),
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: animatedValue / 2000,
                      backgroundColor: Colors.grey[200],
                      color: _getConsumptionColor(animatedValue),
                      minHeight: 10,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Total consumption across all areas',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    ).animate().scale(delay: 200.ms);
  }

  Color _getConsumptionColor(double value) {
    if (value > 1500) return Colors.red;
    if (value > 1000) return Colors.orange;
    return Colors.green;
  }

  Widget _buildUsageSection() {
    return Card(
      elevation: 6,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Water Usage by Area',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
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
                          return Padding(
                            padding: const EdgeInsets.only(top: 8.0),
                            child: Text(
                              areas[value.toInt()],
                              style: TextStyle(fontSize: 10),
                            ),
                          );
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
    // Calculate proper min/max values
    final maxY =
        _liveGraphData.isEmpty
            ? 10.0
            : _liveGraphData.map((spot) => spot.y).reduce(max) * 1.2;
    final minY =
        _liveGraphData.isEmpty
            ? 0.0
            : _liveGraphData.map((spot) => spot.y).reduce(min);
    // Ensure minimum range of 1.0 to avoid interval assertion
    final yRange = max(maxY - minY, 1.0);

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(
              'Real-time Flow Rate Trend',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const Text(
              'L/s over last 24 readings',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(show: true),
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        interval: max(
                          yRange / 5,
                          1.0,
                        ), // Ensure interval >= 1.0
                      ),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    topTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                  ),
                  borderData: FlBorderData(show: true),
                  minX: 0,
                  maxX: 23,
                  minY: minY,
                  maxY: maxY,
                  lineBarsData: [
                    LineChartBarData(
                      spots: _liveGraphData,
                      isCurved: true,
                      color: Colors.blueAccent,
                      barWidth: 3,
                      belowBarData: BarAreaData(
                        show: true,
                        gradient: LinearGradient(
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

  IconData _getWeatherIcon(String? weatherMain) {
    switch (weatherMain?.toLowerCase()) {
      case 'clear':
        return Icons.wb_sunny;
      case 'clouds':
        return Icons.cloud;
      case 'rain':
        return Icons.grain;
      case 'snow':
        return Icons.ac_unit;
      case 'thunderstorm':
        return Icons.flash_on;
      default:
        return Icons.help_outline;
    }
  }

  Widget _buildWeatherCard(
    Map<String, dynamic>? weather,
    Map<String, dynamic>? main,
  ) {
    return Card(
      elevation: 4,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.orange.shade200, Colors.yellow.shade100],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Row(
          children: [
            Icon(
              _getWeatherIcon(weather?['main']),
              size: 40,
              color: Colors.orange,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Current Weather",
                    style: TextStyle(color: Colors.deepOrange),
                  ),
                  if (isLoadingWeather)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 8),
                      child: LinearProgressIndicator(),
                    )
                  else if (errorMessage != null)
                    Text(
                      errorMessage!,
                      style: const TextStyle(color: Colors.red),
                    )
                  else
                    Text(
                      '${(weather?['description']?.toString() ?? 'N/A').capitalize()}\n'
                      '${main?['temp']?.round() ?? '--'}°C | '
                      'Feels ${main?['feels_like']?.round() ?? '--'}°C',
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
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

  Widget _buildRecentActivity() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Recent Activity',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ).animate().fadeIn(delay: 300.ms),
        const SizedBox(height: 12),
        StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
          stream: _userActivityStream,
          builder: (context, snapshot) {
            if (snapshot.hasError) {
              return Text('Error: ${snapshot.error}');
            }
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const CircularProgressIndicator();
            }

            final activityItems = snapshot.data!.docs;

            return Column(
              children: [
                ...activityItems.map((doc) {
                  final data = doc.data();
                  return _buildActivityItem(data);
                }),
                _buildRecentComments(),
              ],
            );
          },
        ),
      ],
    );
  }

  Widget _buildActivityItem(Map<String, dynamic> data) {
    final type = data['type'];
    final message = data['message'];
    final timestamp = (data['timestamp'] as Timestamp).toDate();

    Color? iconColor;
    IconData icon;

    switch (type) {
      case 'user_created':
        iconColor = Colors.green;
        icon = Icons.person_add;
        break;
      case 'user_deleted':
        iconColor = Colors.red;
        icon = Icons.person_remove;
        break;
      case 'comment':
        iconColor = Colors.blue;
        icon = Icons.comment;
        break;
      default:
        iconColor = Colors.grey;
        icon = Icons.notifications;
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: Icon(icon, color: iconColor),
        title: Text(message),
        subtitle: Text(_timeAgo(timestamp)),
        trailing: const Icon(Icons.arrow_forward),
      ),
    ).animate().fadeIn(delay: 100.ms);
  }

  String _timeAgo(DateTime date) {
    final now = DateTime.now();
    final difference = now.difference(date);

    if (difference.inMinutes < 1) return 'Just now';
    if (difference.inHours < 1) return '${difference.inMinutes}m ago';
    if (difference.inDays < 1) return '${difference.inHours}h ago';
    return '${difference.inDays}d ago';
  }

  Widget _buildRecentComments() {
    return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
      stream: _commentsStream,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Text('Error: ${snapshot.error}');
        }
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator();
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),
            Text(
              'Recent User Feedback',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ...snapshot.data!.docs.map((doc) {
              final comment = doc.data();
              return _buildCommentItem(comment);
            }),
          ],
        );
      },
    );
  }

  Widget _buildCommentItem(Map<String, dynamic> comment) {
    final text = comment['text'];
    final user = comment['userName'] ?? 'Anonymous';
    final timestamp = (comment['timestamp'] as Timestamp).toDate();

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: const Icon(Icons.comment, color: Colors.blue),
        title: Text(text),
        subtitle: Text('By $user • ${_timeAgo(timestamp)}'),
        trailing: const Icon(Icons.arrow_forward),
      ),
    ).animate().fadeIn(delay: 100.ms);
  }

  Widget _buildBottomNav() {
    return BottomNavigationBar(
      currentIndex: _currentIndex,
      onTap: (index) {
        setState(() {
          _currentIndex = index;
        });
        switch (index) {
          case 0:
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => HomeScreenAdmin()),
            );
            break;
          case 1:
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => AnalyticsPage()),
            );
            break;
          case 2:
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => UserManagementPage()),
            );
            break;
          case 3:
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => BillingManagementPage()),
            );
            break;
          case 4:
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => MeterManagementPage()),
            );
            break;
          case 5:
            Navigator.push(
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

extension StringExtension on String {
  String capitalize() {
    if (isEmpty) return this;
    return "${this[0].toUpperCase()}${substring(1)}";
  }
}
