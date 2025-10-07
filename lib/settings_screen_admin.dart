// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart' show Provider;
import 'home_screen_admin.dart';
import 'analytics_page.dart';
import 'main.dart';
import 'user_management_page.dart';
import 'billing_management_page.dart';
import 'meter_management_page.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  SettingsScreenState createState() => SettingsScreenState();
}

class SettingsScreenState extends State<SettingsScreen>
    with TickerProviderStateMixin {
  int _currentIndex = 5; // Settings is at index 5
  double _simulatedPressure = 3.8;
  bool _isExporting = false;
  bool _systemStatus = true;
  final List<String> _alerts = [
    'Low pressure in Zone 2',
    'Filter maintenance required',
    'Unexpected flow rate variation',
  ];
  final Map<String, bool> _valveStatus = {
    'Main Supply': true,
    'Zone 1': false,
    'Zone 2': true,
    'Backup': false,
  };
  List<Map<String, dynamic>> _waterQualityParams = [
    {'param': 'pH Level', 'min': 6.5, 'max': 8.5, 'current': 7.2, 'trend': 0.1},
    {
      'param': 'Chlorine',
      'min': 0.2,
      'max': 4.0,
      'current': 2.1,
      'trend': -0.05,
    },
    {
      'param': 'Turbidity',
      'min': 0.0,
      'max': 5.0,
      'current': 1.8,
      'trend': 0.02,
    },
  ];

  late AnimationController _pressureController;
  late Animation<double> _pressureAnimation;

  @override
  void initState() {
    super.initState();
    _pressureController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    _pressureAnimation = Tween<double>(begin: 3.5, end: 4.2).animate(
      CurvedAnimation(parent: _pressureController, curve: Curves.easeInOut),
    );

    _simulateWaterQualityChanges();
  }

  void _simulateWaterQualityChanges() {
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _waterQualityParams =
              _waterQualityParams.map((param) {
                final newValue = (param['current'] + param['trend']).clamp(
                  param['min'],
                  param['max'],
                );
                return {...param, 'current': newValue};
              }).toList();
        });
        _simulateWaterQualityChanges();
      }
    });
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return ListTile(
      leading: Icon(icon, color: Colors.teal),
      title: Text(
        title,
        style: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Colors.teal,
        ),
      ),
    ).animate().slideX(delay: 100.ms);
  }

  Widget _buildWaterQualityMonitor() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader('Water Quality', Icons.water_drop),
        children:
            _waterQualityParams
                .map(
                  (param) => ListTile(
                    title: Row(
                      children: [
                        Text(param['param']),
                        const SizedBox(width: 8),
                        Icon(
                          param['current'] > param['max'] ||
                                  param['current'] < param['min']
                              ? Icons.warning_amber
                              : Icons.check_circle,
                          color:
                              param['current'] > param['max'] ||
                                      param['current'] < param['min']
                                  ? Colors.orange
                                  : Colors.green,
                          size: 20,
                        ),
                      ],
                    ),
                    subtitle: LinearProgressIndicator(
                      value:
                          (param['current'] - param['min']) /
                          (param['max'] - param['min']),
                      backgroundColor: Colors.grey[200],
                      color: _getQualityColor(
                        param['current'],
                        param['min'],
                        param['max'],
                      ),
                    ),
                    trailing: AnimatedCount(
                      count: param['current'],
                      duration: 500.ms,
                      style: TextStyle(
                        color: _getQualityColor(
                          param['current'],
                          param['min'],
                          param['max'],
                        ),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                )
                .toList(),
      ),
    ).animate().fadeIn(delay: 200.ms);
  }

  Color _getQualityColor(double value, double min, double max) {
    final range = max - min;
    final position = (value - min) / range;
    if (position < 0.2 || position > 0.8) return Colors.orange;
    if (position < 0.4 || position > 0.6) return Colors.amber;
    return Colors.teal;
  }

  Widget _buildPressureMonitoring() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader('Pressure System', Icons.compress),
        children: [
          ListTile(
            title: AnimatedBuilder(
              animation: _pressureAnimation,
              builder:
                  (context, child) => Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Current Pressure: ${_pressureAnimation.value.toStringAsFixed(1)} bar',
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: _pressureAnimation.value / 5,
                        backgroundColor: Colors.grey[200],
                        color:
                            _pressureAnimation.value > 4.0
                                ? Colors.orange
                                : Colors.teal,
                      ),
                    ],
                  ),
            ),
            trailing: Switch(
              value: _simulatedPressure > 3.5,
              activeColor: Colors.teal,
              onChanged:
                  (value) =>
                      setState(() => _simulatedPressure = value ? 4.0 : 3.0),
            ),
          ),
        ],
      ),
    ).animate().slideX(begin: 0.2);
  }

  Widget _buildDataExporter() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            const Text(
              'System Data Export',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            AnimatedSwitcher(
              duration: 300.ms,
              child:
                  _isExporting
                      ? const CircularProgressIndicator()
                      : FilledButton.icon(
                        icon: const Icon(Icons.cloud_download),
                        label: const Text('Export Now'),
                        onPressed: () => _simulateDataExport(),
                      ),
            ),
          ],
        ),
      ),
    ).animate().scale(delay: 300.ms);
  }

  Widget _buildSystemStatus() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader('System Status', Icons.sensors),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildStatusIndicator(
                      'Pump',
                      Icons.offline_bolt,
                      Colors.green,
                    ),
                    _buildStatusIndicator(
                      'Filter',
                      Icons.filter_alt,
                      Colors.orange,
                    ),
                    _buildStatusIndicator(
                      'Sensors',
                      Icons.sensors,
                      Colors.teal,
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                SwitchListTile(
                  title: const Text('System Power'),
                  value: _systemStatus,
                  onChanged: (value) => setState(() => _systemStatus = value),
                  activeColor: Colors.teal,
                ),
              ],
            ),
          ),
        ],
      ),
    ).animate().scale(delay: 200.ms);
  }

  Widget _buildStatusIndicator(String label, IconData icon, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 32),
        const SizedBox(height: 8),
        Text(label, style: TextStyle(color: color)),
      ],
    );
  }

  Widget _buildAlerts() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader(
          'Alerts & Notifications',
          Icons.notifications,
        ),
        initiallyExpanded: true,
        children: [
          ..._alerts.map(
            (alert) => ListTile(
              leading: const Icon(Icons.warning_amber, color: Colors.orange),
              title: Text(alert),
              trailing: IconButton(
                icon: const Icon(Icons.close),
                onPressed: () => setState(() => _alerts.remove(alert)),
              ),
            ),
          ),
          TextButton.icon(
            icon: const Icon(Icons.notifications_off),
            label: const Text('Dismiss All Alerts'),
            onPressed: () => setState(() => _alerts.clear()),
          ),
        ],
      ),
    ).animate().slideY(begin: 0.5);
  }

  Widget _buildValveControl() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader('Valve Control', Icons.male),
        children: [
          ..._valveStatus.entries.map(
            (entry) => SwitchListTile(
              title: Text(entry.key),
              value: entry.value,
              onChanged:
                  (value) => setState(() => _valveStatus[entry.key] = value),
              secondary: Icon(
                Icons.male,
                color: entry.value ? Colors.teal : Colors.grey,
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(delay: 300.ms);
  }

  Widget _buildHistoricalData() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ExpansionTile(
        title: _buildSectionHeader('Historical Data', Icons.show_chart),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _buildHistoryChart('Pressure (bar)', [3.8, 4.0, 3.9, 4.2, 4.1]),
                const SizedBox(height: 16),
                _buildHistoryChart('pH Level', [7.0, 7.2, 7.1, 7.3, 7.2]),
              ],
            ),
          ),
        ],
      ),
    ).animate().slideX(begin: -0.1);
  }

  Widget _buildHistoryChart(String title, List<double> data) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        SizedBox(height: 100, child: CustomPaint(painter: _ChartPainter(data))),
      ],
    );
  }

  void _simulateDataExport() async {
    setState(() => _isExporting = true);
    await Future.delayed(const Duration(seconds: 2));
    setState(() => _isExporting = false);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.check_circle, color: Colors.white),
            const SizedBox(width: 8),
            const Text('Export completed successfully'),
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.close, size: 20),
              onPressed:
                  () => ScaffoldMessenger.of(context).hideCurrentSnackBar(),
            ),
          ],
        ),
        backgroundColor: Colors.teal,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('System Settings'),
        backgroundColor: Colors.teal[800],
        actions: [
          IconButton(
            icon: Icon(
              themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
            ),
            onPressed:
                () => themeProvider.toggleTheme(!themeProvider.isDarkMode),
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => _simulateWaterQualityChanges(),
            tooltip: 'Refresh Data',
          ).animate().then(delay: 100.ms).scale(),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildSystemStatus(),
            _buildAlerts(),
            _buildWaterQualityMonitor(),
            _buildPressureMonitoring(),
            _buildValveControl(),
            _buildHistoricalData(),
            _buildDataExporter(),
          ].animate(interval: 100.ms).slideX(begin: 0.1),
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
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
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => const AnalyticsPage()),
            );
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
            // Already on settings page
            break;
        }
      },
      selectedItemColor: Colors.teal,
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

  @override
  void dispose() {
    _pressureController.dispose();
    super.dispose();
  }
}

class _ChartPainter extends CustomPainter {
  final List<double> data;
  final Color lineColor = Colors.teal;
  final Color bgColor = Colors.teal.withAlpha((0.1 * 255).toInt());

  _ChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    final paint =
        Paint()
          ..color = lineColor
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke;

    final bgPaint =
        Paint()
          ..color = bgColor
          ..style = PaintingStyle.fill;

    final path = Path();
    final xStep = size.width / (data.length - 1);
    final yMin = data.reduce((a, b) => a < b ? a : b);
    final yMax = data.reduce((a, b) => a > b ? a : b);
    final yScale = size.height / (yMax - yMin);

    final bgPath =
        Path()
          ..moveTo(0, size.height)
          ..lineTo(0, size.height - (data[0] - yMin) * yScale);

    for (int i = 0; i < data.length; i++) {
      final x = xStep * i;
      final y = size.height - (data[i] - yMin) * yScale;
      if (i == 0) {
        path.moveTo(x, y);
        bgPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        bgPath.lineTo(x, y);
      }
    }

    bgPath
      ..lineTo(size.width, size.height)
      ..close();
    canvas.drawPath(bgPath, bgPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class AnimatedCount extends ImplicitlyAnimatedWidget {
  final double count;
  final TextStyle? style;

  const AnimatedCount({
    super.key,
    required this.count,
    this.style,
    super.duration = const Duration(milliseconds: 300),
  });

  @override
  ImplicitlyAnimatedWidgetState<AnimatedCount> createState() =>
      _AnimatedCountState();
}

class _AnimatedCountState extends AnimatedWidgetBaseState<AnimatedCount> {
  Tween<double>? _countTween;

  @override
  void forEachTween(TweenVisitor<dynamic> visitor) {
    _countTween =
        visitor(
              _countTween,
              widget.count,
              (value) => Tween<double>(begin: value as double),
            )
            as Tween<double>;
  }

  @override
  Widget build(BuildContext context) {
    return Text(
      _countTween?.evaluate(animation).toStringAsFixed(2) ?? '',
      style: widget.style,
    );
  }
}
