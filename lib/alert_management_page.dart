import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AlertManagementPage extends StatefulWidget {
  const AlertManagementPage({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _AlertManagementPageState createState() => _AlertManagementPageState();
}

class _AlertManagementPageState extends State<AlertManagementPage> {
  final List<Map<String, dynamic>> _alerts = [
    {'type': 'Leak', 'location': 'Zone A', 'time': '2h ago', 'resolved': false},
    {
      'type': 'Low Pressure',
      'location': 'Main Line',
      'time': '5h ago',
      'resolved': true,
    },
    {
      'type': 'High Usage',
      'location': 'User 5',
      'time': '8h ago',
      'resolved': false,
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Settings',
          style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.blueGrey[800],
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              // Add functionality to refresh alerts
              setState(() {
                // Dummy refresh action
                _alerts.shuffle();
              });
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Alert List',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ..._alerts.map((alert) => AlertCard(alert: alert)),
            const SizedBox(height: 20),
            Center(
              child: ElevatedButton.icon(
                onPressed: () {
                  // Add functionality to resolve an alert
                  setState(() {
                    for (var alert in _alerts) {
                      alert['resolved'] = true;
                    }
                  });
                },
                icon: const Icon(Icons.check_circle),
                label: const Text('Resolve All Alerts'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  textStyle: const TextStyle(fontSize: 16),
                ),
              ),
            ),
            const SizedBox(height: 40),
            AlertStatsChart(alerts: _alerts),
          ],
        ),
      ),
    );
  }
}

class AlertCard extends StatelessWidget {
  final Map<String, dynamic> alert;

  const AlertCard({super.key, required this.alert});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        contentPadding: const EdgeInsets.all(16),
        leading: Icon(
          Icons.warning,
          color: alert['resolved'] ? Colors.green : Colors.red,
          size: 32,
        ),
        title: Text(
          alert['type'].toString(),
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            Text('Location: ${alert['location']}'),
            Text('Time: ${alert['time']}'),
          ],
        ),
        trailing: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            // ignore: deprecated_member_use
            color:
                alert['resolved']
                    ? Colors.green.withAlpha((0.2 * 255).toInt())
                    : Colors.red.withAlpha((0.2 * 255).toInt()),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            alert['resolved'] ? 'Resolved' : 'Pending',
            style: TextStyle(
              color: alert['resolved'] ? Colors.green : Colors.red,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        onTap: () {
          // Add functionality to view alert details
        },
      ),
    );
  }
}

class AlertStatsChart extends StatelessWidget {
  final List<Map<String, dynamic>> alerts;

  const AlertStatsChart({super.key, required this.alerts});

  @override
  Widget build(BuildContext context) {
    int pendingAlerts = alerts.where((alert) => !alert['resolved']).length;
    int resolvedAlerts = alerts.length - pendingAlerts;

    double totalAlerts = alerts.length.toDouble();
    double pendingRatio = pendingAlerts / totalAlerts;
    double resolvedRatio = resolvedAlerts / totalAlerts;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Alert Statistics',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Container(
          height: 30, // Height of the rectangle
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.grey),
          ),
          child: Row(
            children: [
              Expanded(
                flex: (pendingRatio * 100).round(),
                child: Container(
                  decoration: BoxDecoration(
                    // ignore: deprecated_member_use
                    color: Colors.red.withOpacity(0.6 + (pendingRatio * 0.4)),
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(8),
                      bottomLeft: Radius.circular(8),
                    ),
                  ),
                  child: Center(
                    child: Text(
                      '$pendingAlerts Pending',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
              Expanded(
                flex: (resolvedRatio * 100).round(),
                child: Container(
                  decoration: BoxDecoration(
                    // ignore: deprecated_member_use
                    color: Colors.green.withOpacity(
                      0.6 + (resolvedRatio * 0.4),
                    ),
                    borderRadius: const BorderRadius.only(
                      topRight: Radius.circular(8),
                      bottomRight: Radius.circular(8),
                    ),
                  ),
                  child: Center(
                    child: Text(
                      '$resolvedAlerts Resolved',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
