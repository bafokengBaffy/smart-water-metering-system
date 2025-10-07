import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';

class UsageHistoryScreen extends StatelessWidget {
  const UsageHistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final List<FlSpot> waterUsageData = [
      FlSpot(0, 100), // Day 1
      FlSpot(1, 150), // Day 2
      FlSpot(2, 200), // Day 3
      FlSpot(3, 180), // Day 4
      FlSpot(4, 250), // Day 5
      FlSpot(5, 300), // Day 6
      FlSpot(6, 280), // Day 7
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Usage History'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Animated Water Usage Chart
            _buildWaterUsageChart(waterUsageData)
                .animate()
                .fadeIn(duration: 500.ms) // Fade in animation
                .slideY(
                  begin: -0.5,
                  end: 0,
                  duration: 500.ms,
                ), // Slide up animation
            const SizedBox(height: 32),
            // Animated Usage Summary Cards
            Column(
              children: [
                _buildUsageCard(
                      title: 'Total Usage',
                      value: '1,200 L',
                      icon: Icons.water_drop,
                      color: Colors.blue,
                    )
                    .animate()
                    .fadeIn(delay: 200.ms, duration: 500.ms)
                    .slideX(begin: -0.5, end: 0, duration: 500.ms),
                const SizedBox(height: 16),
                _buildUsageCard(
                      title: 'Average Daily Usage',
                      value: '171 L',
                      icon: Icons.timeline,
                      color: Colors.green,
                    )
                    .animate()
                    .fadeIn(delay: 400.ms, duration: 500.ms)
                    .slideX(begin: -0.5, end: 0, duration: 500.ms),
                const SizedBox(height: 16),
                _buildUsageCard(
                      title: 'Highest Usage Day',
                      value: '300 L (Sat)',
                      icon: Icons.arrow_upward,
                      color: Colors.orange,
                    )
                    .animate()
                    .fadeIn(delay: 600.ms, duration: 500.ms)
                    .slideX(begin: -0.5, end: 0, duration: 500.ms),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWaterUsageChart(List<FlSpot> data) {
    return Container(
      height: 300,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            // ignore: deprecated_member_use
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            spreadRadius: 2,
          ),
        ],
      ),
      child: LineChart(
        LineChartData(
          gridData: FlGridData(show: false),
          titlesData: FlTitlesData(
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  const days = [
                    'Mon',
                    'Tue',
                    'Wed',
                    'Thu',
                    'Fri',
                    'Sat',
                    'Sun',
                  ];
                  return Text(days[value.toInt()]);
                },
              ),
            ),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  return Text('${value.toInt()}L');
                },
              ),
            ),
            rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: data,
              isCurved: true,
              color: Colors.blue,
              barWidth: 4,
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  colors: [
                    // ignore: deprecated_member_use
                    Colors.blue.withOpacity(0.3),
                    // ignore: deprecated_member_use
                    Colors.blue.withOpacity(0.1),
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
              dotData: FlDotData(show: true),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUsageCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            // ignore: deprecated_member_use
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            spreadRadius: 2,
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              // ignore: deprecated_member_use
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(fontSize: 14, color: Colors.grey[600]),
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
        ],
      ),
    );
  }
}
