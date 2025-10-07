import 'package:flutter/material.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('About'),
        backgroundColor: Colors.teal,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // App Introduction Section
            _buildSectionTitle('App Introduction'),
            const Text(
              'The Smart Water Metering System is designed to help users monitor and manage water usage efficiently. '
              'It enables smarter decision-making and promotes sustainable water management practices.',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),

            // Features List Section
            _buildSectionTitle('Features'),
            const BulletList(items: [
              'Real-time water usage tracking.',
              'Customizable scheduling for water usage.',
              'Analytics dashboard to monitor consumption trends.',
              'User-friendly interface for easy navigation.',
            ]),
            const SizedBox(height: 16),

            // Benefits Section
            _buildSectionTitle('Benefits'),
            const Text(
              'This app helps users save time, reduce water wastage, and lower utility bills by providing actionable insights '
              'and flexible control over water usage.',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),

            // Usage Instructions Section
            _buildSectionTitle('Usage Instructions'),
            const BulletList(items: [
              'Navigate to the Home tab to view your dashboard.',
              'Use the Schedule section to set water usage timings.',
              'Access the Analytics section to monitor trends.',
              'Modify app settings in the Settings tab.',
            ]),
            const SizedBox(height: 16),

            // Developer/Team Information Section
            _buildSectionTitle('Developer/Team Information'),
            const Text(
              'Developed by: Bafokeng Khoali\n'
              'Contributors: Kesi Motlatla, Hlompho Notsi, and Sello Mohami',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),

            // Contact Information Section
            _buildSectionTitle('Contact Information'),
            const Text(
              'For support or feedback, contact us at:\n'
              'Email: baffykay@20.com\n'
              'Phone: +266 5376 4723',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),

            // Version Information Section
            _buildSectionTitle('Version Information'),
            const Text(
              'Version: 1.0.0',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),

            // Acknowledgments Section
            _buildSectionTitle('Acknowledgments'),
            const Text(
              'Special thanks to Limkokwing University for their valuable feedback and support.',
              style: TextStyle(fontSize: 16, height: 1.5),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  // Reusable method for section titles
  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: Colors.teal,
        ),
      ),
    );
  }
}

// BulletList Widget Implementation
class BulletList extends StatelessWidget {
  final List<String> items;

  const BulletList({super.key, required this.items});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: items
          .map((item) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.circle, size: 8, color: Colors.teal),
                    const SizedBox(width: 8),
                    Expanded(
                        child:
                            Text(item, style: const TextStyle(fontSize: 16))),
                  ],
                ),
              ))
          .toList(),
    );
  }
}
