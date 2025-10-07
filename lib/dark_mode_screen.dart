import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'main.dart';

class DarkModeScreen extends StatefulWidget {
  const DarkModeScreen({super.key});

  @override
  State<DarkModeScreen> createState() => _DarkModeScreenState();
}

class _DarkModeScreenState extends State<DarkModeScreen> {
  late bool _isDarkMode;

  @override
  void initState() {
    super.initState();
    _isDarkMode = Provider.of<ThemeProvider>(context, listen: false).isDarkMode;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dark Mode'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            SwitchListTile(
              title: const Text('Enable Dark Mode'),
              value: _isDarkMode,
              onChanged: (value) => setState(() => _isDarkMode = value),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Provider.of<ThemeProvider>(context, listen: false)
                    .toggleTheme(_isDarkMode);
                Navigator.pop(context);
              },
              child: const Text('Save Settings'),
            ),
          ],
        ),
      ),
    );
  }
}