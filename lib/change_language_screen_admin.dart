import 'package:flutter/material.dart';

class ChangeLanguageScreen extends StatelessWidget {
  const ChangeLanguageScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Change Language'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            ListTile(
              title: const Text('English'),
              onTap: () {
                // Change to English
              },
            ),
            ListTile(
              title: const Text('Sesotho'),
              onTap: () {
                // Change to Sesotho
              },
            ),
          ],
        ),
      ),
    );
  }
}
