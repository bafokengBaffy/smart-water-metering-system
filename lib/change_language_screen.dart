import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fluttertoast/fluttertoast.dart';

class ChangeLanguageScreen extends StatelessWidget {
  const ChangeLanguageScreen({super.key});

  // Function to set the selected language
  Future<void> _setLanguage(BuildContext context, String languageCode) async {
    // Save the selected language to SharedPreferences
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('language', languageCode);

    // Show a toast message indicating the language change
    Fluttertoast.showToast(
      msg:
          'Language changed to ${languageCode == "en" ? "English" : "Sesotho"}',
      toastLength: Toast.LENGTH_SHORT,
      gravity: ToastGravity.BOTTOM,
      backgroundColor: Colors.grey[800],
      textColor: Colors.white,
    );

    // Navigate back to the previous screen
    // ignore: use_build_context_synchronously
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Change Language')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            // English Language Option
            ListTile(
              title: const Text('English'),
              onTap:
                  () => _setLanguage(context, 'en'), // Set language to English
            ),
            // Sesotho Language Option
            ListTile(
              title: const Text('Sesotho'),
              onTap:
                  () => _setLanguage(context, 'st'), // Set language to Sesotho
            ),
          ],
        ),
      ),
    );
  }
}
