import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:demo/analysis_screen.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'bill.dart';
import 'change_language_screen.dart';
import 'dark_mode_screen.dart';
import 'feedback_and_support_screen.dart';
import 'home_screen.dart';
import 'notification_service.dart';
import 'payment_screen.dart';
import 'sign_in_screen.dart'; // Ensure this import exists for navigation
import 'usage_history_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  SettingsScreenState createState() => SettingsScreenState();
}

class SettingsScreenState extends State<SettingsScreen> {
  bool isPushNotificationsEnabled = true;
  bool isEmailNotificationsEnabled = false;
  bool isUsageAlertsEnabled = false;
  bool isLeakAlertsEnabled = false;
  int _selectedIndex = 3; // Corrected index for Settings screen
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  @override
  void initState() {
    super.initState();
    loadSettings();
  }

  Future<void> loadSettings() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      isPushNotificationsEnabled = prefs.getBool('pushNotifications') ?? true;
      isEmailNotificationsEnabled =
          prefs.getBool('emailNotifications') ?? false;
      isUsageAlertsEnabled = prefs.getBool('usageAlerts') ?? false;
      isLeakAlertsEnabled = prefs.getBool('leakAlerts') ?? false;
    });
  }

  Future<void> saveSetting(String key, bool value) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
    Fluttertoast.showToast(msg: 'Setting saved');
  }

  void changeLanguage() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const ChangeLanguageScreen()),
    );
  }

  void toggleDarkMode() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const DarkModeScreen()),
    );
  }

  Future<void> logout() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      // Update Firestore status before signing out
      await _firestore.collection('users').doc(user.uid).update({
        'isOnline': false,
        'lastActive': FieldValue.serverTimestamp(),
      });
    }

    await FirebaseAuth.instance.signOut();

    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const SignInPage()),
      );
    }
  }

  void _onNavItemTapped(int index) {
    if (index == _selectedIndex) return;

    setState(() {
      _selectedIndex = index;
    });

    switch (index) {
      case 0:
        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (_, __, ___) => const HomeScreen(),
            transitionDuration: Duration.zero,
          ),
        );
        break;
      case 1:
        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (_, __, ___) => const AnalysisScreen(),
            transitionDuration: Duration.zero,
          ),
        );
        break;
      case 2:
        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (_, __, ___) => const PaymentScreen(),
            transitionDuration: Duration.zero,
          ),
        );
        break;
      case 3:
        // Already on settings screen
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 80.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Settings',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            ExpansionTile(
              title: const Text(
                'Notification Settings',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              children: [
                SwitchListTile(
                  title: const Text('Enable Push Notifications'),
                  value: isPushNotificationsEnabled,
                  onChanged: (value) async {
                    setState(() => isPushNotificationsEnabled = value);
                    await saveSetting('pushNotifications', value);
                    await NotificationService.handlePushNotification(value);
                  },
                ),
                SwitchListTile(
                  title: const Text('Email Notifications'),
                  value: isEmailNotificationsEnabled,
                  onChanged: (value) {
                    setState(() => isEmailNotificationsEnabled = value);
                    saveSetting('emailNotifications', value);
                  },
                ),
              ],
            ),
            const Divider(),
            ExpansionTile(
              title: const Text(
                'App Settings',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              children: [
                ListTile(
                  leading: const Icon(Icons.language),
                  title: const Text('Language'),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: changeLanguage,
                ),
                ListTile(
                  leading: const Icon(Icons.dark_mode),
                  title: const Text('Dark Mode'),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: toggleDarkMode,
                ),
              ],
            ),
            const Divider(),
            ExpansionTile(
              title: const Text(
                'Water Usage Tracking',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              children: [
                ListTile(
                  leading: const Icon(Icons.water_damage),
                  title: const Text('View Usage History'),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const UsageHistoryScreen(),
                      ),
                    );
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.show_chart),
                  title: const Text('Usage Alerts'),
                  trailing: Switch(
                    value: isUsageAlertsEnabled,
                    onChanged: (value) {
                      setState(() => isUsageAlertsEnabled = value);
                      saveSetting('usageAlerts', value);
                    },
                  ),
                ),
              ],
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.receipt_long),
              title: const Text('Billing Information'),
              trailing: const Icon(Icons.arrow_forward),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const BillScreen()),
                );
              },
            ),
            const Divider(),
            ExpansionTile(
              title: const Text(
                'Leak Detection',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              children: [
                ListTile(
                  leading: const Icon(Icons.warning),
                  title: const Text('Enable Leak Alerts'),
                  trailing: Switch(
                    value: isLeakAlertsEnabled,
                    onChanged: (value) {
                      setState(() => isLeakAlertsEnabled = value);
                      saveSetting('leakAlerts', value);
                    },
                  ),
                ),
              ],
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.av_timer),
              title: const Text('Automatic Meter Reading'),
              trailing: const Icon(Icons.arrow_forward),
              onTap: () {
                // Implement automatic meter reading
              },
            ),
            const Divider(),
            ExpansionTile(
              title: const Text(
                'Usage Goals',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              children: [
                ListTile(
                  leading: const Icon(Icons.track_changes),
                  title: const Text('Set Usage Goals'),
                  trailing: const Icon(Icons.arrow_forward),
                  onTap: () {
                    // Implement usage goals
                  },
                ),
              ],
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.info_outline),
              title: const Text('About'),
              trailing: const Icon(Icons.arrow_forward),
              onTap: () {
                showDialog(
                  context: context,
                  builder:
                      (context) => AlertDialog(
                        title: const Text('About'),
                        content: const Text(
                          'App Version 1.0.0\nDeveloped by [Your Team]',
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('Close'),
                          ),
                        ],
                      ),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.feedback_outlined),
              title: const Text('Feedback & Support'),
              trailing: const Icon(Icons.arrow_forward),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const FeedbackAndSupportScreen(),
                  ),
                );
              },
            ),
            const Divider(),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                showDialog(
                  context: context,
                  builder:
                      (context) => AlertDialog(
                        title: const Text('Logout'),
                        content: const Text(
                          'Are you sure you want to log out?',
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('Cancel'),
                          ),
                          TextButton(
                            onPressed: () async {
                              Navigator.pop(context);
                              await logout();
                            },
                            child: const Text('Logout'),
                          ),
                        ],
                      ),
                );
              },
              style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
              child: const Text('Logout'),
            ),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildBottomNav() {
    return SafeArea(
      child: Container(
        height: 70,
        decoration: const BoxDecoration(
          border: Border(top: BorderSide(color: Colors.grey, width: 0.3)),
        ),
        child: BottomAppBar(
          padding: EdgeInsets.zero,
          elevation: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavButton(
                icon: Icons.home,
                label: "Home",
                isActive: _selectedIndex == 0,
                onTap: () => _onNavItemTapped(0),
              ),
              _NavButton(
                icon: Icons.bar_chart,
                label: "Stats",
                isActive: _selectedIndex == 1,
                onTap: () => _onNavItemTapped(1),
              ),
              _NavButton(
                icon: Icons.payment,
                label: "Pay",
                isActive: _selectedIndex == 2,
                onTap: () => _onNavItemTapped(2),
              ),
              _NavButton(
                icon: Icons.settings,
                label: "Settings",
                isActive: _selectedIndex == 3,
                onTap: () => _onNavItemTapped(3),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NavButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final VoidCallback onTap;

  const _NavButton({
    required this.icon,
    required this.label,
    this.isActive = false,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color:
              isActive
                  // ignore: deprecated_member_use
                  ? Colors.blueAccent.withOpacity(0.2)
                  : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isActive ? Colors.blueAccent : Colors.grey,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: isActive ? Colors.blueAccent : Colors.grey,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
