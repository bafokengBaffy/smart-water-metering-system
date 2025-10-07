import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'analysis_screen.dart';
import 'payment_screen.dart';
import 'settings_screen.dart' as settings;

class NavigationProvider with ChangeNotifier {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    HomeScreen(),
    AnalysisScreen(),
    PaymentScreen(),
    settings.SettingsScreen(),
  ];

  int get currentIndex => _currentIndex;
  Widget get currentScreen => _screens[_currentIndex];

  void updateIndex(int index) {
    if (index == _currentIndex) return;
    _currentIndex = index;
    notifyListeners();
  }
}
