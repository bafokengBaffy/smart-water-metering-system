// Core Flutter/Dart
import 'package:demo/sensor_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// State Management
import 'package:provider/provider.dart';

// Firebase
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firebase_options.dart';

// Screens
import 'package:demo/welcome_screen.dart';
import 'package:demo/sign_up_screen.dart';
import 'package:demo/sign_in_screen.dart';
import 'package:demo/home_screen_admin.dart';
import 'package:demo/about_screen.dart';
import 'package:demo/dark_mode_screen.dart';
import 'package:demo/analysis_screen.dart';
import 'package:demo/profile_screen.dart';
import 'package:demo/payment_screen.dart';
import 'package:demo/home_screen.dart' as home;
import 'package:demo/settings_screen.dart' as settings;
import 'package:demo/dashboard_page.dart';
import 'package:demo/meter_management_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await dotenv.load(fileName: ".env");

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
    if (kDebugMode) {
      print('Firebase initialized successfully');
    }
  } catch (e) {
    if (kDebugMode) {
      print('Firebase initialization error: $e');
    }
  }

  final prefs = await SharedPreferences.getInstance();
  final isDarkMode = prefs.getBool('isDarkMode') ?? false;

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider<ThemeProvider>(
          create:
              (context) =>
                  ThemeProvider(isDarkMode ? ThemeMode.dark : ThemeMode.light),
        ),
        Provider<SensorService>(
          create: (context) => SensorService()..initialize(),
          lazy: false,
        ),
      ],
      child: const MyApp(),
    ),
  );
}

class ThemeProvider extends ChangeNotifier {
  ThemeMode _themeMode;
  ThemeProvider(this._themeMode);

  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _themeMode == ThemeMode.dark;

  Future<void> toggleTheme(bool isDarkMode) async {
    _themeMode = isDarkMode ? ThemeMode.dark : ThemeMode.light;
    notifyListeners();

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isDarkMode', isDarkMode);
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeProvider>(
      builder: (context, themeProvider, child) {
        // Set the status bar style based on the current theme
        SystemChrome.setSystemUIOverlayStyle(
          SystemUiOverlayStyle(
            statusBarColor: Colors.transparent,
            statusBarIconBrightness:
                themeProvider.isDarkMode ? Brightness.light : Brightness.dark,
          ),
        );

        return MaterialApp(
          title: 'Smart Water Metering System',
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.light,
            ),
            useMaterial3: true,
          ),
          darkTheme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.dark,
            ),
            useMaterial3: true,
          ),
          themeMode: themeProvider.themeMode,
          debugShowCheckedModeBanner: false,
          home: const AuthWrapper(),
          routes: {
            '/welcome': (context) => const WelcomeScreen(),
            '/meter': (context) => MeterManagementPage(),
            '/home': (context) => const home.HomeScreen(),
            '/dashboard': (context) => const DashboardPage(),
            '/sign': (context) => const SignUpPage(),
            '/signin': (context) => const SignInPage(),
            '/admin_home': (context) => const HomeScreenAdmin(),
            '/about': (context) => const AboutScreen(),
            '/settings': (context) => const settings.SettingsScreen(),
            '/darkmode': (context) => const DarkModeScreen(),
            '/analysis': (context) => const AnalysisScreen(),
            '/profile': (context) => const ProfileScreen(),
            '/payment': (context) => const PaymentScreen(),
          },
        );
      },
    );
  }
}

class AuthWrapper extends StatelessWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
        return snapshot.hasData
            ? const home.HomeScreen()
            : const WelcomeScreen();
      },
    );
  }
}
