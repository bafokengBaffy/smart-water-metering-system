import 'package:demo/sign_in_screen.dart';
import 'package:demo/sign_up_screen.dart'; // Import the SignUp screen
import 'package:flutter/material.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Image
          Positioned.fill(
            child: Image.asset(
              'images/water3.jpg', // Make sure this path is correct
              fit: BoxFit.cover,
              errorBuilder: (context, error, stackTrace) {
                return Container(
                  color: Colors.grey,
                  child: const Center(
                    child: Text('Error loading background image'),
                  ),
                );
              },
            ),
          ),
          // Overlay
          Positioned.fill(
            child: Container(
              // ignore: deprecated_member_use
              color: Colors.black.withOpacity(0.5),
            ),
          ),
          // Main Content
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Logo
                Center(
                  child: Image.asset(
                    'images/water.jpg', // Make sure this path is correct
                    width: 150,
                    height: 150,
                    errorBuilder: (context, error, stackTrace) {
                      return Container(); // Remove the alert icon
                    },
                  ),
                ),
                const SizedBox(height: 20),
                const Text(
                  'Welcome to the Smart Water Metering System',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                    fontFamily: 'Roboto',
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  'Track and optimize your water usage effortlessly.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.white70),
                ),
                const SizedBox(height: 40),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildFeatureIcon(Icons.water_drop, 'Track Usage'),
                    const SizedBox(width: 20), // Add space between icons
                    _buildFeatureIcon(Icons.bar_chart, 'Analyze Trends'),
                    const SizedBox(width: 20), // Add space between icons
                    _buildFeatureIcon(Icons.device_hub, 'Manage Devices'),
                  ],
                ),
                const SizedBox(height: 40),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (BuildContext context) => const SignInPage(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 50,
                      vertical: 15,
                    ),
                    backgroundColor: Colors.blueAccent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: const Text('Sign In', style: TextStyle(fontSize: 18)),
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (BuildContext context) => const SignUpPage(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 50,
                      vertical: 15,
                    ),
                    backgroundColor: Colors.greenAccent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: const Text('Sign Up', style: TextStyle(fontSize: 18)),
                ),
                const SizedBox(height: 20),
                TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/about');
                  },
                  child: const Text(
                    'Learn More',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureIcon(IconData icon, String label) {
    return Column(
      children: [
        Icon(icon, size: 40, color: Colors.white),
        const SizedBox(height: 8),
        Text(label, style: const TextStyle(fontSize: 14, color: Colors.white)),
      ],
    );
  }
}
