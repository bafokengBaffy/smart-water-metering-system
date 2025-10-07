import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'sign_in_screen.dart';

class SignUpPage extends StatefulWidget {
  const SignUpPage({super.key});

  @override
  SignUpPageState createState() => SignUpPageState();
}

class SignUpPageState extends State<SignUpPage> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _addressController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController =
      TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _addressController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  /// Validate email format
  bool isValidEmail(String email) {
    return email.contains('@') && email.contains('.');
  }

  /// Validate phone number format
  bool isValidPhone(String phone) {
    return phone.length >= 10 && RegExp(r'^\+?[0-9]+$').hasMatch(phone);
  }

  /// Validate password strength
  bool isValidPassword(String password) {
    return RegExp(
      r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
    ).hasMatch(password);
  }

  /// Main sign-up function
  Future<void> signUp() async {
    try {
      // 1. Create user in Firebase Auth
      UserCredential userCredential = await FirebaseAuth.instance
          .createUserWithEmailAndPassword(
            email: _emailController.text.trim(),
            password: _passwordController.text.trim(),
          );
      if (kDebugMode) {
        print("User created: ${userCredential.user?.uid}");
      }

      final User? user = userCredential.user;
      if (user == null) {
        _showErrorDialog('User registration failed');
        return;
      }

      // 2. Store additional user info in Firestore
      await FirebaseFirestore.instance.collection('users').doc(user.uid).set({
        'name': _nameController.text.trim(),
        'email': _emailController.text.trim(),
        'phone': _phoneController.text.trim(),
        'address': _addressController.text.trim(),
        'role': 'user', // Default role
        'createdAt': DateTime.now(),
        'waterUsage': 0, // Initialize water usage to 0
        'avatar': '👤', // Default avatar
      });

      // 3. Show success dialog
      _showSuccessDialog();
    } on FirebaseAuthException catch (e) {
      // Firebase-specific error
      if (kDebugMode) {
        print("FirebaseAuthException: ${e.code} - ${e.message}");
      }
      _showErrorDialog(e.message ?? 'Sign-up failed');
    } catch (e) {
      // Other exceptions
      if (kDebugMode) {
        print("Exception: $e");
      }
      _showErrorDialog('An error occurred: $e');
    }
  }

  /// Success Dialog
  void _showSuccessDialog() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Success'),
            content: const Text('Your account has been created successfully!'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context); // Close dialog
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => const SignInPage()),
                  );
                },
                child: const Text('OK'),
              ),
            ],
          ),
    );
  }

  /// Error Dialog
  void _showErrorDialog(String message) {
    if (kDebugMode) {
      print("Error: $message");
    } // Debug print
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Error'),
            content: Text(message),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context); // Close dialog
                },
                child: const Text('OK'),
              ),
            ],
          ),
    );
  }

  /// Validate user inputs
  bool _validateInputs() {
    if (_nameController.text.isEmpty ||
        _emailController.text.isEmpty ||
        _phoneController.text.isEmpty ||
        _addressController.text.isEmpty ||
        _passwordController.text.isEmpty ||
        _confirmPasswordController.text.isEmpty) {
      _showErrorDialog('Please fill in all fields');
      return false;
    }
    if (!isValidEmail(_emailController.text)) {
      _showErrorDialog('Invalid email format');
      return false;
    }
    if (!isValidPhone(_phoneController.text)) {
      _showErrorDialog('Invalid phone number');
      return false;
    }
    if (!isValidPassword(_passwordController.text)) {
      _showErrorDialog(
        'Password must contain at least 8 characters, an uppercase letter, a number, and a special character',
      );
      return false;
    }
    if (_passwordController.text != _confirmPasswordController.text) {
      _showErrorDialog('Passwords do not match');
      return false;
    }
    return true;
  }

  /// Build a text field
  Widget _buildTextField({
    required TextEditingController controller,
    required String labelText,
    required IconData icon,
    bool obscureText = false,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      decoration: InputDecoration(
        labelText: labelText,
        prefixIcon: Icon(icon),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8.0)),
      ),
    );
  }

  /// Build method
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sign Up'),
        backgroundColor: Colors.blueAccent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              const Text(
                'Create an Account',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),
              _buildTextField(
                controller: _nameController,
                labelText: 'Full Name',
                icon: Icons.person,
              ),
              const SizedBox(height: 15),
              _buildTextField(
                controller: _emailController,
                labelText: 'Email Address',
                icon: Icons.email,
              ),
              const SizedBox(height: 15),
              _buildTextField(
                controller: _phoneController,
                labelText: 'Phone Number',
                icon: Icons.phone,
              ),
              const SizedBox(height: 15),
              _buildTextField(
                controller: _addressController,
                labelText: 'Address',
                icon: Icons.location_on,
              ),
              const SizedBox(height: 15),
              _buildTextField(
                controller: _passwordController,
                labelText: 'Password',
                icon: Icons.lock,
                obscureText: true,
              ),
              const SizedBox(height: 15),
              _buildTextField(
                controller: _confirmPasswordController,
                labelText: 'Confirm Password',
                icon: Icons.lock,
                obscureText: true,
              ),
              const SizedBox(height: 25),
              ElevatedButton(
                onPressed: () async {
                  if (_validateInputs()) {
                    await signUp();
                  }
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  minimumSize: const Size(double.infinity, 50),
                ),
                child: const Text('Sign Up', style: TextStyle(fontSize: 18)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
