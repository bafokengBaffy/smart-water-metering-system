import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FeedbackAndSupportScreen extends StatefulWidget {
  const FeedbackAndSupportScreen({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _FeedbackAndSupportScreenState createState() =>
      _FeedbackAndSupportScreenState();
}

class _FeedbackAndSupportScreenState extends State<FeedbackAndSupportScreen> {
  final TextEditingController _feedbackController = TextEditingController();
  final TextEditingController _issueController = TextEditingController();
  bool isFeedbackSubmitted = false;

  @override
  void dispose() {
    _feedbackController.dispose();
    _issueController.dispose();
    super.dispose();
  }

  Future<void> saveFeedback(String feedback) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('userFeedback', feedback);
    Fluttertoast.showToast(msg: 'Feedback saved successfully');
  }

  void submitFeedback() async {
    if (_feedbackController.text.isEmpty) {
      Fluttertoast.showToast(msg: 'Please provide your feedback');
    } else {
      await saveFeedback(_feedbackController.text);
      setState(() {
        isFeedbackSubmitted = true;
      });
    }
  }

  void submitSupportRequest() {
    if (_issueController.text.isEmpty) {
      Fluttertoast.showToast(msg: 'Please describe your issue');
    } else {
      Fluttertoast.showToast(msg: 'Support request submitted');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Feedback and Support'),
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            const Text(
              'We value your feedback and are here to assist you with any issues.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 24),
            const Text(
              'Feedback',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _feedbackController,
              decoration: const InputDecoration(
                labelText: 'Enter your feedback',
                border: OutlineInputBorder(),
                hintText: 'Let us know how we can improve',
              ),
              maxLines: 4,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: submitFeedback,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
              child: const Text('Submit Feedback'),
            ),
            if (isFeedbackSubmitted)
              const Padding(
                padding: EdgeInsets.only(top: 8.0),
                child: Text(
                  'Thank you for your feedback!',
                  style: TextStyle(color: Colors.green),
                ),
              ),
            const SizedBox(height: 24),
            const Text(
              'Support Request',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _issueController,
              decoration: const InputDecoration(
                labelText: 'Describe your issue',
                border: OutlineInputBorder(),
                hintText: 'Provide details about the issue you are facing',
              ),
              maxLines: 4,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: submitSupportRequest,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
              child: const Text('Submit Support Request'),
            ),
          ],
        ),
      ),
    );
  }
}
