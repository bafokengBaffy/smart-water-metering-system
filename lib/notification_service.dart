import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:fluttertoast/fluttertoast.dart';

class NotificationService {
  static final FirebaseMessaging messaging = FirebaseMessaging.instance;

  static Future<void> handlePushNotification(bool isEnabled) async {
    if (isEnabled) {
      NotificationSettings settings = await messaging.requestPermission();
      if (settings.authorizationStatus == AuthorizationStatus.authorized) {
        Fluttertoast.showToast(msg: "Push notifications enabled.");
      } else {
        Fluttertoast.showToast(
            msg: "Push notifications disabled in system settings.");
      }
    } else {
      Fluttertoast.showToast(msg: "Push notifications disabled.");
    }
  }
}
