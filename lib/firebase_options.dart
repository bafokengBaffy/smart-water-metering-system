import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        return windows;
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for Linux',
        );
      default:
        throw UnsupportedError(
          'Unsupported platform',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyBNmYZF7Bp98p2pVIiN4k4hEi0udnb9a8Q',
    appId: '1:201555512766:web:13600b40dca527aecabdef',
    messagingSenderId: '201555512766',
    projectId: 'smart-water-metering-sys',
    authDomain: 'smart-water-metering-sys.firebaseapp.com',
    storageBucket: 'smart-water-metering-sys.firebasestorage.app',
    measurementId: 'G-LZKHHKK1TF',
    databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyDKrjZl2esqmYnw6DNBon2DtfIU9JZLvHQ',
    appId: '1:201555512766:android:8c52f505d3ae0355cabdef',
    messagingSenderId: '201555512766',
    projectId: 'smart-water-metering-sys',
    storageBucket: 'smart-water-metering-sys.firebasestorage.app',
    databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyBciSax5oKFx9NDjo9FWjI_yTPksTc8u78',
    appId: '1:201555512766:ios:91c66fc2ac888f6ecabdef',
    messagingSenderId: '201555512766',
    projectId: 'smart-water-metering-sys',
    storageBucket: 'smart-water-metering-sys.firebasestorage.app',
    iosBundleId: 'com.example.smartWaterMeteringSystem',
    databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyBciSax5oKFx9NDjo9FWjI_yTPksTc8u78',
    appId: '1:201555512766:ios:91c66fc2ac888f6ecabdef',
    messagingSenderId: '201555512766',
    projectId: 'smart-water-metering-sys',
    storageBucket: 'smart-water-metering-sys.firebasestorage.app',
    iosBundleId: 'com.example.smartWaterMeteringSystem',
    databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
  );

  static const FirebaseOptions windows = FirebaseOptions(
    apiKey: 'AIzaSyBNmYZF7Bp98p2pVIiN4k4hEi0udnb9a8Q',
    appId: '1:201555512766:web:31d2f60612105140cabdef',
    messagingSenderId: '201555512766',
    projectId: 'smart-water-metering-sys',
    authDomain: 'smart-water-metering-sys.firebaseapp.com',
    storageBucket: 'smart-water-metering-sys.firebasestorage.app',
    measurementId: 'G-VQ5K8GBKH2',
    databaseURL: 'https://smart-water-metering-sys-default-rtdb.firebaseio.com/',
  );
}