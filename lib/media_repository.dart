import 'dart:io';

import 'package:demo/media_model.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:uuid/uuid.dart';

class MediaRepository {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseStorage _storage = FirebaseStorage.instance;
  final Uuid _uuid = const Uuid();

  Future<void> uploadExistingMedia(File file, String type) async {
    // Upload to Firebase Storage
    final String fileId = _uuid.v4();
    final Reference storageRef = _storage.ref().child('media/$fileId');
    await storageRef.putFile(file);

    // Get download URL
    final String downloadUrl = await storageRef.getDownloadURL();

    // Save metadata to Firestore
    await _firestore.collection('media').add({
      'fileUrl': downloadUrl,
      'type': type,
      'timestamp': DateTime.now(),
    });
  }

  Stream<List<MediaItem>> getMediaStream() {
    return _firestore
        .collection('media')
        .orderBy('timestamp', descending: true)
        .snapshots()
        .map(
          (snapshot) =>
              snapshot.docs.map((doc) => MediaItem.fromFirestore(doc)).toList(),
        );
  }

  Future<void> deleteMedia(String id) async {
    // Delete from Firestore
    await _firestore.collection('media').doc(id).delete();
    // Optional: Delete from Storage (requires storing the Storage path in Firestore)
  }
}
