import 'package:intl/intl.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MediaItem {
  final String? id;
  final String fileUrl;
  final String type; // 'photo' or 'video'
  final DateTime timestamp;
  final String? description;
  final String? location;

  MediaItem({
    this.id,
    required this.fileUrl,
    required this.type,
    required this.timestamp,
    this.description,
    this.location,
  });

  String get formattedDate =>
      DateFormat('dd MMM yyyy – HH:mm').format(timestamp);

  factory MediaItem.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return MediaItem(
      id: doc.id,
      fileUrl: data['fileUrl'],
      type: data['type'],
      timestamp: (data['timestamp'] as Timestamp).toDate(),
      description: data['description'],
      location: data['location'],
    );
  }

  Map<String, dynamic> toMap() => {
    'fileUrl': fileUrl,
    'type': type,
    'timestamp': Timestamp.fromDate(timestamp),
    'description': description,
    'location': location,
  };
}
