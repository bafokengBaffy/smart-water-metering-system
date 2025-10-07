import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

class UploadService {
  static Future<void> uploadFile(File file) async {
    try {
      final uri = Uri.parse(
        'http://localhost:3000/upload',
      ); // Replace with your backend URL
      final fileBytes = await file.readAsBytes();
      final fileBase64 = base64Encode(fileBytes);

      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'fileName': file.path.split('/').last,
          'fileData': fileBase64,
        }),
      );

      if (response.statusCode == 200) {
        final fileUrl = jsonDecode(response.body)['fileUrl'];
        if (kDebugMode) {
          print('File uploaded successfully: $fileUrl');
        }
      } else {
        if (kDebugMode) {
          print('Failed to upload file: ${response.statusCode}');
        }
      }
    } catch (error) {
      if (kDebugMode) {
        print('Error uploading file: $error');
      }
    }
  }
}
