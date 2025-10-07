// models/user_model.dart
class UserModel {
  final String id;
  final String name;
  final String email;
  final String phone;
  final String address;
  final String role;
  final double waterUsage;
  final DateTime createdAt;
  final DateTime lastActive;

  UserModel({
    required this.id,
    required this.name,
    required this.email,
    required this.phone,
    required this.address,
    required this.role,
    required this.waterUsage,
    required this.createdAt,
    required this.lastActive,
  });

  factory UserModel.fromMap(Map<String, dynamic> map) {
    return UserModel(
      id: map['id'] ?? '',
      name: map['name'] ?? 'Unknown',
      email: map['email'] ?? '',
      phone: map['phone'] ?? '',
      address: map['address'] ?? '',
      role: map['role'] ?? 'user',
      waterUsage: (map['waterUsage'] ?? 0).toDouble(),
      createdAt: map['createdAt']?.toDate() ?? DateTime.now(),
      lastActive: map['lastActive']?.toDate() ?? DateTime.now(),
    );
  }
}
