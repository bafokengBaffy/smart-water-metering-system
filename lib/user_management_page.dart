import 'package:demo/analytics_page.dart';
import 'package:demo/billing_management_page.dart';
import 'package:demo/home_screen_admin.dart';
import 'package:demo/meter_management_page.dart';
import 'package:demo/settings_screen_admin.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';

class UserManagementPage extends StatefulWidget {
  final int initialIndex;
  const UserManagementPage({super.key, this.initialIndex = 2});

  @override
  UserManagementPageState createState() => UserManagementPageState();
}

class UserManagementPageState extends State<UserManagementPage> {
  late int _currentIndex;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  int _selectedTab = 0; // 0: Active, 1: Deleted, 2: Insights

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
  }

  void _onTabTapped(int index) {
    if (index == _currentIndex) return;

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => _getPage(index)),
    );
  }

  Widget _getPage(int index) {
    switch (index) {
      case 0:
        return HomeScreenAdmin();
      case 1:
        return AnalyticsPage();
      case 2:
        return UserManagementPage(initialIndex: index);
      case 3:
        return BillingManagementPage();
      case 4:
        return MeterManagementPage();
      case 5:
        return const SettingsScreen();
      default:
        return HomeScreenAdmin();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'User Management',
          style: GoogleFonts.poppins(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => setState(() {}),
          ),
        ],
      ),
      body: Column(
        children: [
          _buildSegmentedControl(),
          Expanded(
            child: IndexedStack(
              index: _selectedTab,
              children: [
                ActiveUsersTab(firestore: _firestore, auth: _auth),
                DeletedUsersTab(firestore: _firestore),
                UserInsightsTab(firestore: _firestore),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildSegmentedControl() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            _buildSegmentedButton('Active', 0),
            _buildSegmentedButton('Deleted', 1),
            _buildSegmentedButton('Insights', 2),
          ],
        ),
      ),
    );
  }

  Widget _buildSegmentedButton(String text, int index) {
    final isSelected = _selectedTab == index;
    return Expanded(
      child: AnimatedContainer(
        duration: 300.ms,
        margin: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          color: isSelected ? Colors.blueAccent : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: TextButton(
          onPressed: () => setState(() => _selectedTab = index),
          child: Text(
            text,
            style: GoogleFonts.poppins(
              color: isSelected ? Colors.white : Colors.black87,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomNav() {
    return BottomNavigationBar(
      currentIndex: _currentIndex,
      onTap: _onTabTapped,
      selectedItemColor: Colors.blueAccent,
      unselectedItemColor: Colors.grey,
      type: BottomNavigationBarType.fixed,
      items: const [
        BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
        BottomNavigationBarItem(
          icon: Icon(Icons.analytics),
          label: 'Analytics',
        ),
        BottomNavigationBarItem(icon: Icon(Icons.people), label: 'Users'),
        BottomNavigationBarItem(icon: Icon(Icons.payment), label: 'Billing'),
        BottomNavigationBarItem(icon: Icon(Icons.speed), label: 'Meters'),
        BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Settings'),
      ],
    );
  }
}

class ActiveUsersTab extends StatefulWidget {
  final FirebaseFirestore firestore;
  final FirebaseAuth auth;

  const ActiveUsersTab({
    super.key,
    required this.firestore,
    required this.auth,
  });

  @override
  State<ActiveUsersTab> createState() => _ActiveUsersTabState();
}

class _ActiveUsersTabState extends State<ActiveUsersTab> {
  String? _selectedUserId;
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _deleteUser(String userId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder:
          (context) => AlertDialog(
        title: const Text('Confirm Delete'),
        content: const Text('Are you sure you want to delete this user?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text(
              'Delete',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      try {
        // Move to deleted_users collection first
        final userDoc =
        await widget.firestore.collection('users').doc(userId).get();
        await widget.firestore.collection('deleted_users').doc(userId).set({
          ...userDoc.data() as Map<String, dynamic>,
          'deletedAt': FieldValue.serverTimestamp(),
        });

        // Then delete from active users
        await widget.firestore.collection('users').doc(userId).delete();

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('User moved to deleted users'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error: ${e.toString()}'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: TextField(
            controller: _searchController,
            decoration: InputDecoration(
              hintText: 'Search users...',
              prefixIcon: const Icon(Icons.search),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            onChanged: (value) => setState(() {}),
          ),
        ),
        Expanded(
          child: StreamBuilder<QuerySnapshot>(
            stream: widget.firestore.collection('users').snapshots(),
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(child: CircularProgressIndicator());
              }
              if (snapshot.hasError) {
                return Center(child: Text('Error: ${snapshot.error}'));
              }
              if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                return const Center(child: Text('No active users found'));
              }

              // Filter based on search
              var users =
              snapshot.data!.docs.where((doc) {
                final user = doc.data() as Map<String, dynamic>;
                final searchTerm = _searchController.text.toLowerCase();
                return user['name'].toString().toLowerCase().contains(
                  searchTerm,
                ) ||
                    user['email'].toString().toLowerCase().contains(
                      searchTerm,
                    );
              }).toList();

              return ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: users.length,
                itemBuilder: (context, index) {
                  final doc = users[index];
                  final user = doc.data() as Map<String, dynamic>;
                  return _buildUserCard(user, doc.id, index);
                },
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildUserCard(Map<String, dynamic> user, String userId, int index) {
    final isExpanded = _selectedUserId == userId;
    final isOnline = user['isOnline'] as bool? ?? false;
    final lastActive = user['lastActive'] as Timestamp?;

    return GestureDetector(
      onTap: () => setState(() => _selectedUserId = isExpanded ? null : userId),
      child: AnimatedContainer(
        duration: 300.ms,
        margin: const EdgeInsets.only(bottom: 12),
        curve: Curves.easeInOut,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha((0.05 * 255).toInt()),
              blurRadius: 6,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          children: [
            _buildCardHeader(user, userId, isOnline),
            if (isExpanded) _buildExpandedContent(user, isOnline, lastActive),
          ],
        ),
      ),
    ).animate().fadeIn(delay: (100 * index).ms);
  }

  Widget _buildCardHeader(
      Map<String, dynamic> user,
      String userId,
      bool isOnline,
      ) {
    return ListTile(
      leading: Stack(
        children: [
          CircleAvatar(
            backgroundColor: Colors.blue[100],
            child: Text(
              user['name'].toString().substring(0, 1).toUpperCase(),
              style: const TextStyle(color: Colors.blue),
            ),
          ),
          Positioned(
            right: 0,
            bottom: 0,
            child: Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: isOnline ? Colors.green : Colors.grey,
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 2),
              ),
            ),
          ),
        ],
      ),
      title: Text(
        user['name']?.toString() ?? 'Guest User',
        style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        user['email']?.toString() ?? 'No email provided',
        style: GoogleFonts.poppins(color: Colors.grey[600]),
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            icon: const Icon(Icons.delete, color: Colors.red),
            onPressed: () => _deleteUser(userId),
          ),
          Icon(
            _selectedUserId == userId ? Icons.expand_less : Icons.expand_more,
            color: Colors.blueGrey,
          ),
        ],
      ),
    );
  }

  Widget _buildExpandedContent(
      Map<String, dynamic> user,
      bool isOnline,
      Timestamp? lastActive,
      ) {
    return Padding(
      padding: const EdgeInsets.all(16).copyWith(top: 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Divider(),
          _buildDetailItem(
            Icons.phone,
            'Phone',
            user['phone']?.toString() ?? 'N/A',
          ),
          _buildDetailItem(
            Icons.location_on,
            'Address',
            user['address']?.toString() ?? 'N/A',
          ),
          _buildDetailItem(
            Icons.date_range,
            'Registered',
            user['createdAt'] != null
                ? DateFormat(
              'dd MMM yyyy',
            ).format((user['createdAt'] as Timestamp).toDate())
                : 'N/A',
          ),
          _buildDetailItem(
            Icons.water_drop,
            'Water Usage',
            '${user['usage']?.toString() ?? '0'}L',
          ),
          const SizedBox(height: 12),
          _buildStatusIndicator(isOnline, lastActive),
        ],
      ),
    );
  }

  Widget _buildDetailItem(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, size: 20, color: Colors.blueGrey),
          const SizedBox(width: 12),
          Text(
            '$label: ',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w500),
          ),
          Text(value, style: GoogleFonts.poppins(color: Colors.grey[600])),
        ],
      ),
    );
  }

  Widget _buildStatusIndicator(bool isOnline, Timestamp? lastActive) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: isOnline ? Colors.green[50] : Colors.red[50],
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            children: [
              Icon(
                Icons.circle,
                color: isOnline ? Colors.green : Colors.red,
                size: 12,
              ),
              const SizedBox(width: 8),
              Text(
                isOnline ? 'Online Now' : 'Offline',
                style: GoogleFonts.poppins(
                  color: isOnline ? Colors.green : Colors.red,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        if (!isOnline && lastActive != null)
          Text(
            'Last active: ${DateFormat('dd MMM, hh:mm a').format(lastActive.toDate())}',
            style: GoogleFonts.poppins(color: Colors.grey),
          ),
      ],
    );
  }
}

class DeletedUsersTab extends StatelessWidget {
  final FirebaseFirestore firestore;

  const DeletedUsersTab({super.key, required this.firestore});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream:
      firestore
          .collection('deleted_users')
          .orderBy('deletedAt', descending: true)
          .snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return Center(child: Text('Error: ${snapshot.error}'));
        }
        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
          return const Center(child: Text('No deleted users found'));
        }

        return ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: snapshot.data!.docs.length,
          itemBuilder: (context, index) {
            final doc = snapshot.data!.docs[index];
            final user = doc.data() as Map<String, dynamic>;
            return _buildDeletedUserCard(user, index);
          },
        );
      },
    );
  }

  Widget _buildDeletedUserCard(Map<String, dynamic> user, int index) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      color: Colors.grey[100],
      child: ListTile(
        leading: const CircleAvatar(
          backgroundColor: Colors.grey,
          child: Icon(Icons.person_off, color: Colors.white),
        ),
        title: Text(
          user['name']?.toString() ?? 'Deleted User',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w500),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(user['email']?.toString() ?? 'No email'),
            const SizedBox(height: 4),
            Text(
              'Deleted on: ${user['deletedAt'] != null ? DateFormat('dd MMM yyyy, hh:mm a').format((user['deletedAt'] as Timestamp).toDate()) : 'Unknown'}',
              style: GoogleFonts.poppins(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: (100 * index).ms);
  }
}

class UserInsightsTab extends StatelessWidget {
  final FirebaseFirestore firestore;

  const UserInsightsTab({super.key, required this.firestore});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream: firestore.collection('users').snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        int activeCount = 0;
        int onlineCount = 0;
        final now = DateTime.now();
        final activeUsers = <Map<String, dynamic>>[];

        if (snapshot.hasData) {
          for (final doc in snapshot.data!.docs) {
            final user = doc.data() as Map<String, dynamic>;
            activeCount++;
            if (user['isOnline'] == true) onlineCount++;
            if (user['lastActive'] != null) {
              final lastActive = (user['lastActive'] as Timestamp).toDate();
              if (now.difference(lastActive).inHours < 24) {
                activeUsers.add(user);
              }
            }
          }
        }

        return SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              _buildInsightCard(
                title: 'User Statistics',
                children: [
                  _buildStatItem(
                    'Total Active Users',
                    activeCount.toString(),
                    Icons.people,
                  ),
                  _buildStatItem(
                    'Currently Online',
                    '$onlineCount',
                    Icons.online_prediction,
                  ),
                  _buildStatItem(
                    'Active Today',
                    '${activeUsers.length}',
                    Icons.today,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildInsightCard(
                title: 'Recent Activity',
                children: [
                  if (activeUsers.isEmpty)
                    const Text('No recent activity')
                  else
                    ...activeUsers
                        .take(5)
                        .map(
                          (user) => ListTile(
                        leading: CircleAvatar(
                          child: Text(
                            user['name'].toString().substring(0, 1),
                          ),
                        ),
                        title: Text(user['name']),
                        subtitle: Text(
                          'Last active: ${DateFormat('hh:mm a').format((user['lastActive'] as Timestamp).toDate())}',
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildInsightCard({
    required String title,
    required List<Widget> children,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: GoogleFonts.poppins(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Divider(),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, color: Colors.blue),
          const SizedBox(width: 12),
          Text(label, style: GoogleFonts.poppins()),
          const Spacer(),
          Text(value, style: GoogleFonts.poppins(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
