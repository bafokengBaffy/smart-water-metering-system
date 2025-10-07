// ignore_for_file: unused_field

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:demo/settings_screen.dart';
import 'package:demo/user_management_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import 'analytics_page.dart';
import 'home_screen_admin.dart' show HomeScreenAdmin;
import 'meter_management_page.dart';

class BillingManagementPage extends StatefulWidget {
  final int initialIndex;
  const BillingManagementPage({super.key, this.initialIndex = 3});

  @override
  BillingManagementPageState createState() => BillingManagementPageState();
}

class BillingManagementPageState extends State<BillingManagementPage> {
  late int _currentIndex;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  int _selectedTab = 0; // 0: Active Bills, 1: Paid Bills, 2: Insights
  String? _selectedBillId;

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
        return const HomeScreenAdmin();
      case 1:
        return const AnalyticsPage();
      case 2:
        return const UserManagementPage();
      case 3:
        return BillingManagementPage(initialIndex: index);
      case 4:
        return MeterManagementPage();
      case 5:
        return const SettingsScreen();
      default:
        return const HomeScreenAdmin();
    }
  }

  Future<void> _updateBillStatus(String billId, String newStatus) async {
    await _firestore.collection('bills').doc(billId).update({
      'status': newStatus,
    });
    setState(() => _selectedBillId = null);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Billing Management',
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
                ActiveBillsTab(firestore: _firestore),
                PaidBillsTab(firestore: _firestore),
                BillingInsightsTab(firestore: _firestore),
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
            _buildSegmentedButton('Paid', 1),
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

class ActiveBillsTab extends StatefulWidget {
  final FirebaseFirestore firestore;

  const ActiveBillsTab({super.key, required this.firestore});

  @override
  State<ActiveBillsTab> createState() => _ActiveBillsTabState();
}

class _ActiveBillsTabState extends State<ActiveBillsTab> {
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
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
              hintText: 'Search bills...',
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
            stream:
                widget.firestore
                    .collection('bills')
                    .where('status', isEqualTo: 'Pending')
                    .snapshots(),
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(child: CircularProgressIndicator());
              }
              if (snapshot.hasError) {
                return Center(child: Text('Error: ${snapshot.error}'));
              }
              if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                return const Center(child: Text('No pending bills found'));
              }

              var bills =
                  snapshot.data!.docs.where((doc) {
                    final bill = doc.data() as Map<String, dynamic>;
                    final searchTerm = _searchController.text.toLowerCase();
                    return bill['customerName']
                            .toString()
                            .toLowerCase()
                            .contains(searchTerm) ||
                        bill['amount'].toString().toLowerCase().contains(
                          searchTerm,
                        );
                  }).toList();

              return ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: bills.length,
                itemBuilder: (context, index) {
                  final doc = bills[index];
                  final bill = doc.data() as Map<String, dynamic>;
                  return _buildBillCard(bill, doc.id, index);
                },
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildBillCard(Map<String, dynamic> bill, String billId, int index) {
    final isExpanded =
        (context
                .findAncestorStateOfType<BillingManagementPageState>()
                ?._selectedBillId ==
            billId);
    final dueDate =
        bill['dueDate'] != null
            ? (bill['dueDate'] as Timestamp).toDate()
            : null;

    return GestureDetector(
      onTap:
          () => context
              .findAncestorStateOfType<BillingManagementPageState>()
              ?.setState(
                () =>
                    context
                        .findAncestorStateOfType<BillingManagementPageState>()
                        ?._selectedBillId = isExpanded ? null : billId,
              ),
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
            _buildCardHeader(bill, billId),
            if (isExpanded) _buildExpandedContent(bill, dueDate),
          ],
        ),
      ),
    ).animate().fadeIn(delay: (100 * index).ms);
  }

  Widget _buildCardHeader(Map<String, dynamic> bill, String billId) {
    return ListTile(
      leading: CircleAvatar(
        backgroundColor: Colors.orange[100],
        child: const Icon(Icons.receipt, color: Colors.orange),
      ),
      title: Text(
        bill['customerName']?.toString() ?? 'Unknown Customer',
        style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        'Amount: M${bill['amount']?.toStringAsFixed(2) ?? '0.00'}',
        style: GoogleFonts.poppins(color: Colors.grey[600]),
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            icon: const Icon(Icons.check_circle, color: Colors.green),
            onPressed:
                () => context
                    .findAncestorStateOfType<BillingManagementPageState>()
                    ?._updateBillStatus(billId, 'Paid'),
          ),
          Icon(
            context
                        .findAncestorStateOfType<BillingManagementPageState>()
                        ?._selectedBillId ==
                    billId
                ? Icons.expand_less
                : Icons.expand_more,
            color: Colors.blueGrey,
          ),
        ],
      ),
    );
  }

  Widget _buildExpandedContent(Map<String, dynamic> bill, DateTime? dueDate) {
    final usageDifference =
        (bill['usage'] ?? 0) - (bill['predictedUsage'] ?? 0);

    return Padding(
      padding: const EdgeInsets.all(16).copyWith(top: 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Divider(),
          _buildDetailItem(
            Icons.person,
            'Customer',
            bill['customerName']?.toString() ?? 'N/A',
          ),
          _buildDetailItem(
            Icons.attach_money,
            'Amount',
            'M${bill['amount']?.toStringAsFixed(2) ?? '0.00'}',
          ),
          _buildDetailItem(
            Icons.water_drop,
            'Water Usage',
            '${bill['usage']?.toStringAsFixed(1) ?? '0'} m³',
          ),
          _buildDetailItem(
            Icons.trending_up,
            'Predicted Usage',
            '${bill['predictedUsage']?.toStringAsFixed(1) ?? '0'} m³',
          ),
          Row(
            children: [
              const Icon(Icons.compare, size: 20, color: Colors.blueGrey),
              const SizedBox(width: 12),
              const Text(
                'Difference: ',
                style: TextStyle(fontWeight: FontWeight.w500),
              ),
              Icon(
                usageDifference > 0 ? Icons.arrow_upward : Icons.arrow_downward,
                color: usageDifference > 0 ? Colors.red : Colors.green,
                size: 16,
              ),
              Text(
                '${usageDifference.abs().toStringAsFixed(1)} m³',
                style: TextStyle(
                  color: usageDifference > 0 ? Colors.red : Colors.green,
                ),
              ),
            ],
          ),
          if (dueDate != null) ...[
            const SizedBox(height: 8),
            _buildDetailItem(
              Icons.calendar_today,
              'Due Date',
              DateFormat('dd MMM yyyy').format(dueDate),
            ),
          ],
          const SizedBox(height: 12),
          Align(
            alignment: Alignment.centerRight,
            child: ElevatedButton(
              onPressed:
                  () => context
                      .findAncestorStateOfType<BillingManagementPageState>()
                      ?._updateBillStatus(bill['id'], 'Paid'),
              child: const Text('Mark as Paid'),
            ),
          ),
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
}

class PaidBillsTab extends StatelessWidget {
  final FirebaseFirestore firestore;

  const PaidBillsTab({super.key, required this.firestore});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream:
          firestore
              .collection('bills')
              .where('status', isEqualTo: 'Paid')
              .orderBy('paidDate', descending: true)
              .snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return Center(child: Text('Error: ${snapshot.error}'));
        }
        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
          return const Center(child: Text('No paid bills found'));
        }

        return ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: snapshot.data!.docs.length,
          itemBuilder: (context, index) {
            final doc = snapshot.data!.docs[index];
            final bill = doc.data() as Map<String, dynamic>;
            return _buildPaidBillCard(bill, index);
          },
        );
      },
    );
  }

  Widget _buildPaidBillCard(Map<String, dynamic> bill, int index) {
    final paidDate =
        bill['paidDate'] != null
            ? DateTime.fromMillisecondsSinceEpoch(bill['paidDate'])
            : null;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      color: Colors.green[50],
      child: ListTile(
        leading: const CircleAvatar(
          backgroundColor: Colors.green,
          child: Icon(Icons.check, color: Colors.white),
        ),
        title: Text(
          bill['customerName']?.toString() ?? 'Unknown Customer',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w500),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Amount: M${bill['amount']?.toStringAsFixed(2) ?? '0.00'}'),
            const SizedBox(height: 4),
            if (paidDate != null)
              Text(
                'Paid on: ${DateFormat('dd MMM yyyy, hh:mm a').format(paidDate)}',
                style: GoogleFonts.poppins(fontSize: 12, color: Colors.grey),
              ),
          ],
        ),
        trailing: Text(
          '${bill['usage']?.toStringAsFixed(1) ?? '0'} m³',
          style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
        ),
      ),
    ).animate().fadeIn(delay: (100 * index).ms);
  }
}

class BillingInsightsTab extends StatelessWidget {
  final FirebaseFirestore firestore;

  const BillingInsightsTab({super.key, required this.firestore});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream: firestore.collection('bills').snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        double totalRevenue = 0;
        double pendingAmount = 0;
        int paidCount = 0;
        int pendingCount = 0;
        final recentBills = <Map<String, dynamic>>[];

        if (snapshot.hasData) {
          for (final doc in snapshot.data!.docs) {
            final bill = doc.data() as Map<String, dynamic>;
            if (bill['status'] == 'Paid') {
              totalRevenue += (bill['amount'] ?? 0).toDouble();
              paidCount++;
            } else {
              pendingAmount += (bill['amount'] ?? 0).toDouble();
              pendingCount++;
            }

            if (bill['createdAt'] != null) {
              final createdAt = DateTime.fromMillisecondsSinceEpoch(
                bill['createdAt'],
              );
              if (DateTime.now().difference(createdAt).inDays < 7) {
                recentBills.add(bill);
              }
            }
          }
        }

        return SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              _buildInsightCard(
                title: 'Billing Statistics',
                children: [
                  _buildStatItem(
                    'Total Revenue',
                    'M${totalRevenue.toStringAsFixed(2)}',
                    Icons.attach_money,
                  ),
                  _buildStatItem(
                    'Pending Amount',
                    'M${pendingAmount.toStringAsFixed(2)}',
                    Icons.pending,
                  ),
                  _buildStatItem(
                    'Paid Bills',
                    paidCount.toString(),
                    Icons.check_circle,
                  ),
                  _buildStatItem(
                    'Pending Bills',
                    pendingCount.toString(),
                    Icons.pending_actions,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildInsightCard(
                title: 'Recent Bills',
                children: [
                  if (recentBills.isEmpty)
                    const Text('No recent bills')
                  else
                    ...recentBills
                        .take(5)
                        .map(
                          (bill) => ListTile(
                            leading: CircleAvatar(
                              backgroundColor:
                                  bill['status'] == 'Paid'
                                      ? Colors.green[100]
                                      : Colors.orange[100],
                              child: Icon(
                                bill['status'] == 'Paid'
                                    ? Icons.check
                                    : Icons.pending,
                                color:
                                    bill['status'] == 'Paid'
                                        ? Colors.green
                                        : Colors.orange,
                              ),
                            ),
                            title: Text(bill['customerName']),
                            subtitle: Text(
                              'M${(bill['amount'] ?? 0).toStringAsFixed(2)} - ${bill['status']}',
                            ),
                            trailing: Text(
                              '${bill['usage']?.toStringAsFixed(1) ?? '0'} m³',
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
