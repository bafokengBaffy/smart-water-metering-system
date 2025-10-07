// ignore_for_file: unused_field, avoid_types_as_parameter_names

import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:intl/intl.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

// Import your other pages
import 'analytics_page.dart';
import 'billing_management_page.dart';
import 'home_screen_admin.dart';
import 'user_management_page.dart';
import 'settings_screen_admin.dart';

class MeterManagementPage extends StatefulWidget {
  const MeterManagementPage({super.key});

  @override
  State<MeterManagementPage> createState() => _MeterManagementPageState();
}

class _MeterManagementPageState extends State<MeterManagementPage> {
  final int _currentIndex = 4;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final User? _user = FirebaseAuth.instance.currentUser;

  // Filter and search variables
  String _searchQuery = '';
  String _statusFilter = 'All';
  String _typeFilter = 'All';

  // View mode (list or map)
  bool _isMapView = false;

  // Selected meter for details
  Map<String, dynamic>? _selectedMeter;

  // Controller for Google Maps
  late GoogleMapController _mapController;
  final LatLng _center = const LatLng(-29.3167, 27.4833); // Default center

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Meter Management'),
        backgroundColor: Colors.blueGrey[800],
        actions: [
          IconButton(
            icon: Icon(_isMapView ? Icons.list : Icons.map),
            onPressed: () => setState(() => _isMapView = !_isMapView),
            tooltip: _isMapView ? 'List View' : 'Map View',
          ),
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: _showAddMeterDialog,
            tooltip: 'Add New Meter',
          ),
          IconButton(
            icon: const Icon(Icons.filter_list),
            onPressed: _showFilterDialog,
            tooltip: 'Filter Meters',
          ),
        ],
      ),
      body: Column(
        children: [
          // Search Bar
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Search meters...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              onChanged: (value) => setState(() => _searchQuery = value),
            ),
          ),

          // Quick Stats
          _buildQuickStats(),

          // Main Content (List or Map)
          Expanded(child: _isMapView ? _buildMapView() : _buildListView()),
        ],
      ),
      bottomNavigationBar: _buildBottomNavBar(),
    );
  }

  Widget _buildQuickStats() {
    return StreamBuilder<QuerySnapshot>(
      stream: _firestore.collection('meters').snapshots(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const SizedBox.shrink();
        }

        final meters = snapshot.data!.docs;
        final activeMeters =
            meters.where((m) => m['status'] == 'Active').length;
        final needsMaintenance =
            meters.where((m) => m['status'] == 'Maintenance').length;
        final totalConsumption = meters.fold<double>(
          0,
          (sum, m) => sum + (m['totalConsumption'] ?? 0),
        );

        return Container(
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
          color: Colors.grey[100],
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStatItem(
                'Total Meters',
                meters.length.toString(),
                Icons.speed,
              ),
              _buildStatItem(
                'Active',
                activeMeters.toString(),
                Icons.check_circle,
                Colors.green,
              ),
              _buildStatItem(
                'Maintenance',
                needsMaintenance.toString(),
                Icons.build,
                Colors.orange,
              ),
              _buildStatItem(
                'Total Usage',
                '${totalConsumption.toStringAsFixed(0)} L',
                Icons.water_drop,
                Colors.blue,
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildStatItem(
    String title,
    String value,
    IconData icon, [
    Color? color,
  ]) {
    return Column(
      children: [
        Icon(icon, color: color ?? Colors.grey, size: 20),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        Text(title, style: const TextStyle(fontSize: 12, color: Colors.grey)),
      ],
    );
  }

  Widget _buildListView() {
    return StreamBuilder<QuerySnapshot>(
      stream: _firestore.collection('meters').snapshots(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        var meters = snapshot.data!.docs;

        // Apply filters
        if (_statusFilter != 'All') {
          meters = meters.where((m) => m['status'] == _statusFilter).toList();
        }

        if (_typeFilter != 'All') {
          meters = meters.where((m) => m['type'] == _typeFilter).toList();
        }

        // Apply search
        if (_searchQuery.isNotEmpty) {
          meters =
              meters
                  .where(
                    (m) =>
                        m['meterID'].toString().toLowerCase().contains(
                          _searchQuery.toLowerCase(),
                        ) ||
                        m['location'].toString().toLowerCase().contains(
                          _searchQuery.toLowerCase(),
                        ),
                  )
                  .toList();
        }

        if (meters.isEmpty) {
          return const Center(
            child: Text('No meters found', style: TextStyle(fontSize: 18)),
          );
        }

        return ListView.builder(
          itemCount: meters.length,
          itemBuilder: (context, index) {
            final meter = meters[index];
            return MeterCard(
              meter: meter.data() as Map<String, dynamic>,
              onTap:
                  () => _showMeterDetails(meter.data() as Map<String, dynamic>),
              onStatusToggle:
                  () => _toggleMeterStatus(meter.id, meter['status']),
              onEdit:
                  () => _showEditMeterDialog(
                    meter.id,
                    meter.data() as Map<String, dynamic>,
                  ),
            );
          },
        );
      },
    );
  }

  Widget _buildMapView() {
    return StreamBuilder<QuerySnapshot>(
      stream: _firestore.collection('meters').snapshots(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        final meters = snapshot.data!.docs;
        final markers =
            meters.map((meter) {
              final data = meter.data() as Map<String, dynamic>;
              return Marker(
                markerId: MarkerId(meter.id),
                position: LatLng(
                  data['latitude'] ?? 0.0,
                  data['longitude'] ?? 0.0,
                ),
                infoWindow: InfoWindow(
                  title: data['meterID'],
                  snippet: data['location'],
                  onTap: () => _showMeterDetails(data),
                ),
                icon: BitmapDescriptor.defaultMarkerWithHue(
                  data['status'] == 'Active'
                      ? BitmapDescriptor.hueGreen
                      : data['status'] == 'Maintenance'
                      ? BitmapDescriptor.hueOrange
                      : BitmapDescriptor.hueRed,
                ),
              );
            }).toSet();

        return GoogleMap(
          initialCameraPosition: CameraPosition(target: _center, zoom: 10),
          markers: markers,
          myLocationEnabled: true,
          myLocationButtonEnabled: true,
          onMapCreated: (GoogleMapController controller) {
            _mapController = controller;
          },
        );
      },
    );
  }

  Widget _buildBottomNavBar() {
    return BottomNavigationBar(
      currentIndex: _currentIndex,
      onTap: _onTabTapped,
      selectedItemColor: Colors.blueAccent,
      unselectedItemColor: Colors.grey,
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

  void _showAddMeterDialog() {
    final formKey = GlobalKey<FormState>();
    final meterIDController = TextEditingController();
    final locationController = TextEditingController();
    final latitudeController = TextEditingController(text: '-29.3167');
    final longitudeController = TextEditingController(text: '27.4833');
    String selectedType = 'Residential';
    String selectedStatus = 'Active';

    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Add New Meter'),
            content: Form(
              key: formKey,
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TextFormField(
                      controller: meterIDController,
                      decoration: const InputDecoration(labelText: 'Meter ID'),
                      validator: (value) => value!.isEmpty ? 'Required' : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: locationController,
                      decoration: const InputDecoration(labelText: 'Location'),
                      validator: (value) => value!.isEmpty ? 'Required' : null,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            controller: latitudeController,
                            decoration: const InputDecoration(
                              labelText: 'Latitude',
                            ),
                            keyboardType: TextInputType.number,
                            validator:
                                (value) => value!.isEmpty ? 'Required' : null,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: TextFormField(
                            controller: longitudeController,
                            decoration: const InputDecoration(
                              labelText: 'Longitude',
                            ),
                            keyboardType: TextInputType.number,
                            validator:
                                (value) => value!.isEmpty ? 'Required' : null,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      initialValue: selectedType,
                      items:
                          ['Residential', 'Commercial', 'Industrial']
                              .map(
                                (type) => DropdownMenuItem(
                                  value: type,
                                  child: Text(type),
                                ),
                              )
                              .toList(),
                      onChanged: (value) => selectedType = value!,
                      decoration: const InputDecoration(
                        labelText: 'Meter Type',
                      ),
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      initialValue: selectedStatus,
                      items:
                          ['Active', 'Inactive', 'Maintenance']
                              .map(
                                (status) => DropdownMenuItem(
                                  value: status,
                                  child: Text(status),
                                ),
                              )
                              .toList(),
                      onChanged: (value) => selectedStatus = value!,
                      decoration: const InputDecoration(labelText: 'Status'),
                    ),
                  ],
                ),
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () {
                  if (formKey.currentState!.validate()) {
                    _addMeterToFirebase(
                      meterIDController.text,
                      locationController.text,
                      selectedType,
                      selectedStatus,
                      double.parse(latitudeController.text),
                      double.parse(longitudeController.text),
                    );
                    Navigator.pop(context);
                  }
                },
                child: const Text('Add Meter'),
              ),
            ],
          ),
    );
  }

  void _addMeterToFirebase(
    String meterID,
    String location,
    String type,
    String status,
    double latitude,
    double longitude,
  ) {
    _firestore
        .collection('meters')
        .add({
          'meterID': meterID,
          'location': location,
          'type': type,
          'status': status,
          'latitude': latitude,
          'longitude': longitude,
          'installationDate': Timestamp.now(),
          'lastReading': 0.0,
          'totalConsumption': 0.0,
          'createdBy': _user?.uid,
          'createdAt': Timestamp.now(),
        })
        .then((value) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Meter added successfully')),
          );
        })
        .catchError((error) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to add meter: $error')),
          );
        });
  }

  void _showFilterDialog() {
    String status = _statusFilter;
    String type = _typeFilter;

    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Filter Meters'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                DropdownButtonFormField<String>(
                  initialValue: status,
                  items:
                      ['All', 'Active', 'Inactive', 'Maintenance']
                          .map(
                            (s) => DropdownMenuItem(value: s, child: Text(s)),
                          )
                          .toList(),
                  onChanged: (value) => status = value!,
                  decoration: const InputDecoration(labelText: 'Status'),
                ),
                const SizedBox(height: 10),
                DropdownButtonFormField<String>(
                  initialValue: type,
                  items:
                      ['All', 'Residential', 'Commercial', 'Industrial']
                          .map(
                            (t) => DropdownMenuItem(value: t, child: Text(t)),
                          )
                          .toList(),
                  onChanged: (value) => type = value!,
                  decoration: const InputDecoration(labelText: 'Type'),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () {
                  setState(() {
                    _statusFilter = 'All';
                    _typeFilter = 'All';
                  });
                  Navigator.pop(context);
                },
                child: const Text('Reset'),
              ),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _statusFilter = status;
                    _typeFilter = type;
                  });
                  Navigator.pop(context);
                },
                child: const Text('Apply'),
              ),
            ],
          ),
    );
  }

  void _showMeterDetails(Map<String, dynamic> meter) {
    setState(() {
      _selectedMeter = meter;
    });

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => MeterDetailsSheet(meter: meter),
    );
  }

  void _toggleMeterStatus(String meterId, String currentStatus) async {
    String newStatus = currentStatus == 'Active' ? 'Inactive' : 'Active';

    try {
      await _firestore.collection('meters').doc(meterId).update({
        'status': newStatus,
        'updatedAt': Timestamp.now(),
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Meter status changed to $newStatus')),
      );
    } catch (error) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to update meter status: $error')),
      );
    }
  }

  void _showEditMeterDialog(String meterId, Map<String, dynamic> meterData) {
    final formKey = GlobalKey<FormState>();
    final meterIDController = TextEditingController(text: meterData['meterID']);
    final locationController = TextEditingController(
      text: meterData['location'],
    );
    final latitudeController = TextEditingController(
      text: meterData['latitude'].toString(),
    );
    final longitudeController = TextEditingController(
      text: meterData['longitude'].toString(),
    );
    String selectedType = meterData['type'];
    String selectedStatus = meterData['status'];

    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Edit Meter'),
            content: Form(
              key: formKey,
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TextFormField(
                      controller: meterIDController,
                      decoration: const InputDecoration(labelText: 'Meter ID'),
                      validator: (value) => value!.isEmpty ? 'Required' : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: locationController,
                      decoration: const InputDecoration(labelText: 'Location'),
                      validator: (value) => value!.isEmpty ? 'Required' : null,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            controller: latitudeController,
                            decoration: const InputDecoration(
                              labelText: 'Latitude',
                            ),
                            keyboardType: TextInputType.number,
                            validator:
                                (value) => value!.isEmpty ? 'Required' : null,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: TextFormField(
                            controller: longitudeController,
                            decoration: const InputDecoration(
                              labelText: 'Longitude',
                            ),
                            keyboardType: TextInputType.number,
                            validator:
                                (value) => value!.isEmpty ? 'Required' : null,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      initialValue: selectedType,
                      items:
                          ['Residential', 'Commercial', 'Industrial']
                              .map(
                                (type) => DropdownMenuItem(
                                  value: type,
                                  child: Text(type),
                                ),
                              )
                              .toList(),
                      onChanged: (value) => selectedType = value!,
                      decoration: const InputDecoration(
                        labelText: 'Meter Type',
                      ),
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      initialValue: selectedStatus,
                      items:
                          ['Active', 'Inactive', 'Maintenance']
                              .map(
                                (status) => DropdownMenuItem(
                                  value: status,
                                  child: Text(status),
                                ),
                              )
                              .toList(),
                      onChanged: (value) => selectedStatus = value!,
                      decoration: const InputDecoration(labelText: 'Status'),
                    ),
                  ],
                ),
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () {
                  if (formKey.currentState!.validate()) {
                    _updateMeterInFirebase(
                      meterId,
                      meterIDController.text,
                      locationController.text,
                      selectedType,
                      selectedStatus,
                      double.parse(latitudeController.text),
                      double.parse(longitudeController.text),
                    );
                    Navigator.pop(context);
                  }
                },
                child: const Text('Update Meter'),
              ),
            ],
          ),
    );
  }

  void _updateMeterInFirebase(
    String meterId,
    String meterID,
    String location,
    String type,
    String status,
    double latitude,
    double longitude,
  ) {
    _firestore
        .collection('meters')
        .doc(meterId)
        .update({
          'meterID': meterID,
          'location': location,
          'type': type,
          'status': status,
          'latitude': latitude,
          'longitude': longitude,
          'updatedAt': Timestamp.now(),
        })
        .then((value) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Meter updated successfully')),
          );
        })
        .catchError((error) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to update meter: $error')),
          );
        });
  }

  void _onTabTapped(int index) {
    if (index == _currentIndex) return; // Already on this page

    switch (index) {
      case 0:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const HomeScreenAdmin()),
        );
        break;
      case 1:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const AnalyticsPage()),
        );
        break;
      case 2:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const UserManagementPage()),
        );
        break;
      case 3:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => const BillingManagementPage(),
          ),
        );
        break;
      case 5:
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const SettingsScreen()),
        );
        break;
    }
  }
}

class MeterCard extends StatelessWidget {
  final Map<String, dynamic> meter;
  final VoidCallback onTap;
  final VoidCallback onStatusToggle;
  final VoidCallback onEdit;

  const MeterCard({
    super.key,
    required this.meter,
    required this.onTap,
    required this.onStatusToggle,
    required this.onEdit,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        contentPadding: const EdgeInsets.all(16),
        leading: _buildStatusIndicator(meter['status']),
        title: Text(
          meter['meterID'],
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(meter['location']),
            const SizedBox(height: 4),
            Text('Type: ${meter['type']}'),
            const SizedBox(height: 4),
            Text(
              'Last Reading: ${meter['lastReading']?.toStringAsFixed(1) ?? '0.0'} L',
            ),
          ],
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              icon: const Icon(Icons.edit, size: 20),
              onPressed: onEdit,
              tooltip: 'Edit Meter',
            ),
            IconButton(
              icon: Icon(
                meter['status'] == 'Active'
                    ? Icons.toggle_on
                    : Icons.toggle_off,
                color: meter['status'] == 'Active' ? Colors.green : Colors.grey,
              ),
              onPressed: onStatusToggle,
              tooltip: 'Toggle Status',
            ),
          ],
        ),
        onTap: onTap,
      ),
    );
  }

  Widget _buildStatusIndicator(String status) {
    Color color;
    IconData icon;

    switch (status) {
      case 'Active':
        color = Colors.green;
        icon = Icons.check_circle;
        break;
      case 'Maintenance':
        color = Colors.orange;
        icon = Icons.build;
        break;
      default:
        color = Colors.grey;
        icon = Icons.remove_circle;
    }

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, color: color, size: 28),
        const SizedBox(height: 4),
        Text(status, style: TextStyle(color: color, fontSize: 10)),
      ],
    );
  }
}

class MeterDetailsSheet extends StatelessWidget {
  final Map<String, dynamic> meter;

  const MeterDetailsSheet({super.key, required this.meter});

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.7,
      maxChildSize: 0.9,
      builder: (context, scrollController) {
        return SingleChildScrollView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 5,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Text(
                meter['meterID'],
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                meter['location'],
                style: TextStyle(fontSize: 16, color: Colors.grey[600]),
              ),
              const SizedBox(height: 16),
              _buildDetailRow('Type', meter['type']),
              _buildDetailRow('Status', meter['status']),
              _buildDetailRow(
                'Last Reading',
                '${meter['lastReading']?.toStringAsFixed(1) ?? '0.0'} L',
              ),
              _buildDetailRow(
                'Total Consumption',
                '${meter['totalConsumption']?.toStringAsFixed(1) ?? '0.0'} L',
              ),
              _buildDetailRow(
                'Installation Date',
                DateFormat(
                  'MMM dd, yyyy',
                ).format((meter['installationDate'] as Timestamp).toDate()),
              ),
              const SizedBox(height: 24),
              const Text(
                'Usage History',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 200,
                child: SfCartesianChart(
                  primaryXAxis: CategoryAxis(),
                  series: <CartesianSeries>[
                    LineSeries<Map<String, dynamic>, String>(
                      dataSource: _generateSampleData(),
                      xValueMapper: (data, _) => data['month'],
                      yValueMapper: (data, _) => data['usage'],
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Text('$label: ', style: const TextStyle(fontWeight: FontWeight.bold)),
          Text(value),
        ],
      ),
    );
  }

  List<Map<String, dynamic>> _generateSampleData() {
    // This would typically come from Firebase
    return [
      {'month': 'Jan', 'usage': 120},
      {'month': 'Feb', 'usage': 135},
      {'month': 'Mar', 'usage': 128},
      {'month': 'Apr', 'usage': 141},
      {'month': 'May', 'usage': 156},
      {'month': 'Jun', 'usage': 162},
    ];
  }
}
