// ignore_for_file: deprecated_member_use, use_build_context_synchronously

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import 'analysis_screen.dart';
import 'home_screen.dart';
import 'settings_screen.dart';

class PaymentScreen extends StatefulWidget {
  const PaymentScreen({super.key});

  @override
  PaymentScreenState createState() => PaymentScreenState();
}

class PaymentScreenState extends State<PaymentScreen> {
  final _formKey = GlobalKey<FormState>();
  String _selectedPaymentMethod = 'M-Pesa';
  bool _isProcessing = false;
  String _paymentReference = '';
  bool _paymentInitiated = false;
  bool _autoPayEnabled = false;
  int _selectedTab = 0; // 0: Pay Now, 1: History, 2: Invoices

  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _amountController = TextEditingController(
    text: '245.60',
  );
  final TextEditingController _referenceController = TextEditingController();

  final List<String> paymentMethods = [
    'M-Pesa',
    'EcoCash',
    'Credit Card',
    'Bank Transfer',
  ];
  final List<Map<String, dynamic>> paymentHistory = [
    {
      'date': '2023-09-05',
      'amount': 245.60,
      'method': 'M-Pesa',
      'status': 'Completed',
    },
    {
      'date': '2023-08-05',
      'amount': 210.30,
      'method': 'EcoCash',
      'status': 'Completed',
    },
    {
      'date': '2023-07-05',
      'amount': 198.75,
      'method': 'Bank Transfer',
      'status': 'Completed',
    },
  ];

  final List<Map<String, dynamic>> invoices = [
    {
      'period': 'Sep 2023',
      'amount': 245.60,
      'usage': 3200,
      'status': 'Due Sept 10',
    },
    {'period': 'Aug 2023', 'amount': 210.30, 'usage': 2800, 'status': 'Paid'},
    {'period': 'Jul 2023', 'amount': 198.75, 'usage': 2650, 'status': 'Paid'},
  ];

  @override
  void initState() {
    super.initState();
    _generateReference();
  }

  @override
  void dispose() {
    _phoneController.dispose();
    _amountController.dispose();
    _referenceController.dispose();
    super.dispose();
  }

  void _generateReference() {
    final now = DateTime.now();
    setState(() {
      _paymentReference = 'PAY-${DateFormat('yyyyMMddHHmmss').format(now)}';
      _referenceController.text = _paymentReference;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Payment & Billing'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.blue.shade700,
        actions: [
          if (_paymentInitiated)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _verifyPayment,
              tooltip: 'Verify Payment',
            ),
        ],
      ),
      body: Column(
        children: [
          _buildTabBar(),
          Expanded(
            child:
                _selectedTab == 0
                    ? _buildPaymentTab()
                    : _selectedTab == 1
                    ? _buildHistoryTab()
                    : _buildInvoicesTab(),
          ),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildTabBar() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(16),
          bottomRight: Radius.circular(16),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _buildTabButton(0, 'Pay Now', Icons.payment),
            _buildTabButton(1, 'History', Icons.history),
            _buildTabButton(2, 'Invoices', Icons.receipt),
          ],
        ),
      ),
    );
  }

  Widget _buildTabButton(int index, String label, IconData icon) {
    return TextButton.icon(
      onPressed: () => setState(() => _selectedTab = index),
      icon: Icon(
        icon,
        color: _selectedTab == index ? Colors.blue.shade700 : Colors.grey,
      ),
      label: Text(
        label,
        style: TextStyle(
          color: _selectedTab == index ? Colors.blue.shade700 : Colors.grey,
          fontWeight:
              _selectedTab == index ? FontWeight.bold : FontWeight.normal,
        ),
      ),
    );
  }

  Widget _buildPaymentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildBillingSummary(),
          const SizedBox(height: 24),
          _buildPaymentOptions(),
          const SizedBox(height: 24),
          _buildInvoiceBreakdown(),
          const SizedBox(height: 24),
          _buildSmartFeatures(),
        ],
      ),
    );
  }

  Widget _buildBillingSummary() {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Billing Summary',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            _buildSummaryRow('Account ID', 'SWM-728491'),
            _buildSummaryRow('Billing Period', 'Aug 1 - Aug 31, 2023'),
            _buildSummaryRow('Water Usage', '3,200 L'),
            _buildSummaryRow('Amount Due', 'R 245.60', isAmount: true),
            _buildSummaryRow('Due Date', 'Sept 10, 2023'),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Status:',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.grey.shade700,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade100,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(
                    'Due in 5 days',
                    style: TextStyle(
                      color: Colors.orange.shade800,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryRow(String label, String value, {bool isAmount = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey.shade700)),
          Text(
            value,
            style: TextStyle(
              fontWeight: isAmount ? FontWeight.bold : FontWeight.normal,
              fontSize: isAmount ? 16 : 14,
              color: isAmount ? Colors.blue.shade700 : Colors.black,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPaymentOptions() {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Payment Methods',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            _buildPaymentMethodSelector(),
            const SizedBox(height: 16),
            if (_selectedPaymentMethod == 'M-Pesa' ||
                _selectedPaymentMethod == 'EcoCash')
              _buildMobilePaymentForm()
            else if (_selectedPaymentMethod == 'Credit Card')
              _buildCreditCardForm()
            else
              _buildBankTransferForm(),
            const SizedBox(height: 16),
            _buildPayButton(),
          ],
        ),
      ),
    );
  }

  Widget _buildPaymentMethodSelector() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children:
          paymentMethods.map((method) {
            final isSelected = _selectedPaymentMethod == method;
            return ChoiceChip(
              label: Text(method),
              selected: isSelected,
              onSelected: (selected) {
                setState(() {
                  _selectedPaymentMethod = method;
                });
              },
              selectedColor: Colors.blue.shade700,
              labelStyle: TextStyle(
                color: isSelected ? Colors.white : Colors.black,
              ),
            );
          }).toList(),
    );
  }

  Widget _buildMobilePaymentForm() {
    return Column(
      children: [
        TextFormField(
          controller: _phoneController,
          keyboardType: TextInputType.phone,
          decoration: InputDecoration(
            labelText: 'Mobile Number',
            prefixText: '+266 ',
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
            filled: true,
            fillColor: Colors.grey[100],
          ),
          validator: (value) {
            if (value == null || value.isEmpty) return 'Enter phone number';
            if (value.length != 8) return 'Must be 8 digits';
            return null;
          },
        ),
        const SizedBox(height: 16),
        TextFormField(
          controller: _referenceController,
          readOnly: true,
          decoration: InputDecoration(
            labelText: 'Payment Reference',
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
            filled: true,
            fillColor: Colors.grey[100],
          ),
        ),
      ],
    );
  }

  Widget _buildCreditCardForm() {
    return Column(
      children: [
        TextFormField(
          decoration: InputDecoration(
            labelText: 'Card Number',
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
            filled: true,
            fillColor: Colors.grey[100],
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: TextFormField(
                decoration: InputDecoration(
                  labelText: 'Expiry Date',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  filled: true,
                  fillColor: Colors.grey[100],
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: TextFormField(
                decoration: InputDecoration(
                  labelText: 'CVV',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  filled: true,
                  fillColor: Colors.grey[100],
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildBankTransferForm() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Bank Transfer Details:'),
        const SizedBox(height: 8),
        _buildBankDetailRow('Bank Name', 'Standard Lesotho Bank'),
        _buildBankDetailRow('Account Number', '9801754321'),
        _buildBankDetailRow('Branch Code', '12345'),
        _buildBankDetailRow('Reference', _paymentReference),
        const SizedBox(height: 8),
        const Text('Please use the reference code when making your transfer.'),
      ],
    );
  }

  Widget _buildBankDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          Text(value),
        ],
      ),
    );
  }

  Widget _buildPayButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: _isProcessing ? null : _initiatePayment,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          backgroundColor: Colors.blue.shade700,
        ),
        child:
            _isProcessing
                ? const SizedBox(
                  height: 20,
                  width: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation(Colors.white),
                  ),
                )
                : Text(
                  'PAY R ${_amountController.text}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
      ),
    );
  }

  Widget _buildInvoiceBreakdown() {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Invoice Breakdown',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                IconButton(
                  icon: const Icon(Icons.download),
                  onPressed: () => _downloadInvoice(),
                  tooltip: 'Download Invoice',
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildInvoiceRow('Base Fee', 'R 50.00'),
            _buildInvoiceRow('Usage (3200 L @ R0.075/L)', 'R 240.00'),
            _buildInvoiceRow('VAT (15%)', 'R 36.00'),
            const Divider(),
            _buildInvoiceRow('Total Amount', 'R 245.60', isTotal: true),
            const SizedBox(height: 8),
            Text(
              'Tariff Rate: R 0.075 per liter',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInvoiceRow(String label, String value, {bool isTotal = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontWeight: isTotal ? FontWeight.bold : FontWeight.normal,
              color: isTotal ? Colors.blue.shade700 : Colors.black,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontWeight: isTotal ? FontWeight.bold : FontWeight.normal,
              color: isTotal ? Colors.blue.shade700 : Colors.black,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSmartFeatures() {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Smart Features',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Auto-Pay',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                Switch(
                  value: _autoPayEnabled,
                  onChanged: (value) => setState(() => _autoPayEnabled = value),
                  activeThumbColor: Colors.blue.shade700,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Automatically pay your bill on the due date',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 16),
            const Text(
              'Estimated Next Bill: R 230.40',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SizedBox(height: 120, child: _buildUsageCostChart()),
          ],
        ),
      ),
    );
  }

  Widget _buildUsageCostChart() {
    return LineChart(
      LineChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                final months = ['Jun', 'Jul', 'Aug', 'Sep'];
                if (value.toInt() < months.length) {
                  return Padding(
                    padding: const EdgeInsets.only(top: 8.0),
                    child: Text(months[value.toInt()]),
                  );
                }
                return const Text('');
              },
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                return Text(value.toInt().toString());
              },
            ),
          ),
        ),
        borderData: FlBorderData(show: false),
        minX: 0,
        maxX: 3,
        minY: 0,
        maxY: 300,
        lineBarsData: [
          LineChartBarData(
            spots: const [
              FlSpot(0, 180),
              FlSpot(1, 200),
              FlSpot(2, 245),
              FlSpot(3, 230),
            ],
            isCurved: true,
            color: Colors.blue.shade700,
            barWidth: 4,
            isStrokeCapRound: true,
            dotData: FlDotData(show: true),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                colors: [
                  Colors.blue.shade700.withOpacity(0.3),
                  Colors.blue.shade100.withOpacity(0.1),
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryTab() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: paymentHistory.length,
      itemBuilder: (context, index) {
        final payment = paymentHistory[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            leading: CircleAvatar(
              backgroundColor: Colors.blue.shade100,
              child: Icon(Icons.payment, color: Colors.blue.shade700),
            ),
            title: Text('R ${payment['amount']}'),
            subtitle: Text('${payment['method']} • ${payment['date']}'),
            trailing: Chip(
              label: Text(payment['status']),
              backgroundColor: Colors.green.shade100,
              labelStyle: TextStyle(color: Colors.green.shade800),
            ),
          ),
        );
      },
    );
  }

  Widget _buildInvoicesTab() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: invoices.length,
      itemBuilder: (context, index) {
        final invoice = invoices[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            leading: CircleAvatar(
              backgroundColor: Colors.blue.shade100,
              child: Icon(Icons.receipt, color: Colors.blue.shade700),
            ),
            title: Text('${invoice['period']} • ${invoice['usage']} L'),
            subtitle: Text('R ${invoice['amount']}'),
            trailing: Chip(
              label: Text(invoice['status']),
              backgroundColor:
                  invoice['status'] == 'Paid'
                      ? Colors.green.shade100
                      : Colors.orange.shade100,
              labelStyle: TextStyle(
                color:
                    invoice['status'] == 'Paid'
                        ? Colors.green.shade800
                        : Colors.orange.shade800,
              ),
            ),
            onTap: () => _viewInvoiceDetails(invoice),
          ),
        );
      },
    );
  }

  Future<void> _initiatePayment() async {
    if ((_selectedPaymentMethod == 'M-Pesa' ||
            _selectedPaymentMethod == 'EcoCash') &&
        !_formKey.currentState!.validate()) {
      return;
    }

    setState(() => _isProcessing = true);

    try {
      // Simulate payment processing
      await Future.delayed(const Duration(seconds: 2));

      setState(() {
        _isProcessing = false;
        _paymentInitiated = true;
      });

      // Show payment confirmation
      _showPaymentConfirmation();
    } catch (e) {
      setState(() => _isProcessing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Payment failed: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _showPaymentConfirmation() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Payment Successful'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Your payment has been processed successfully.'),
                const SizedBox(height: 16),
                _buildConfirmationRow('Amount', 'R ${_amountController.text}'),
                _buildConfirmationRow('Method', _selectedPaymentMethod),
                _buildConfirmationRow('Reference', _paymentReference),
                _buildConfirmationRow(
                  'Date',
                  DateFormat('yyyy-MM-dd HH:mm').format(DateTime.now()),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Checkbox(value: true, onChanged: null),
                    const Text('Email receipt'),
                  ],
                ),
                Row(
                  children: [
                    Checkbox(value: false, onChanged: null),
                    const Text('SMS receipt'),
                  ],
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Done'),
              ),
            ],
          ),
    );
  }

  Widget _buildConfirmationRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('$label:', style: const TextStyle(fontWeight: FontWeight.bold)),
          Text(value),
        ],
      ),
    );
  }

  Future<void> _verifyPayment() async {
    // Simulate payment verification
    setState(() => _isProcessing = true);
    await Future.delayed(const Duration(seconds: 2));
    setState(() => _isProcessing = false);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Payment verified successfully!'),
        backgroundColor: Colors.green,
      ),
    );
  }

  void _downloadInvoice() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Invoice download started'),
        backgroundColor: Colors.green,
      ),
    );
  }

  void _viewInvoiceDetails(Map<String, dynamic> invoice) {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: Text('Invoice ${invoice['period']}'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildInvoiceDetailRow('Period', invoice['period']),
                _buildInvoiceDetailRow('Water Usage', '${invoice['usage']} L'),
                _buildInvoiceDetailRow('Amount', 'R ${invoice['amount']}'),
                _buildInvoiceDetailRow('Status', invoice['status']),
                const SizedBox(height: 16),
                const Text('Breakdown:'),
                _buildInvoiceDetailRow('Base Fee', 'R 50.00'),
                _buildInvoiceDetailRow(
                  'Usage Charge',
                  'R ${(invoice['usage'] * 0.075).toStringAsFixed(2)}',
                ),
                _buildInvoiceDetailRow(
                  'VAT (15%)',
                  'R ${(invoice['amount'] * 0.15).toStringAsFixed(2)}',
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                  _downloadInvoice();
                },
                child: const Text('Download'),
              ),
            ],
          ),
    );
  }

  Widget _buildInvoiceDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('$label:', style: const TextStyle(fontWeight: FontWeight.bold)),
          Text(value),
        ],
      ),
    );
  }

  Widget _buildBottomNav() {
    return SafeArea(
      child: Container(
        height: 70,
        decoration: const BoxDecoration(
          border: Border(top: BorderSide(color: Colors.grey, width: 0.3)),
        ),
        child: BottomAppBar(
          padding: EdgeInsets.zero,
          elevation: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavButton(
                icon: Icons.home,
                label: 'Home',
                isActive: false,
                onTap:
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const HomeScreen()),
                    ),
              ),
              _NavButton(
                icon: Icons.analytics,
                label: 'Stats',
                isActive: false,
                onTap:
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const AnalysisScreen()),
                    ),
              ),
              _NavButton(
                icon: Icons.payment,
                label: 'Pay',
                isActive: true,
                onTap: () {},
              ),
              _NavButton(
                icon: Icons.settings,
                label: 'Settings',
                isActive: false,
                onTap:
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const SettingsScreen()),
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NavButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final VoidCallback onTap;

  const _NavButton({
    required this.icon,
    required this.label,
    this.isActive = false,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color:
              isActive
                  ? Colors.blueAccent.withAlpha((0.2 * 255).toInt())
                  : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isActive ? Colors.blueAccent : Colors.grey,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: isActive ? Colors.blueAccent : Colors.grey,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
