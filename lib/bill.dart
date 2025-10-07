import 'package:flutter/material.dart';
import 'payment_screen.dart'; // Ensure you import the PaymentScreen

class BillScreen extends StatelessWidget {
  const BillScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Billing Information'),
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            // Billing Overview Section
            _buildSectionTitle('Billing Overview'),
            const SizedBox(height: 12),
            Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              color: Colors.teal.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Your bill is generated based on your water usage along with additional charges.',
                      style: TextStyle(fontSize: 16),
                    ),
                    const SizedBox(height: 16),
                    _buildBillDetailRow('Water Usage:', '15,000 L'),
                    _buildBillDetailRow('Rate per Litre:', '1.25 Maluti'),
                    _buildBillDetailRow('Fixed Charges:', '5.00 Maluti'),
                    const Divider(height: 30, thickness: 1),
                    _buildBillDetailRow(
                      'Total Amount:',
                      '20.00 Maluti',
                      isTotal: true,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Bill Summary Section
            _buildSectionTitle('Bill Summary'),
            const SizedBox(height: 12),
            Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              color: Colors.orange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    _buildBillDetailRow('Subtotal:', '18.75 Maluti'),
                    _buildBillDetailRow('Taxes (5%):', '1.25 Maluti'),
                    _buildBillDetailRow('Discount Applied:', '-0.00 Maluti'),
                    const Divider(height: 30, thickness: 1),
                    _buildBillDetailRow(
                      'Total Due:',
                      '20.00 Maluti',
                      isTotal: true,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Payment Button Section
            _buildSectionTitle('Payment'),
            const SizedBox(height: 12),
            Center(
              child: ElevatedButton.icon(
                onPressed: () {
                  // Redirect to PaymentScreen
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const PaymentScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                    horizontal: 32,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                icon: const Icon(Icons.payment, size: 24),
                label: const Text('Pay Bill', style: TextStyle(fontSize: 18)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 26,
        fontWeight: FontWeight.bold,
        color: Colors.black87,
      ),
    );
  }

  Widget _buildBillDetailRow(
    String label,
    String value, {
    bool isTotal = false,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 16,
              color: isTotal ? Colors.black : Colors.grey[700],
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontWeight: isTotal ? FontWeight.bold : FontWeight.normal,
              color: isTotal ? Colors.black : Colors.grey[800],
            ),
          ),
        ],
      ),
    );
  }
}
