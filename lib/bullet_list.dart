// Custom widget for rendering bullet points in a list format.
import 'package:flutter/material.dart';

class BulletList extends StatelessWidget {
  final List<String> items;
  const BulletList({super.key, required this.items});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: items
          .map(
            (item) => Row(
              children: [
                const Icon(Icons.circle, size: 8, color: Colors.black),
                const SizedBox(width: 8),
                Expanded(child: Text(item)),
              ],
            ),
          )
          .toList(),
    );
  }
}
