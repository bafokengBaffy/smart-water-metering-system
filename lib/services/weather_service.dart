import 'dart:convert';
import 'package:http/http.dart' as http;

class WeatherService {
  // Hardcoded API key for testing
  static const String _apiKey = 'fa49971f87eebb578199a5a203e1c1b6';
  static const String _baseUrl =
      'https://api.openweathermap.org/data/2.5/weather';

  Future<Map<String, dynamic>> fetchWeather(double lat, double lon) async {
    final response = await http.get(
      Uri.parse('$_baseUrl?lat=$lat&lon=$lon&appid=$_apiKey&units=metric'),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception(
        'Failed to load weather data. Status code: ${response.statusCode}',
      );
    }
  }
}
