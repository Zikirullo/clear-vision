// User settings, saved on the phone with shared_preferences (like localStorage in a
// browser: small key/value pairs that survive app restarts).
//
// Flutter notes:
// - `ChangeNotifier` is Flutter's simplest observable: call `notifyListeners()` after a
//   change and every widget listening rebuilds (think of a tiny Redux store / useState
//   that lives outside a component). Widgets listen with `ListenableBuilder`.
// - `Future<void>` = a JS Promise<void>; `async`/`await` work the same as in JS.

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Normal = speech + vibration. Silent = obstacle warnings by vibration only
/// (system alerts like "Camera lost" and "Battery low" are still spoken: safety first).
enum OutputMode { normal, silent }

/// Languages with a phrase pack. English comes from nav_config.json; others need
/// assets/phrases/<code>.json (see README) and are offered only when that file exists.
const Map<String, String> languageNames = {'en': 'English', 'ko': 'Korean', 'uz': 'Uzbek'};
const Map<String, String> ttsLocales = {'en': 'en-US', 'ko': 'ko-KR', 'uz': 'uz-UZ'};

class Settings extends ChangeNotifier {
  Settings._(this._prefs, {required this.defaultRate, required this.defaultHazardsOnly});

  /// Loads saved settings (or defaults from the config).
  static Future<Settings> load({required double defaultRate, required bool defaultHazardsOnly}) async {
    final prefs = await SharedPreferences.getInstance();
    return Settings._(prefs, defaultRate: defaultRate, defaultHazardsOnly: defaultHazardsOnly);
  }

  final SharedPreferences _prefs;
  final double defaultRate;
  final bool defaultHazardsOnly;

  double get speechRate => _prefs.getDouble('speechRate') ?? defaultRate;
  String get language => _prefs.getString('language') ?? 'en';
  OutputMode get mode => OutputMode.values.byName(_prefs.getString('mode') ?? OutputMode.normal.name);
  bool get vibration => _prefs.getBool('vibration') ?? true;
  bool get hazardsOnly => _prefs.getBool('hazardsOnly') ?? defaultHazardsOnly;

  // Setters write through to storage, then tell listeners.
  set speechRate(double v) => _save(() => _prefs.setDouble('speechRate', v));
  set language(String v) => _save(() => _prefs.setString('language', v));
  set mode(OutputMode v) => _save(() => _prefs.setString('mode', v.name));
  set vibration(bool v) => _save(() => _prefs.setBool('vibration', v));
  set hazardsOnly(bool v) => _save(() => _prefs.setBool('hazardsOnly', v));

  void _save(Future<bool> Function() write) {
    write(); // not awaited: the value is readable right away, storage catches up
    notifyListeners();
  }
}
