// Vibration for warnings: pulses = side, speed = distance (lib/navigation/haptics_pattern.dart).
// Uses the `vibration` package (Core Haptics on iPhone, Vibrator on Android).

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:vibration/vibration.dart';

import '../navigation/haptics_pattern.dart';
import '../navigation/nav_config.dart';
import '../navigation/navigator.dart';

class HapticsService {
  HapticsService(this.cfg);

  final HapticsConfig cfg;
  bool _available = false;

  Future<void> init() async {
    try {
      _available = await Vibration.hasVibrator();
    } catch (e) {
      debugPrint('Vibration unavailable: $e');
    }
  }

  void warn(Message msg) {
    if (!_available) return;
    unawaited(Vibration.vibrate(pattern: hapticPattern(msg.direction, msg.proximity, cfg)));
  }

  /// A short double buzz for system alerts (camera lost, battery).
  void alert() {
    if (!_available) return;
    unawaited(Vibration.vibrate(pattern: const [0, 300, 100, 300]));
  }
}
