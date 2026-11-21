// Health checks: camera lost/back, slow detection, low battery. Pure logic with an
// explicit clock (`t` in seconds), so it is unit-tested without a phone. The app feeds it
// frame times and battery readings and speaks the events it returns.
//
// Rules (numbers from the `app` section of nav_config.json):
// - Camera lost: no frame for `stale_timeout` (1.5 s) while running. Shorter gaps stay
//   silent. "Camera back" is said once when frames return.
// - Slow: fewer than `slow_fps` (8) frames in the last second for `slow_fps_seconds` (3 s).
//   Repeated at most every `slow_fps_repeat_s` while still slow.
// - Battery: once at <= `battery_low_percent`, once more at <= `battery_critical_percent`;
//   re-armed after charging 5 points above the low level.
//
// Dart notes:
// - `enum HealthEvent { ... }` lists the possible events; the app maps them to phrases.
// - `List<HealthEvent> tick(...)` returns a (usually empty) list, like returning [] in JS.

import 'nav_config.dart';

enum HealthEvent { cameraLost, cameraBack, slow, batteryLow, batteryCritical }

class HealthMonitor {
  HealthMonitor(this.cfg);

  final AppConfig cfg;

  bool _running = false;
  double _since = 0; // when monitoring (re)started
  double? _lastFrame;
  bool _cameraLost = false;
  final List<double> _recentFrames = []; // frame times within the last second
  double? _slowSince;
  double? _lastSlowAlert;
  bool _batteryLowSaid = false;
  bool _batteryCriticalSaid = false;

  bool get cameraLost => _cameraLost;

  /// Frames per second over the last second (what the debug screen shows).
  double fps(double t) {
    _recentFrames.removeWhere((f) => t - f > 1.0);
    return _recentFrames.length.toDouble();
  }

  void start(double t) {
    _running = true;
    _since = t;
    _lastFrame = null;
    _cameraLost = false;
    _recentFrames.clear();
    _slowSince = null;
  }

  void stop() => _running = false;

  /// A frame arrived (with or without detections). Returns cameraBack once after a loss.
  List<HealthEvent> onFrame(double t) {
    _lastFrame = t;
    _recentFrames.add(t);
    if (_running && _cameraLost) {
      _cameraLost = false;
      _slowSince = null; // the first frames after a gap are not "slow"
      return [HealthEvent.cameraBack];
    }
    return const [];
  }

  /// Called regularly (every 250 ms) by the app.
  List<HealthEvent> tick(double t) {
    if (!_running) return const [];
    final events = <HealthEvent>[];
    final last = _lastFrame ?? _since; // no frame yet: count from the start
    if (!_cameraLost && t - last >= cfg.staleTimeout) {
      _cameraLost = true;
      events.add(HealthEvent.cameraLost);
    }
    if (_cameraLost) return events; // "slow" is meaningless without frames

    final warmedUp = t - _since >= 1.0;
    if (warmedUp && fps(t) < cfg.slowFps) {
      _slowSince ??= t; // `??=` assigns only if null
      final slowFor = t - _slowSince!;
      final canRepeat = _lastSlowAlert == null || t - _lastSlowAlert! >= cfg.slowFpsRepeatS;
      if (slowFor >= cfg.slowFpsSeconds && canRepeat) {
        _lastSlowAlert = t;
        events.add(HealthEvent.slow);
      }
    } else {
      _slowSince = null;
    }
    return events;
  }

  /// A battery reading in percent (0-100).
  List<HealthEvent> onBattery(int percent) {
    if (percent > cfg.batteryLowPercent + 5) {
      _batteryLowSaid = false;
      _batteryCriticalSaid = false;
      return const [];
    }
    if (percent <= cfg.batteryCriticalPercent && !_batteryCriticalSaid) {
      _batteryCriticalSaid = true;
      _batteryLowSaid = true;
      return [HealthEvent.batteryCritical];
    }
    if (percent <= cfg.batteryLowPercent && !_batteryLowSaid) {
      _batteryLowSaid = true;
      return [HealthEvent.batteryLow];
    }
    return const [];
  }
}
