// Unit tests for the phone-only logic: health checks, vibration patterns, the long-press
// scene description, config loading and the class-threshold filter.

import 'dart:convert';
import 'dart:io';

import 'package:clear_vision/navigation/detection.dart';
import 'package:clear_vision/navigation/haptics_pattern.dart';
import 'package:clear_vision/navigation/health.dart';
import 'package:clear_vision/navigation/nav_config.dart';
import 'package:clear_vision/navigation/navigator.dart';
import 'package:clear_vision/navigation/planner.dart';
import 'package:flutter_test/flutter_test.dart';

final config = ClearVisionConfig.fromJson(
    jsonDecode(File('assets/nav_config.json').readAsStringSync()) as Map<String, dynamic>);

/// Feed frames at `fps` from `from` to `to` seconds, collecting health events.
List<HealthEvent> feed(HealthMonitor h, double from, double to, double fps) {
  final events = <HealthEvent>[];
  for (var t = from; t < to; t += 1 / fps) {
    events.addAll(h.onFrame(t));
    events.addAll(h.tick(t));
  }
  return events;
}

/// Ticks with no frames (camera stalled).
List<HealthEvent> silence(HealthMonitor h, double from, double to) {
  final events = <HealthEvent>[];
  for (var t = from; t < to; t += 0.25) {
    events.addAll(h.tick(t));
  }
  return events;
}

void main() {
  group('config', () {
    test('loads the app numbers from nav_config.json', () {
      expect(config.app.staleTimeout, 1.5);
      expect(config.app.slowFps, 8);
      expect(config.app.slowFpsSeconds, 3);
      expect(config.classes.length, 20);
      expect(config.pluginThreshold, 0.15);
      expect(config.detectionThreshold('curb'), 0.15);
      expect(config.detectionThreshold('car'), 0.25);
      expect(config.nav.nameHistoryFrames, 5);
    });

    test('system phrases fill placeholders and survive missing keys', () {
      final p = config.nav.phrases;
      expect(p.systemText('battery_low', {'percent': 18}), 'Battery low, 18 percent');
      expect(p.systemText('camera_lost'), 'Camera lost');
      expect(p.systemText('no_such_key'), 'no such key');
    });

    test('a language file only overrides what it translates', () {
      final ko = config.nav.phrases.merged({
        'ahead': '앞에 {name}',
        'system': {'camera_lost': '카메라 연결 끊김'},
      });
      expect(ko.ahead, '앞에 {name}');
      expect(ko.systemText('camera_lost'), '카메라 연결 끊김');
      expect(ko.systemText('camera_back'), 'Camera back');
    });
  });

  group('health: camera lost', () {
    test('a 1 s hiccup stays silent', () {
      final h = HealthMonitor(config.app)..start(0);
      final events = [...feed(h, 0, 2, 20), ...silence(h, 2, 3), ...feed(h, 3, 5, 20)];
      expect(events, isEmpty);
    });

    test('1.5 s without frames: Camera lost once, Camera back once', () {
      final h = HealthMonitor(config.app)..start(0);
      final events = [...feed(h, 0, 2, 20), ...silence(h, 2, 6), ...feed(h, 6, 8, 20)];
      expect(events, [HealthEvent.cameraLost, HealthEvent.cameraBack]);
    });

    test('no frame at all after start is a lost camera too', () {
      final h = HealthMonitor(config.app)..start(10);
      expect(silence(h, 10, 11.4), isEmpty);
      expect(silence(h, 11.4, 12), [HealthEvent.cameraLost]);
    });

    test('stopped: no alerts', () {
      final h = HealthMonitor(config.app)..start(0);
      feed(h, 0, 1, 20);
      h.stop();
      expect(silence(h, 1, 10), isEmpty);
    });
  });

  group('health: slow detection', () {
    test('below 8 fps for 3 s is reported once, then not again within 60 s', () {
      final h = HealthMonitor(config.app)..start(0);
      final events = feed(h, 0, 10, 5);
      expect(events, [HealthEvent.slow]);
      expect(feed(h, 10, 50, 5), isEmpty);
      expect(feed(h, 50, 80, 5), [HealthEvent.slow]);
    });

    test('short slow spells are ignored', () {
      final h = HealthMonitor(config.app)..start(0);
      final events = [...feed(h, 0, 2, 20), ...feed(h, 2, 4, 5), ...feed(h, 4, 8, 20)];
      expect(events, isEmpty);
    });

    test('15 fps is fine', () {
      final h = HealthMonitor(config.app)..start(0);
      expect(feed(h, 0, 20, 15), isEmpty);
    });
  });

  group('health: battery', () {
    test('low once, critical once, re-armed after charging', () {
      final h = HealthMonitor(config.app);
      expect(h.onBattery(50), isEmpty);
      expect(h.onBattery(20), [HealthEvent.batteryLow]);
      expect(h.onBattery(15), isEmpty);
      expect(h.onBattery(10), [HealthEvent.batteryCritical]);
      expect(h.onBattery(8), isEmpty);
      expect(h.onBattery(30), isEmpty);
      expect(h.onBattery(19), [HealthEvent.batteryLow]);
    });
  });

  group('haptics', () {
    final hc = config.app.haptics;
    test('pulses tell the side', () {
      int pulses(List<int> p) => (p.length - 1 + 1) ~/ 2; // [0, on, off, on, ...]
      expect(pulses(hapticPattern(Direction.left, Proximity.mid, hc)), 1);
      expect(pulses(hapticPattern(Direction.right, Proximity.mid, hc)), 2);
      expect(pulses(hapticPattern(Direction.ahead, Proximity.mid, hc)), 3);
    });

    test('closer = faster', () {
      expect(hapticPattern(Direction.right, Proximity.near, hc), [0, 60, 60, 60]);
      expect(hapticPattern(Direction.right, Proximity.far, hc), [0, 120, 300, 120]);
    });
  });

  group('describe scene (long press)', () {
    final nav = config.nav;
    test('nothing', () {
      expect(describeScene([], [], nav, 3), 'Nothing detected');
    });

    test('riskiest first, at most 3, naming rule applied', () {
      const stairs = Detection('stairs', 0.8, 0.4, 0.4, 0.6, 0.95);
      const pole = Detection('pole', 0.9, 0.05, 0.5, 0.1, 0.9);
      const cone = Detection('traffic_cone', 0.5, 0.8, 0.6, 0.9, 0.9); // < 0.6: Obstacle
      const bench = Detection('bench', 0.9, 0.7, 0.7, 0.9, 0.8);
      expect(describeScene([bench, pole, cone, stairs], [], nav, 3),
          'Stairs ahead, close. Pole, left. Obstacle, right.');
    });
  });

  group('planner', () {
    test('keeps enough history for both windows and resets', () {
      final p = SpeechPlanner(config.nav);
      expect(p.historyLength, 4);
      const s = Detection('stairs', 0.8, 0.4, 0.4, 0.6, 0.95);
      for (var i = 0; i < 10; i++) {
        p.update([s], i / 10);
      }
      expect(p.history.length, 4);
      p.reset();
      expect(p.history, isEmpty);
      expect(p.lastSpoken, isEmpty);
    });
  });
}
