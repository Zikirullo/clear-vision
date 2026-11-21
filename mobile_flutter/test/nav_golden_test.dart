// The Dart navigation rules must reproduce every case in tests/golden/nav_cases.json
// exactly (frame, time, text, priority, cooldown key), with the numbers from
// models/nav_config.json. Both files are read from the repository root (one level up),
// so this test always checks against the same files as Python's tests/test_nav_golden.py.
//
// Run: cd mobile_flutter && flutter test
//
// Dart test notes:
// - `test('name', () { ... })` = Jest's `test()`. `group()` = `describe()`.
// - `expect(actual, matcher)` = Jest's `expect(actual).toEqual(...)`.

import 'dart:convert';
import 'dart:io';

import 'package:clear_vision/navigation/detection.dart';
import 'package:clear_vision/navigation/nav_config.dart';
import 'package:clear_vision/navigation/planner.dart';
import 'package:flutter_test/flutter_test.dart';

/// `flutter test` runs with the package folder (mobile_flutter/) as working directory.
final repoRoot = Directory.current.parent;

Map<String, dynamic> readJson(String relative) =>
    jsonDecode(File('${repoRoot.path}/$relative').readAsStringSync()) as Map<String, dynamic>;

/// Replays one case like scripts/make_nav_golden.py run_case(): fresh state, every
/// planned message counts as spoken at its frame time.
List<Map<String, Object>> runCase(Map<String, dynamic> c, NavConfig nav) {
  final planner = SpeechPlanner(nav)..muteNonHazards = c['mute_non_hazards'] as bool;
  // `..` is the cascade operator: call/set on the same object and return the object.
  final out = <Map<String, Object>>[];
  final frames = c['frames'] as List;
  for (var i = 0; i < frames.length; i++) {
    final frame = frames[i] as Map<String, dynamic>;
    final t = (frame['t'] as num).toDouble();
    final dets = [
      for (final d in frame['detections'] as List)
        Detection(
          d['class'] as String,
          (d['conf'] as num).toDouble(),
          ((d['box'] as List)[0] as num).toDouble(),
          ((d['box'] as List)[1] as num).toDouble(),
          ((d['box'] as List)[2] as num).toDouble(),
          ((d['box'] as List)[3] as num).toDouble(),
        ),
    ];
    final plan = planner.update(dets, t);
    final msg = plan.message;
    if (msg != null) {
      planner.markSpoken(msg, t);
      out.add({'frame': i, 't': t, 'text': msg.text, 'priority': msg.priority.name, 'key': msg.key});
    }
  }
  return out;
}

void main() {
  final golden = readJson('tests/golden/nav_cases.json');
  final nav = ClearVisionConfig.fromJson(readJson('models/nav_config.json')).nav;
  final cases = (golden['cases'] as List).cast<Map<String, dynamic>>();

  test('golden file is the expected one', () {
    expect(golden['format_version'], 1);
    expect(cases.length, 35);
  });

  test('the app copy of nav_config.json matches models/nav_config.json', () {
    final appCopy = jsonDecode(File('assets/nav_config.json').readAsStringSync());
    expect(appCopy, readJson('models/nav_config.json'),
        reason: 'run: python scripts/sync_mobile_assets.py --config-only');
  });

  group('Dart matches Python on every golden case', () {
    for (final c in cases) {
      test(c['id'] as String, () {
        final expected = [
          for (final e in c['expected'] as List)
            {
              'frame': e['frame'],
              't': (e['t'] as num).toDouble(),
              'text': e['text'],
              'priority': e['priority'],
              'key': e['key'],
            },
        ];
        expect(runCase(c, nav), expected, reason: c['description'] as String);
      });
    }
  });
}
