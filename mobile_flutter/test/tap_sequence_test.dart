// The tap counter behind double-tap (start/stop) and triple-tap (debug view).
// `fakeAsync` lets the test move a fake clock instead of really waiting (like Jest's
// fake timers).

import 'package:clear_vision/ui/tap_sequence.dart';
import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('single, double and triple taps are told apart', () {
    fakeAsync((clock) {
      final seen = <int>[];
      final taps = TapSequence(onSequence: seen.add);
      taps.tap();
      clock.elapse(const Duration(milliseconds: 400));
      taps
        ..tap()
        ..tap();
      clock.elapse(const Duration(milliseconds: 400));
      taps.tap();
      clock.elapse(const Duration(milliseconds: 200));
      taps.tap();
      clock.elapse(const Duration(milliseconds: 200));
      taps.tap();
      clock.elapse(const Duration(milliseconds: 400));
      expect(seen, [1, 2, 3]);
    });
  });

  test('taps further apart than the gap are separate', () {
    fakeAsync((clock) {
      final seen = <int>[];
      final taps = TapSequence(onSequence: seen.add);
      taps.tap();
      clock.elapse(const Duration(milliseconds: 360));
      taps.tap();
      clock.elapse(const Duration(milliseconds: 360));
      expect(seen, [1, 1]);
    });
  });
}
