// Counts quick taps so one button can tell single, double and triple taps apart.
// (Flutter's GestureDetector has double tap but no triple tap, and its double tap
// would swallow the third tap.)
//
// Taps less than `gap` apart form one sequence; when the gap passes, `onSequence(count)`
// fires once with the number of taps.

import 'dart:async';

class TapSequence {
  TapSequence({required this.onSequence, this.gap = const Duration(milliseconds: 350)});

  final void Function(int taps) onSequence;
  final Duration gap;
  int _count = 0;
  Timer? _timer;

  void tap() {
    _count++;
    _timer?.cancel();
    _timer = Timer(gap, () {
      final n = _count;
      _count = 0;
      onSequence(n);
    });
  }

  void dispose() => _timer?.cancel();
}
