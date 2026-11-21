// Per-session speech state around the pure rules: the last frames' candidates, when
// each message was last spoken, and the hazards-only switch. Port of src/planner.py.
//
// Dart notes:
// - A class with mutable fields (no `final`) holds state, like a small JS class.
// - `_name` (leading underscore) = private to this file (Dart has no `private` keyword).

import 'dart:math' as math;

import 'detection.dart';
import 'nav_config.dart';
import 'navigator.dart';

class SpeechPlanner {
  SpeechPlanner(this.cfg) : muteNonHazards = cfg.muteNonHazards;
  // ^ the part after `:` is an "initializer list": sets fields before the body runs.

  NavConfig cfg;
  bool muteNonHazards;
  final List<List<Detection>> history = [];
  final Map<String, double> lastSpoken = {};

  /// How many earlier frames to keep: enough for confirmation and the naming window.
  int get historyLength => math.max(1, math.max(cfg.windowFrames - 1, cfg.nameHistoryFrames - 1));

  /// Plan this frame and remember its candidates for the next frames.
  SpeechPlan update(List<Detection> detections, double now) {
    final plan = planSpeech(detections, history, now, lastSpoken, cfg, muteNonHazards: muteNonHazards);
    history.add(plan.candidates);
    while (history.length > historyLength) {
      history.removeAt(0);
    }
    return plan;
  }

  void markSpoken(Message msg, double now) => lastSpoken[msg.key] = now;

  bool toggleMuteNonHazards() => muteNonHazards = !muteNonHazards;

  /// Forget everything (after a stop/start, old frames must not confirm new objects).
  void reset() {
    history.clear();
    lastSpoken.clear();
  }
}
