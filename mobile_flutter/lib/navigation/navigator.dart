// Navigation rules: turn one frame's detections into at most one short spoken warning.
//
// A line-by-line port of Python's src/navigator.py, specified in docs/navigation_spec.md.
// Every function here is pure (no camera, no audio, no clock), so test/nav_golden_test.dart
// can replay tests/golden/nav_cases.json and demand the exact same messages as Python.
// If you change a rule, change Python first, regenerate the golden file, then port it here.
//
// Dart notes:
// - Top-level functions (outside any class) are normal in Dart, like exported functions
//   in a JS module. Files are libraries; `import` pulls them in.
// - `enum Direction { left, ahead, right }` is a real type (safer than JS strings).
//   `.name` gives the string ("left").
// - `Iterable` is a lazy sequence (like a JS iterator); `List` is an array.
//   `.where(...)` = `.filter(...)`, `.map(...)` = `.map(...)`, `.any(...)` = `.some(...)`.
// - `String?` means "String or null". `??` = JS `??`. `!` asserts non-null.
// - Records: `(double, Detection)` is a tuple-like value; `$1`/`$2` read its parts.

import 'dart:math' as math;

import 'detection.dart';
import 'nav_config.dart';

enum Direction { left, ahead, right }

enum Proximity { near, mid, far }

enum Priority { urgent, normal }

/// A warning to speak. `key` identifies it for cooldowns ("car:ahead", "post:left").
/// `direction` and `proximity` are extra (not in Python's Message): they drive vibration.
class Message {
  const Message({
    required this.text,
    required this.priority,
    required this.key,
    required this.direction,
    required this.proximity,
  });

  final String text;
  final Priority priority;
  final String key;
  final Direction direction;
  final Proximity proximity;

  @override
  String toString() => 'Message("$text", ${priority.name}, $key)';
}

// ------------------------------------------------------------------ geometry (spec 2)

Direction direction(Detection det, NavConfig cfg) {
  if (det.centerX < cfg.leftBoundary) return Direction.left;
  if (det.centerX > cfg.rightBoundary) return Direction.right;
  return Direction.ahead; // boundaries inclusive
}

Proximity proximity(Detection det, NavConfig cfg) {
  if (det.height > cfg.nearHeight) return Proximity.near;
  if (det.height >= cfg.midHeight) return Proximity.mid;
  return Proximity.far;
}

double severity(String className, NavConfig cfg) => cfg.severity[className] ?? cfg.defaultSeverity;

/// `proximity_weight * direction_weight * severity` (spec 5), in Python's order of
/// operations so the floating-point result is identical.
double risk(Detection det, NavConfig cfg) {
  final proxWeight = cfg.proximityWeights[proximity(det, cfg).name]!;
  final dirWeight = direction(det, cfg) == Direction.ahead ? cfg.aheadWeight : cfg.sideWeight;
  return proxWeight * dirWeight * severity(det.className, cfg);
}

/// The name group a class belongs to ("pole" -> "post"), or the class itself.
String classGroup(String className, NavConfig cfg) {
  // `for (final e in map.entries)` = `for (const [k, v] of Object.entries(obj))`.
  for (final e in cfg.nameGroups.entries) {
    if (e.value.contains(className)) return e.key;
  }
  return className;
}

double boxIou(Detection a, Detection b) {
  final ix = math.max(0.0, math.min(a.x2, b.x2) - math.max(a.x1, b.x1));
  final iy = math.max(0.0, math.min(a.y2, b.y2) - math.max(a.y1, b.y1));
  final inter = ix * iy;
  final union = a.area + b.area - inter;
  return union > 0 ? inter / union : 0.0;
}

/// Cooldown key: group + direction ("post:left").
String messageKey(Detection det, NavConfig cfg) => '${classGroup(det.className, cfg)}:${direction(det, cfg).name}';

/// Spoken name: `spoken_names` entry, else "traffic_cone" -> "Traffic cone".
String displayName(String className, NavConfig cfg) {
  final custom = cfg.spokenNames[className];
  if (custom != null && custom.isNotEmpty) return custom;
  final words = className.replaceAll('_', ' ');
  if (words.isEmpty) return words;
  return words[0].toUpperCase() + words.substring(1).toLowerCase();
}

// ------------------------------------------------------------------ speech filter (spec 3, 4)

double speakThreshold(String className, NavConfig cfg) => cfg.speakConf[className] ?? cfg.speakConfDefault;

bool confidentEnough(Detection det, NavConfig cfg) => det.confidence >= speakThreshold(det.className, cfg);

/// Near enough to matter: vehicles by height; others by bottom edge or height.
bool inSpeakingZone(Detection det, NavConfig cfg) {
  if (cfg.vehicles.contains(det.className)) return det.height > cfg.vehicleMinHeight;
  return det.y2 >= cfg.horizonBottom || det.height > cfg.minSpeakHeight;
}

bool allowedClass(Detection det, bool muteNonHazards, NavConfig cfg) =>
    !muteNonHazards || cfg.hazardClasses.contains(det.className);

/// Same group, centers close, boxes overlapping (spec 4).
bool sameObject(Detection a, Detection b, NavConfig cfg) {
  if (classGroup(a.className, cfg) != classGroup(b.className, cfg)) return false;
  final dx = a.centerX - b.centerX;
  final dy = a.centerY - b.centerY;
  // Python writes `** 0.5`; math.pow(x, 0.5) keeps the exact same result.
  if (math.pow(dx * dx + dy * dy, 0.5) > cfg.confirmCenterDist) return false;
  return boxIou(a, b) >= cfg.confirmIou;
}

/// `det` and its matches in the last `frames - 1` history frames, newest first.
/// `history` holds earlier frames' candidates, oldest first.
List<Detection> objectTrack(Detection det, List<List<Detection>> history, int frames, NavConfig cfg) {
  final track = <Detection>[det]; // `<Detection>[]` = a typed empty array
  var last = det; // `var` = type inferred (like `let`)
  final previous = frames > 1 ? history.sublist(math.max(0, history.length - (frames - 1))) : <List<Detection>>[];
  for (final frame in previous.reversed) {
    Detection? match;
    for (final d in frame) {
      if (sameObject(last, d, cfg)) {
        match = d;
        break; // the first match in list order
      }
    }
    if (match != null) {
      track.add(match);
      last = match;
    }
  }
  return track;
}

/// Seen in at least `confirm_frames` of the last `window_frames` frames (3 of 5).
bool isConfirmed(Detection det, List<List<Detection>> history, NavConfig cfg) =>
    objectTrack(det, history, cfg.windowFrames, cfg).length >= cfg.confirmFrames;

// ------------------------------------------------------------------ naming rule (spec 6)

/// The same object carried more than one class name in the last `name_history_frames` frames.
bool nameChangedRecently(Detection det, List<List<Detection>> history, NavConfig cfg) {
  final track = objectTrack(det, history, cfg.nameHistoryFrames, cfg);
  return track.map((d) => d.className).toSet().length > 1;
}

/// True if a group member should be spoken as "Obstacle": low confidence, a different
/// group class overlapping it in this frame, or a class change in the last few frames.
bool nameIsDoubtful(Detection det, Iterable<Detection> context, NavConfig cfg,
    [List<List<Detection>> history = const []]) {
  final group = classGroup(det.className, cfg);
  if (group == det.className) return false; // not in a name group
  if (det.confidence < cfg.nameMinConf) return true;
  final overlapping = context.any((other) =>
      other.className != det.className &&
      classGroup(other.className, cfg) == group &&
      boxIou(det, other) >= cfg.nameOverlapIou);
  if (overlapping) return true;
  return nameChangedRecently(det, history, cfg);
}

// ------------------------------------------------------------------ phrases and choice (spec 5, 7)

String buildPhrase(String className, Direction where, Proximity howClose, NavConfig cfg, [String? name]) {
  final p = cfg.phrases;
  final spoken = name ?? displayName(className, cfg);
  var text = where == Direction.ahead
      ? p.ahead.replaceAll('{name}', spoken)
      : p.side.replaceAll('{name}', spoken).replaceAll('{direction}', p.directions[where.name] ?? where.name);
  if (howClose == Proximity.near) text += p.closeSuffix;
  return text;
}

bool isOnCooldown(String key, Priority priority, double now, Map<String, double> lastSpoken, NavConfig cfg) {
  final last = lastSpoken[key];
  if (last == null) return false;
  final wait = priority == Priority.urgent ? cfg.urgentCooldown : cfg.cooldown;
  return now - last < wait;
}

/// Detections sorted by risk, highest first. STABLE: equal risks keep list order.
/// (Dart's List.sort is not guaranteed stable, so the original index breaks ties.)
List<(double, Detection)> _byRisk(List<Detection> detections, NavConfig cfg) {
  final scored = [for (var i = 0; i < detections.length; i++) (risk(detections[i], cfg), i, detections[i])];
  scored.sort((a, b) {
    final byRisk = b.$1.compareTo(a.$1);
    return byRisk != 0 ? byRisk : a.$2.compareTo(b.$2);
  });
  return [for (final s in scored) (s.$1, s.$3)]; // collection-for, like [...].map in JS
}

/// The single most dangerous object worth announcing, or null. Pure: `lastSpoken` is
/// not modified; the caller records `lastSpoken[msg.key] = now` once it is spoken.
Message? chooseMessage(
  List<Detection> detections,
  double now,
  Map<String, double> lastSpoken,
  NavConfig cfg, {
  List<Detection>? context,
  List<List<Detection>> history = const [],
}) {
  final ctx = context ?? detections;
  for (final (score, det) in _byRisk(detections, cfg)) {
    if (score < cfg.riskThreshold) break;
    final where = direction(det, cfg);
    final howClose = proximity(det, cfg);
    final priority = where == Direction.ahead && howClose == Proximity.near ? Priority.urgent : Priority.normal;
    final key = messageKey(det, cfg);
    if (isOnCooldown(key, priority, now, lastSpoken, cfg)) continue; // next riskiest object
    final name = nameIsDoubtful(det, ctx, cfg, history) ? cfg.phrases.unknown : null;
    return Message(
      text: buildPhrase(det.className, where, howClose, cfg, name),
      priority: priority,
      key: key,
      direction: where,
      proximity: howClose,
    );
  }
  return null;
}

/// What happened to one frame's detections.
class SpeechPlan {
  const SpeechPlan(this.candidates, this.speakable, this.message);

  final List<Detection> candidates; // confident enough to be spoken
  final List<Detection> speakable; // ... and confirmed, near enough, not muted
  final Message? message;
}

/// Filter one frame for speech and pick at most one message (spec 3).
/// Pure: the caller appends `plan.candidates` to `history` afterwards.
SpeechPlan planSpeech(
  List<Detection> detections,
  List<List<Detection>> history,
  double now,
  Map<String, double> lastSpoken,
  NavConfig cfg, {
  bool? muteNonHazards,
}) {
  final mute = muteNonHazards ?? cfg.muteNonHazards;
  final candidates = detections.where((d) => confidentEnough(d, cfg)).toList();
  final speakable = candidates
      .where((d) => inSpeakingZone(d, cfg) && allowedClass(d, mute, cfg) && isConfirmed(d, history, cfg))
      .toList();
  final message = chooseMessage(speakable, now, lastSpoken, cfg, context: candidates, history: history);
  return SpeechPlan(candidates, speakable, message);
}

// ------------------------------------------------------------------ phone app extras

/// Long press "what is in front": the riskiest confident objects of the latest frame,
/// at most `maxObjects`, as one sentence ("Stairs ahead, close. Pole, left."). No
/// confirmation or cooldown: the user asked. The naming rule still applies.
String describeScene(List<Detection> candidates, List<List<Detection>> history, NavConfig cfg, int maxObjects) {
  if (candidates.isEmpty) return cfg.phrases.systemText('nothing_detected');
  final parts = <String>[];
  for (final (_, det) in _byRisk(candidates, cfg).take(maxObjects)) {
    final name = nameIsDoubtful(det, candidates, cfg, history) ? cfg.phrases.unknown : null;
    parts.add(buildPhrase(det.className, direction(det, cfg), proximity(det, cfg), cfg, name));
  }
  return '${parts.join('. ')}.';
}
