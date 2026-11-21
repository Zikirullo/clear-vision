// Wires the pipeline together (the phone's version of Python's src/main.py):
//
//   YOLOView camera + model --onResult--> Detections --> SpeechPlanner --> speech + vibration
//                                     \--> HealthMonitor (camera lost / slow / battery)
//
// Flutter notes:
// - `AppController extends ChangeNotifier`: the screens rebuild when it calls
//   `notifyListeners()` (like a React context value changing).
// - `Timer.periodic` = JS `setInterval`; `timer.cancel()` = `clearInterval`.
// - Nothing here blocks: speech and vibration calls are fire-and-forget.

import 'dart:async';
import 'dart:math' as math;

import 'package:battery_plus/battery_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'navigation/detection.dart';
import 'navigation/health.dart';
import 'navigation/nav_config.dart';
import 'navigation/navigator.dart';
import 'navigation/planner.dart';
import 'services/assets.dart';
import 'services/haptics.dart';
import 'services/settings.dart';
import 'services/speech.dart';

/// One line of the debug screen's "last spoken" list.
class SpokenLine {
  const SpokenLine(this.t, this.text, this.urgent, {this.silent = false});
  final double t;
  final String text;
  final bool urgent;
  final bool silent; // silent mode: vibrated, not spoken
}

class AppController extends ChangeNotifier {
  AppController._(this.config, this.settings, this.modelPath, this.labels, this._nav)
      : planner = SpeechPlanner(_nav),
        health = HealthMonitor(config.app),
        haptics = HapticsService(config.app.haptics);

  /// Loads everything; the app shows a splash screen until this finishes.
  static Future<AppController> create() async {
    final config = await loadConfig();
    final settings = await Settings.load(
      defaultRate: config.app.speechRate,
      defaultHazardsOnly: config.nav.muteNonHazards,
    );
    final nav = await navForLanguage(config.nav, settings.language);
    final c = AppController._(config, settings, await findModelAsset(), await loadLabels(), nav);
    await c._init();
    return c;
  }

  final ClearVisionConfig config;
  final Settings settings;
  final String? modelPath; // null: no model bundled (spoken as "model failed")
  final List<String> labels;
  NavConfig _nav;
  final SpeechPlanner planner;
  final HealthMonitor health;
  final HapticsService haptics;
  final SpeechService speech = SpeechService();
  final YOLOViewController yolo = YOLOViewController();
  final Battery _battery = Battery();
  final Stopwatch _clock = Stopwatch()..start(); // monotonic seconds, like time.monotonic()

  Timer? _healthTimer;
  Timer? _batteryTimer;
  bool running = false;
  bool _pausedByLifecycle = false;
  bool _autoStarted = false;
  bool modelLoaded = false;
  String? modelError;
  String? labelWarning;

  // Debug screen data.
  List<YOLOResult> lastResults = const [];
  List<Detection> lastCandidates = const [];
  final List<SpokenLine> spoken = [];
  double inferenceMs = 0;
  double fps = 0;

  double get now => _clock.elapsedMicroseconds / 1e6;
  NavConfig get nav => _nav;
  Phrases get phrases => _nav.phrases;

  Future<void> _init() async {
    planner.muteNonHazards = settings.hazardsOnly;
    await speech.init(rate: settings.speechRate, locale: ttsLocales[settings.language] ?? 'en-US');
    await haptics.init();
    settings.addListener(_applySettings);
    _healthTimer = Timer.periodic(const Duration(milliseconds: 250), (_) => _tickHealth());
    _batteryTimer = Timer.periodic(
      Duration(milliseconds: (config.app.batteryCheckS * 1000).round()),
      (_) => _checkBattery(),
    );
    if (modelPath == null) {
      modelError = 'No model in assets/models/ (run scripts/sync_mobile_assets.py)';
      system('model_failed', urgent: true);
    }
  }

  Future<void> _applySettings() async {
    planner.muteNonHazards = settings.hazardsOnly;
    await speech.setRate(settings.speechRate);
    await speech.setLocale(ttsLocales[settings.language] ?? 'en-US');
    _nav = await navForLanguage(config.nav, settings.language);
    planner.cfg = _nav;
    notifyListeners();
  }

  // ---------------------------------------------------------------- start / stop

  void start() {
    if (running || modelPath == null) return;
    running = true;
    planner.reset();
    health.start(now);
    unawaited(WakelockPlus.enable()); // keep the screen (and camera) awake
    unawaited(yolo.resume());
    system('started');
    unawaited(_checkBattery());
    notifyListeners();
  }

  void stop() {
    if (!running) return;
    running = false;
    health.stop();
    unawaited(WakelockPlus.disable());
    unawaited(yolo.pause());
    system('stopped');
    notifyListeners();
  }

  void toggle() => running ? stop() : start();

  /// The app went to the background: iOS stops the camera, so don't report it as lost.
  void onBackground() {
    if (!running) return;
    _pausedByLifecycle = true;
    health.stop();
  }

  void onForeground() {
    if (!_pausedByLifecycle) return;
    _pausedByLifecycle = false;
    planner.reset();
    health.start(now);
  }

  // ---------------------------------------------------------------- per frame

  /// Called by YOLOView for every processed frame (also frames with no detections).
  void onResults(List<YOLOResult> results) {
    final t = now;
    for (final e in health.onFrame(t)) {
      _onHealth(e);
    }
    lastResults = results;
    fps = health.fps(t);
    if (!running) return;

    // Plugin results -> our Detection type, with the model's per-class thresholds
    // (the plugin applies only one global threshold, the lowest of them).
    final dets = <Detection>[
      for (final r in results)
        if (r.confidence >= config.detectionThreshold(r.className))
          Detection(
            r.className,
            r.confidence,
            r.normalizedBox.left.clamp(0.0, 1.0),
            r.normalizedBox.top.clamp(0.0, 1.0),
            r.normalizedBox.right.clamp(0.0, 1.0),
            r.normalizedBox.bottom.clamp(0.0, 1.0),
          ),
    ];
    final plan = planner.update(dets, t);
    lastCandidates = plan.candidates;
    final msg = plan.message;
    if (msg != null) _deliver(msg, t);
    notifyListeners();
  }

  void onMetrics(YOLOPerformanceMetrics m) => inferenceMs = m.inferenceMs;

  void _deliver(Message msg, double t) {
    final urgent = msg.priority == Priority.urgent;
    if (settings.mode == OutputMode.silent) {
      planner.markSpoken(msg, t); // vibrated: counts as said, so cooldowns apply
      if (settings.vibration) haptics.warn(msg);
      _log(SpokenLine(t, msg.text, urgent, silent: true));
      return;
    }
    if (speech.say(msg.text, urgent: urgent)) {
      planner.markSpoken(msg, t);
      if (settings.vibration) haptics.warn(msg);
      _log(SpokenLine(t, msg.text, urgent));
    }
  }

  /// Long press: say what is in front right now (riskiest objects first).
  void describe() {
    final text = running
        ? describeScene(lastCandidates, planner.history, _nav, config.app.sceneMaxObjects)
        : phrases.systemText('stopped');
    speech.say(text, urgent: true);
    _log(SpokenLine(now, text, false));
  }

  /// A system message by phrase key. Spoken in every mode (safety first).
  void system(String key, {bool urgent = false, Map<String, Object> values = const {}}) {
    final text = phrases.systemText(key, values);
    speech.say(text, urgent: urgent);
    if (urgent && settings.vibration) haptics.alert();
    _log(SpokenLine(now, text, urgent));
  }

  void _log(SpokenLine line) {
    spoken.insert(0, line);
    if (spoken.length > config.app.debugMessages) spoken.removeLast();
    notifyListeners();
  }

  // ---------------------------------------------------------------- health

  void _tickHealth() {
    for (final e in health.tick(now)) {
      _onHealth(e);
    }
  }

  Future<void> _checkBattery() async {
    try {
      final level = await _battery.batteryLevel;
      for (final e in health.onBattery(level)) {
        _onHealth(e, battery: level);
      }
    } catch (e) {
      debugPrint('Battery level unavailable: $e');
    }
  }

  void _onHealth(HealthEvent e, {int? battery}) {
    switch (e) {
      // Dart 3 `switch` on an enum must cover every case (the compiler checks).
      case HealthEvent.cameraLost:
        system('camera_lost', urgent: true);
      case HealthEvent.cameraBack:
        system('camera_back');
      case HealthEvent.slow:
        system('slow');
      case HealthEvent.batteryLow:
        system('battery_low', values: {'percent': battery ?? 0});
      case HealthEvent.batteryCritical:
        system('battery_critical', urgent: true, values: {'percent': battery ?? 0});
    }
  }

  // ---------------------------------------------------------------- model

  /// The model is ready: the first time, start right away (a blind user shouldn't have
  /// to find the button just to begin), later double-taps stop and start.
  Future<void> onModelLoaded() async {
    modelLoaded = true;
    modelError = null;
    if (!_autoStarted) {
      _autoStarted = true;
      start();
    }
    notifyListeners();
    // Check the model's class names against labels.txt: wrong names would be spoken.
    try {
      final info = await YOLO.inspectModel(modelPath!);
      final names = (info['labels'] as List?)?.map((e) => '$e').toList();
      if (names != null && names.isNotEmpty && !listEquals(names, labels)) {
        labelWarning = 'Model classes ${names.take(math.min(5, names.length)).join(', ')}... '
            'differ from labels.txt';
        system('labels_mismatch', urgent: true);
      }
    } catch (e) {
      debugPrint('inspectModel failed (labels not checked): $e');
    }
    notifyListeners();
  }

  void onModelError(Object error) {
    modelLoaded = false;
    modelError = '$error';
    system('model_failed', urgent: true);
    if (running) {
      running = false;
      health.stop();
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _batteryTimer?.cancel();
    settings.removeListener(_applySettings);
    unawaited(WakelockPlus.disable());
    unawaited(speech.stop());
    super.dispose();
  }
}
