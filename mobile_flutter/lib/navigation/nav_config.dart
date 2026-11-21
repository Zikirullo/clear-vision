// The numbers and phrases of the navigation rules, loaded from assets/nav_config.json.
//
// That file is a copy of models/nav_config.json, generated from configs/default.yaml by
// scripts/export_nav_config.py and copied here by scripts/sync_mobile_assets.py.
// Nothing is hard-coded: change the YAML, re-export, re-sync.
//
// Dart notes:
// - `Map<String, dynamic>` is what `jsonDecode` returns for a JSON object
//   (like a plain JS object; `dynamic` = "any type, checked at runtime").
// - `factory NavConfig.fromJson(...)` is a named constructor that can run code before
//   creating the object (like a static `fromJson()` helper in JS).
// - `as num` is a type cast; JSON numbers can be int or double, so we read `num` and
//   call `.toDouble()`.
// - `List<String>.from(x)` copies a JSON list into a typed list.

double _d(Object? v) => (v as num).toDouble();

Map<String, double> _doubleMap(Object? v) =>
    (v as Map<String, dynamic>? ?? const {}).map((k, val) => MapEntry(k, _d(val)));

/// The `navigator` section: everything docs/navigation_spec.md refers to.
class NavConfig {
  NavConfig({
    required this.leftBoundary,
    required this.rightBoundary,
    required this.nearHeight,
    required this.midHeight,
    required this.proximityWeights,
    required this.aheadWeight,
    required this.sideWeight,
    required this.severity,
    required this.defaultSeverity,
    required this.riskThreshold,
    required this.cooldown,
    required this.urgentCooldown,
    required this.confirmFrames,
    required this.windowFrames,
    required this.confirmCenterDist,
    required this.confirmIou,
    required this.speakConf,
    required this.speakConfDefault,
    required this.horizonBottom,
    required this.minSpeakHeight,
    required this.vehicles,
    required this.vehicleMinHeight,
    required this.muteNonHazards,
    required this.hazardClasses,
    required this.nameGroups,
    required this.nameMinConf,
    required this.nameOverlapIou,
    required this.nameHistoryFrames,
    required this.phrases,
    required this.spokenNames,
  });

  // `required` named parameters: callers must write `NavConfig(leftBoundary: 0.333, ...)`,
  // like passing one props object in React, but checked by the compiler.
  factory NavConfig.fromJson(Map<String, dynamic> j) => NavConfig(
        leftBoundary: _d(j['left_boundary']),
        rightBoundary: _d(j['right_boundary']),
        nearHeight: _d(j['near_height']),
        midHeight: _d(j['mid_height']),
        proximityWeights: _doubleMap(j['proximity_weights']),
        aheadWeight: _d(j['ahead_weight']),
        sideWeight: _d(j['side_weight']),
        severity: _doubleMap(j['severity']),
        defaultSeverity: _d(j['default_severity']),
        riskThreshold: _d(j['risk_threshold']),
        cooldown: _d(j['cooldown']),
        urgentCooldown: _d(j['urgent_cooldown']),
        confirmFrames: (j['confirm_frames'] as num).toInt(),
        windowFrames: (j['window_frames'] as num).toInt(),
        confirmCenterDist: _d(j['confirm_center_dist']),
        confirmIou: _d(j['confirm_iou']),
        speakConf: _doubleMap(j['speak_conf']),
        speakConfDefault: _d(j['speak_conf_default']),
        horizonBottom: _d(j['horizon_bottom']),
        minSpeakHeight: _d(j['min_speak_height']),
        vehicles: List<String>.from(j['vehicles'] as List),
        vehicleMinHeight: _d(j['vehicle_min_height']),
        muteNonHazards: j['mute_non_hazards'] as bool,
        hazardClasses: List<String>.from(j['hazard_classes'] as List),
        nameGroups: (j['name_groups'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, List<String>.from(v as List))),
        nameMinConf: _d(j['name_min_conf']),
        nameOverlapIou: _d(j['name_overlap_iou']),
        // Older config files had no class-change check: 1 frame = off.
        nameHistoryFrames: (j['name_history_frames'] as num? ?? 1).toInt(),
        phrases: Phrases.fromJson(j['phrases'] as Map<String, dynamic>),
        spokenNames: Map<String, String>.from(j['spoken_names'] as Map? ?? const {}),
      );

  final double leftBoundary;
  final double rightBoundary;
  final double nearHeight;
  final double midHeight;
  final Map<String, double> proximityWeights;
  final double aheadWeight;
  final double sideWeight;
  final Map<String, double> severity;
  final double defaultSeverity;
  final double riskThreshold;
  final double cooldown;
  final double urgentCooldown;
  final int confirmFrames;
  final int windowFrames;
  final double confirmCenterDist;
  final double confirmIou;
  final Map<String, double> speakConf;
  final double speakConfDefault;
  final double horizonBottom;
  final double minSpeakHeight;
  final List<String> vehicles;
  final double vehicleMinHeight;
  final bool muteNonHazards;
  final List<String> hazardClasses;
  final Map<String, List<String>> nameGroups;
  final double nameMinConf;
  final double nameOverlapIou;
  final int nameHistoryFrames;
  final Phrases phrases;
  final Map<String, String> spokenNames;

  /// A copy with other phrases (a language pack), everything else unchanged.
  NavConfig withPhrases(Phrases p, Map<String, String> names) => NavConfig(
        leftBoundary: leftBoundary, rightBoundary: rightBoundary, nearHeight: nearHeight,
        midHeight: midHeight, proximityWeights: proximityWeights, aheadWeight: aheadWeight,
        sideWeight: sideWeight, severity: severity, defaultSeverity: defaultSeverity,
        riskThreshold: riskThreshold, cooldown: cooldown, urgentCooldown: urgentCooldown,
        confirmFrames: confirmFrames, windowFrames: windowFrames,
        confirmCenterDist: confirmCenterDist, confirmIou: confirmIou, speakConf: speakConf,
        speakConfDefault: speakConfDefault, horizonBottom: horizonBottom,
        minSpeakHeight: minSpeakHeight, vehicles: vehicles, vehicleMinHeight: vehicleMinHeight,
        muteNonHazards: muteNonHazards, hazardClasses: hazardClasses, nameGroups: nameGroups,
        nameMinConf: nameMinConf, nameOverlapIou: nameOverlapIou,
        nameHistoryFrames: nameHistoryFrames, phrases: p, spokenNames: names,
      );
}

/// Spoken text templates (spec section 7). `{name}`, `{direction}`, `{percent}` are
/// filled in by the code.
class Phrases {
  const Phrases({
    required this.ahead,
    required this.side,
    required this.closeSuffix,
    required this.unknown,
    required this.directions,
    required this.system,
  });

  factory Phrases.fromJson(Map<String, dynamic> j) => Phrases(
        ahead: j['ahead'] as String,
        side: j['side'] as String,
        closeSuffix: j['close_suffix'] as String,
        unknown: j['unknown'] as String,
        directions: Map<String, String>.from(j['directions'] as Map),
        system: Map<String, String>.from(j['system'] as Map),
      );

  final String ahead;
  final String side;
  final String closeSuffix;
  final String unknown;
  final Map<String, String> directions;
  final Map<String, String> system;

  /// A system message by key ("camera_lost"), with `{placeholders}` filled in.
  /// Unknown keys fall back to the key itself, so a missing phrase is still spoken.
  String systemText(String key, [Map<String, Object> values = const {}]) {
    var text = system[key] ?? key.replaceAll('_', ' ');
    values.forEach((k, v) => text = text.replaceAll('{$k}', '$v'));
    return text;
  }

  /// Phrases with some keys replaced (a language file only lists what it translates).
  Phrases merged(Map<String, dynamic> j) => Phrases(
        ahead: j['ahead'] as String? ?? ahead,
        side: j['side'] as String? ?? side,
        closeSuffix: j['close_suffix'] as String? ?? closeSuffix,
        unknown: j['unknown'] as String? ?? unknown,
        directions: {...directions, ...Map<String, String>.from(j['directions'] as Map? ?? const {})},
        system: {...system, ...Map<String, String>.from(j['system'] as Map? ?? const {})},
      );
}

/// Vibration timing per warning (the `mobile.haptics` config).
class HapticsConfig {
  const HapticsConfig({required this.pulses, required this.onMs, required this.offMs});

  factory HapticsConfig.fromJson(Map<String, dynamic> j) => HapticsConfig(
        pulses: (j['pulses'] as Map<String, dynamic>).map((k, v) => MapEntry(k, (v as num).toInt())),
        onMs: (j['on_ms'] as Map<String, dynamic>).map((k, v) => MapEntry(k, (v as num).toInt())),
        offMs: (j['off_ms'] as Map<String, dynamic>).map((k, v) => MapEntry(k, (v as num).toInt())),
      );

  final Map<String, int> pulses; // by direction
  final Map<String, int> onMs; // by proximity
  final Map<String, int> offMs; // by proximity
}

/// The phone app's own numbers (`app` in nav_config.json = `mobile:` in the YAML).
class AppConfig {
  const AppConfig({
    required this.staleTimeout,
    required this.slowFps,
    required this.slowFpsSeconds,
    required this.slowFpsRepeatS,
    required this.batteryLowPercent,
    required this.batteryCriticalPercent,
    required this.batteryCheckS,
    required this.speechRate,
    required this.sceneMaxObjects,
    required this.debugMessages,
    required this.haptics,
  });

  factory AppConfig.fromJson(Map<String, dynamic> j) => AppConfig(
        staleTimeout: _d(j['stale_timeout']),
        slowFps: _d(j['slow_fps']),
        slowFpsSeconds: _d(j['slow_fps_seconds']),
        slowFpsRepeatS: _d(j['slow_fps_repeat_s']),
        batteryLowPercent: (j['battery_low_percent'] as num).toInt(),
        batteryCriticalPercent: (j['battery_critical_percent'] as num).toInt(),
        batteryCheckS: _d(j['battery_check_s']),
        speechRate: _d(j['speech_rate']),
        sceneMaxObjects: (j['scene_max_objects'] as num).toInt(),
        debugMessages: (j['debug_messages'] as num).toInt(),
        haptics: HapticsConfig.fromJson(j['haptics'] as Map<String, dynamic>),
      );

  final double staleTimeout;
  final double slowFps;
  final double slowFpsSeconds;
  final double slowFpsRepeatS;
  final int batteryLowPercent;
  final int batteryCriticalPercent;
  final double batteryCheckS;
  final double speechRate;
  final int sceneMaxObjects;
  final int debugMessages;
  final HapticsConfig haptics;
}

/// The whole nav_config.json.
class ClearVisionConfig {
  const ClearVisionConfig({
    required this.classes,
    required this.nav,
    required this.app,
    required this.classConf,
    required this.defaultConf,
  });

  factory ClearVisionConfig.fromJson(Map<String, dynamic> j) {
    final detection = j['detection'] as Map<String, dynamic>;
    return ClearVisionConfig(
      classes: List<String>.from(j['classes'] as List),
      nav: NavConfig.fromJson(j['navigator'] as Map<String, dynamic>),
      app: AppConfig.fromJson(j['app'] as Map<String, dynamic>),
      classConf: _doubleMap(detection['class_conf']),
      defaultConf: _d(detection['default_conf']),
    );
  }

  final List<String> classes;
  final NavConfig nav;
  final AppConfig app;

  /// The model's per-class minimum confidence (applied before the navigation rules;
  /// the plugin itself only has one global threshold).
  final Map<String, double> classConf;
  final double defaultConf;

  double detectionThreshold(String className) => classConf[className] ?? defaultConf;

  /// The lowest per-class threshold: what the plugin must let through.
  double get pluginThreshold => classConf.values.fold(defaultConf, (a, b) => b < a ? b : a);
}
