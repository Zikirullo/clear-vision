// Loading the bundled files: nav_config.json, the model, labels.txt, phrase packs.
//
// The model is found by file type, not by name: on iPhone the one *.mlpackage.zip in
// assets/models/, on Android the one *.tflite. Swapping in v2 or v3 therefore means
// replacing that file (scripts/sync_mobile_assets.py --run v2 does it), no code change.
//
// Flutter notes:
// - "Assets" are files bundled into the app (declared in pubspec.yaml under `assets:`).
//   `rootBundle.loadString(path)` reads one, like `fetch('/file.json')` in a web app.
// - `AssetManifest` lists every bundled asset.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

import '../navigation/nav_config.dart';

const String navConfigAsset = 'assets/nav_config.json';
const String labelsAsset = 'assets/models/labels.txt';

Future<ClearVisionConfig> loadConfig() async =>
    ClearVisionConfig.fromJson(jsonDecode(await rootBundle.loadString(navConfigAsset)) as Map<String, dynamic>);

Future<List<String>> loadLabels() async => (await rootBundle.loadString(labelsAsset))
    .split('\n')
    .map((l) => l.trim())
    .where((l) => l.isNotEmpty)
    .toList();

/// The model asset for this phone's OS, or null if none is bundled.
/// If several are bundled (shouldn't happen), the last by name is used.
Future<String?> findModelAsset() async {
  final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final suffix = Platform.isIOS ? '.mlpackage.zip' : '.tflite';
  final models = manifest.listAssets().where((a) => a.startsWith('assets/models/') && a.endsWith(suffix)).toList()
    ..sort();
  return models.isEmpty ? null : models.last;
}

/// Phrase pack codes that exist (English is always there: it is nav_config.json itself).
Future<List<String>> availableLanguages() async {
  final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final packs = manifest
      .listAssets()
      .where((a) => a.startsWith('assets/phrases/') && a.endsWith('.json'))
      .map((a) => a.split('/').last.replaceAll('.json', ''));
  return {'en', ...packs}.toList();
}

/// The navigator config with a language's phrases and class names applied.
/// assets/phrases/<code>.json: {"phrases": {...only translated keys...}, "spoken_names": {"car": "..."}}
Future<NavConfig> navForLanguage(NavConfig base, String code) async {
  if (code == 'en') return base;
  try {
    final j = jsonDecode(await rootBundle.loadString('assets/phrases/$code.json')) as Map<String, dynamic>;
    return base.withPhrases(
      base.phrases.merged(j['phrases'] as Map<String, dynamic>? ?? const {}),
      {...base.spokenNames, ...Map<String, String>.from(j['spoken_names'] as Map? ?? const {})},
    );
  } catch (_) {
    return base; // missing or broken pack: English, never silence
  }
}
