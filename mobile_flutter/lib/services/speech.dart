// Text-to-speech with two priorities, like Python's src/speaker.py:
// - urgent: interrupts whatever is being said
// - normal: only spoken when nothing is being said (otherwise dropped; the caller then
//   doesn't mark it as spoken, so it can come again on a later frame)
//
// iPhone: the audio session category is `playback`, so speech plays even with the
// ring/silent switch on silent, and other audio (music, podcasts) is ducked.
// Android: audio attributes "navigation guidance" + audio focus on every phrase, so
// music ducks while a warning plays.
//
// Dart notes:
// - `unawaited(f)` starts an async call without waiting for it (fire-and-forget, like
//   calling an async function in JS without `await`). The main loop must never block.
// - `Platform.isIOS` (dart:io) tells which phone OS we are on.

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

class SpeechService {
  final FlutterTts _tts = FlutterTts();
  bool _speaking = false;
  Future<void> _chain = Future.value(); // serializes stop/speak calls to the plugin

  bool get speaking => _speaking;

  Future<void> init({required double rate, required String locale}) async {
    if (Platform.isIOS) {
      await _tts.setSharedInstance(true);
      await _tts.setIosAudioCategory(
        IosTextToSpeechAudioCategory.playback, // ignores the silent switch
        [
          IosTextToSpeechAudioCategoryOptions.duckOthers,
          IosTextToSpeechAudioCategoryOptions.interruptSpokenAudioAndMixWithOthers,
          IosTextToSpeechAudioCategoryOptions.allowBluetoothA2DP,
        ],
        IosTextToSpeechAudioMode.voicePrompt,
      );
    } else if (Platform.isAndroid) {
      await _tts.setAudioAttributesForNavigation();
    }
    await _tts.awaitSpeakCompletion(false); // speak() returns at once; handlers track state
    _tts.setStartHandler(() => _speaking = true);
    _tts.setCompletionHandler(() => _speaking = false);
    _tts.setCancelHandler(() => _speaking = false);
    _tts.setErrorHandler((msg) {
      _speaking = false;
      debugPrint('TTS error: $msg');
    });
    await setRate(rate);
    await setLocale(locale);
  }

  Future<void> setRate(double rate) => _tts.setSpeechRate(rate);

  Future<void> setLocale(String locale) async {
    final ok = await _tts.isLanguageAvailable(locale);
    if (ok == true) {
      await _tts.setLanguage(locale);
    } else {
      debugPrint('TTS: $locale not available, keeping the default voice');
    }
  }

  /// Says `text` now if allowed. Returns true if it was accepted (counts as spoken).
  bool say(String text, {required bool urgent}) {
    if (_speaking && !urgent) return false;
    final interrupt = _speaking;
    _speaking = true; // until the start/completion handlers report the real state
    _chain = _chain.then((_) async {
      try {
        if (interrupt) await _tts.stop();
        await _tts.speak(text, focus: true); // focus: Android audio focus (ducks music)
      } catch (e) {
        _speaking = false;
        debugPrint('TTS speak failed: $e');
      }
    });
    return true;
  }

  Future<void> stop() async {
    await _tts.stop();
    _speaking = false;
  }
}
