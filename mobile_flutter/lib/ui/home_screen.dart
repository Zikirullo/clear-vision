// The one screen: the camera (YOLOView) runs underneath, covered by one full-screen
// button. Triple-tap uncovers it as the debug view.
//
// Gestures without VoiceOver:   double-tap = start/stop, long press = what is in front,
//                               triple-tap = debug view, single tap = say the status.
// With VoiceOver on, VoiceOver owns the gestures: its double-tap "activates" the button
// (= start/stop), double-tap-and-hold = long press, and the rotor's "Actions" list offers
// "Debug view" and "Settings".
//
// Flutter notes:
// - `Stack` layers children on top of each other (like position: absolute in CSS).
// - `Semantics` describes a widget for screen readers (like aria-label / role in HTML).
// - `WidgetsBindingObserver` tells us when the app goes to the background/foreground
//   (like the browser's visibilitychange event).
// - `ListenableBuilder` rebuilds its child whenever the controller notifies.

import 'package:flutter/semantics.dart'; // CustomSemanticsAction (VoiceOver rotor actions)
import 'package:material_ui/material_ui.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../app_controller.dart';
import 'debug_overlay.dart';
import 'settings_screen.dart';
import 'tap_sequence.dart';
import 'theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.controller});

  final AppController controller;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  // `with` mixes in extra behavior (here: lifecycle callbacks), a bit like a hook.
  bool _debug = false;
  late final TapSequence _taps = TapSequence(onSequence: _onTaps);

  AppController get c => widget.controller;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _taps.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) c.onBackground();
    if (state == AppLifecycleState.resumed) c.onForeground();
  }

  void _onTaps(int n) {
    if (n == 2) c.toggle();
    if (n >= 3) _setDebug(true);
    if (n == 1) c.speech.say(c.phrases.systemText(c.running ? 'started' : 'stopped'), urgent: true);
  }

  void _setDebug(bool on) {
    setState(() => _debug = on);
    c.yolo.setShowOverlays(on); // the plugin draws boxes only in the debug view
  }

  void _openSettings() {
    Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => SettingsScreen(controller: c)));
  }

  @override
  Widget build(BuildContext context) {
    final modelPath = c.modelPath;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Camera + model. Always mounted while the app runs, so detection never stops;
          //    hidden behind the button unless the debug view is open.
          if (modelPath != null)
            ExcludeSemantics(
              child: YOLOView(
                modelPath: modelPath,
                task: YOLOTask.detect,
                controller: c.yolo,
                confidenceThreshold: c.config.pluginThreshold,
                cameraResolution: '720p',
                streamingConfig: const YOLOStreamingConfig.minimal(),
                onResult: c.onResults,
                onPerformanceMetrics: c.onMetrics,
                onModelLoad: (_, _) {
                  c.yolo.setShowOverlays(_debug);
                  c.onModelLoaded();
                },
                onModelError: (error, _, _) => c.onModelError(error),
              ),
            ),
          // 2. The full-screen button, or the debug view.
          if (_debug)
            DebugOverlay(controller: c, onClose: () => _setDebug(false))
          else
            _MainButton(controller: c, onTap: _taps.tap, onDebug: () => _setDebug(true), onSettings: _openSettings),
        ],
      ),
    );
  }
}

class _MainButton extends StatelessWidget {
  const _MainButton({required this.controller, required this.onTap, required this.onDebug, required this.onSettings});

  final AppController controller;
  final VoidCallback onTap;
  final VoidCallback onDebug;
  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        final c = controller;
        final error = c.modelError;
        final status = error != null ? 'Model failed' : (c.running ? 'Running' : 'Stopped');
        final hint = c.running ? 'Double tap to stop.' : 'Double tap to start.';
        return Container(
          color: bg, // opaque: the camera stays hidden (and the screen dark for bystanders)
          child: SafeArea(
            child: Column(
              children: [
                // Settings: a normal, labeled button at the top (for a helper or VoiceOver).
                Align(
                  alignment: Alignment.topRight,
                  child: Padding(
                    padding: const EdgeInsets.all(8),
                    child: Semantics(
                      button: true,
                      label: 'Settings',
                      excludeSemantics: true,
                      child: IconButton(
                        iconSize: 40,
                        color: fg,
                        icon: const Icon(Icons.settings),
                        onPressed: onSettings,
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: Semantics(
                    button: true,
                    label: 'Clear Vision. $status.',
                    hint: '$hint Long press: what is in front.',
                    onTap: c.toggle, // VoiceOver double-tap
                    onLongPress: c.describe, // VoiceOver double-tap and hold
                    customSemanticsActions: {
                      const CustomSemanticsAction(label: 'Debug view'): onDebug,
                      const CustomSemanticsAction(label: 'Settings'): onSettings,
                    },
                    excludeSemantics: true,
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTap: onTap,
                      onLongPress: c.describe,
                      child: Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(c.running ? Icons.visibility : Icons.visibility_off,
                                  size: 120, color: error != null ? alertRed : fg),
                              const SizedBox(height: 24),
                              Text(status.toUpperCase(),
                                  style: bigText.copyWith(fontSize: 56, color: error != null ? alertRed : fg)),
                              const SizedBox(height: 24),
                              Text('$hint\nLong press: what is in front.',
                                  textAlign: TextAlign.center, style: mediumText),
                              if (error != null) ...[
                                const SizedBox(height: 16),
                                Text(error, textAlign: TextAlign.center, style: smallText),
                              ],
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
