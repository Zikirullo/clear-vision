// Hidden debug view (triple-tap): the camera with the plugin's boxes, FPS, inference
// time, what the rules consider (candidates) and the last spoken messages.
// It is drawn on top of the live YOLOView, so detection keeps running unchanged.

import 'package:material_ui/material_ui.dart';

import '../app_controller.dart';
import 'theme.dart';

class DebugOverlay extends StatelessWidget {
  const DebugOverlay({super.key, required this.controller, required this.onClose});

  final AppController controller;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        final c = controller;
        final fpsColor = c.fps < c.config.app.slowFps ? alertRed : Colors.greenAccent;
        return SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                color: Colors.black54,
                padding: const EdgeInsets.all(8),
                child: DefaultTextStyle(
                  style: smallText,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(children: [
                        Text('${c.fps.toStringAsFixed(0)} FPS',
                            style: mediumText.copyWith(color: fpsColor, fontSize: 28)),
                        const SizedBox(width: 12),
                        Text('model ${c.inferenceMs.toStringAsFixed(0)} ms'),
                        const Spacer(),
                        Semantics(
                          button: true,
                          label: 'Close debug view',
                          excludeSemantics: true,
                          child: FilledButton(onPressed: onClose, child: const Text('Close')),
                        ),
                      ]),
                      Text('${c.running ? 'RUNNING' : 'STOPPED'} · ${c.settings.mode.name}'
                          '${c.planner.muteNonHazards ? ' · hazards only' : ''}'
                          '${c.health.cameraLost ? ' · CAMERA LOST' : ''}'),
                      Text('model: ${c.modelPath?.split('/').last ?? 'none'}'
                          '${c.modelLoaded ? '' : ' (not loaded)'}'),
                      if (c.modelError != null) Text('error: ${c.modelError}', style: const TextStyle(color: alertRed)),
                      if (c.labelWarning != null) Text(c.labelWarning!, style: const TextStyle(color: alertRed)),
                      Text('${c.lastResults.length} detections, ${c.lastCandidates.length} above speak thresholds: '
                          '${c.lastCandidates.map((d) => '${d.className} ${(d.confidence * 100).round()}%').join(', ')}'),
                    ],
                  ),
                ),
              ),
              const Spacer(),
              Container(
                color: Colors.black54,
                padding: const EdgeInsets.all(8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Last messages', style: TextStyle(color: fg, fontWeight: FontWeight.bold)),
                    for (final s in c.spoken)
                      Text(
                        '${s.t.toStringAsFixed(1)} s  ${s.urgent ? '[urgent] ' : ''}${s.silent ? '[vibrated] ' : ''}${s.text}',
                        style: smallText.copyWith(color: s.urgent ? alertRed : Colors.white),
                      ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
