// Settings: voice speed, language, normal/silent mode, vibration, hazards only.
// Every control has a VoiceOver label; values are saved at once (lib/services/settings.dart).
//
// Flutter notes:
// - `Navigator.push` opened this screen (like pushing a route in React Router); the
//   back button / swipe pops it.
// - `ListView` is a scrollable column.

import 'package:material_ui/material_ui.dart';

import '../app_controller.dart';
import '../services/assets.dart';
import '../services/settings.dart';
import 'theme.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key, required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    final s = controller.settings;
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListenableBuilder(
        listenable: s,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text('Voice speed', style: mediumText),
            Semantics(
              label: 'Voice speed',
              value: '${(s.speechRate * 100).round()} percent',
              child: Slider(
                value: s.speechRate,
                min: 0.2,
                max: 0.8,
                divisions: 12,
                label: '${(s.speechRate * 100).round()}%',
                onChanged: (v) => s.speechRate = v,
              ),
            ),
            FilledButton(
              onPressed: () => controller.speech.say('Stairs ahead, close', urgent: true),
              child: const Text('Test voice'),
            ),
            const SizedBox(height: 24),
            Text('Language', style: mediumText),
            FutureBuilder<List<String>>(
              future: availableLanguages(),
              builder: (context, snap) {
                final available = snap.data ?? const ['en'];
                return Column(
                  children: [
                    for (final e in languageNames.entries)
                      RadioListTile<String>(
                        value: e.key,
                        // ignore: deprecated_member_use
                        groupValue: s.language,
                        // ignore: deprecated_member_use
                        onChanged: available.contains(e.key) ? (v) => s.language = v! : null,
                        title: Text(available.contains(e.key) ? e.value : '${e.value} (coming later)',
                            style: mediumText.copyWith(color: available.contains(e.key) ? fg : Colors.white38)),
                      ),
                  ],
                );
              },
            ),
            const SizedBox(height: 24),
            Text('Mode', style: mediumText),
            for (final m in OutputMode.values)
              RadioListTile<OutputMode>(
                value: m,
                // ignore: deprecated_member_use
                groupValue: s.mode,
                // ignore: deprecated_member_use
                onChanged: (v) => s.mode = v!,
                title: Text(m == OutputMode.normal ? 'Normal: speech and vibration' : 'Silent: vibration only',
                    style: mediumText),
                subtitle: m == OutputMode.silent
                    ? const Text('Camera and battery alerts are still spoken.', style: smallText)
                    : null,
              ),
            const SizedBox(height: 24),
            SwitchListTile(
              value: s.vibration,
              onChanged: (v) => s.vibration = v,
              title: Text('Vibration', style: mediumText),
              subtitle: const Text('Pulses: 1 = left, 2 = right, 3 = ahead. Faster = closer.', style: smallText),
            ),
            SwitchListTile(
              value: s.hazardsOnly,
              onChanged: (v) => s.hazardsOnly = v,
              title: Text('Hazards only', style: mediumText),
              subtitle: const Text('Only curbs, stairs, holes, posts, barriers, people and vehicles.', style: smallText),
            ),
          ],
        ),
      ),
    );
  }
}
