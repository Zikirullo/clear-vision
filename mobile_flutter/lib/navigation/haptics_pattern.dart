// Vibration pattern for a warning. Pure, so it is unit-tested.
//
// The number of pulses tells the side (left 1, right 2, ahead 3 by default) and the pulse
// timing tells the distance (closer = shorter gaps = faster). All numbers come from the
// `mobile.haptics` config.
//
// The pattern format is the `vibration` package's: [wait, buzz, wait, buzz, ...] in
// milliseconds, starting with a wait.

import 'nav_config.dart';
import 'navigator.dart';

List<int> hapticPattern(Direction where, Proximity howClose, HapticsConfig cfg) {
  final pulses = cfg.pulses[where.name] ?? 1;
  final on = cfg.onMs[howClose.name] ?? 100;
  final off = cfg.offMs[howClose.name] ?? 150;
  final pattern = <int>[0];
  for (var i = 0; i < pulses; i++) {
    if (i > 0) pattern.add(off);
    pattern.add(on);
  }
  return pattern;
}
