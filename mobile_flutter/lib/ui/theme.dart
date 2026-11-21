// High-contrast look: yellow on black (the easiest pair for low vision), big text.
// Material `ThemeData` is like a global CSS theme for every widget.

import 'package:material_ui/material_ui.dart';

const Color fg = Color(0xFFFFE600); // bright yellow
const Color bg = Colors.black;
const Color alertRed = Color(0xFFFF5252);

const TextStyle bigText = TextStyle(color: fg, fontSize: 34, fontWeight: FontWeight.w800, height: 1.25);
const TextStyle mediumText = TextStyle(color: fg, fontSize: 22, fontWeight: FontWeight.w600);
const TextStyle smallText = TextStyle(color: Colors.white, fontSize: 15);

final ThemeData highContrastTheme = ThemeData(
  brightness: Brightness.dark,
  scaffoldBackgroundColor: bg,
  colorScheme: const ColorScheme.dark(
    primary: fg,
    onPrimary: bg,
    secondary: fg,
    surface: bg,
    onSurface: fg,
    error: alertRed,
  ),
  textTheme: const TextTheme(bodyLarge: mediumText, bodyMedium: mediumText, titleLarge: bigText),
  appBarTheme: const AppBarTheme(backgroundColor: bg, foregroundColor: fg, titleTextStyle: mediumText),
  sliderTheme: const SliderThemeData(activeTrackColor: fg, thumbColor: fg, inactiveTrackColor: Colors.white24),
  switchTheme: SwitchThemeData(
    thumbColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? bg : Colors.white),
    trackColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? fg : Colors.white24),
  ),
  filledButtonTheme: FilledButtonThemeData(
    style: FilledButton.styleFrom(
      backgroundColor: fg,
      foregroundColor: bg,
      minimumSize: const Size(64, 64), // big touch targets
      textStyle: mediumText,
    ),
  ),
);
