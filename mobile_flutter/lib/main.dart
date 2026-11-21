// App entry point.
//
// Flutter notes for a React developer:
// - Everything on screen is a *widget* (like a React component). `build()` returns the
//   widget tree, like a component's render/return JSX.
// - `StatelessWidget` = a pure function component; `StatefulWidget` + its `State` class =
//   a component with useState/useEffect (`initState` ~ mount effect, `dispose` ~ cleanup,
//   `setState(() {...})` ~ calling a state setter).
// - Material widgets come from package:material_ui (split out of Flutter in 3.47).
// - `MaterialApp` is the root (theme, navigation stack), roughly like a router + theme
//   provider at the top of a React app.
// - `main()` is where Dart starts; `runApp()` mounts the root widget (like
//   `ReactDOM.createRoot(...).render(<App />)`).

import 'package:material_ui/material_ui.dart';

import 'app_controller.dart';
import 'ui/home_screen.dart';
import 'ui/theme.dart';

void main() {
  // Needed before using plugins (assets, preferences, TTS) ahead of runApp.
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ClearVisionApp());
}

class ClearVisionApp extends StatelessWidget {
  const ClearVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Clear Vision',
      debugShowCheckedModeBanner: false,
      theme: highContrastTheme,
      darkTheme: highContrastTheme,
      home: const _Loader(),
    );
  }
}

/// Shows a spoken-friendly splash until config, settings, TTS and the model path are loaded.
class _Loader extends StatefulWidget {
  const _Loader();

  @override
  State<_Loader> createState() => _LoaderState();
}

class _LoaderState extends State<_Loader> {
  // A Future is a Promise. FutureBuilder (below) re-renders when it resolves,
  // like a component that awaits data with useEffect + useState.
  late final Future<AppController> _controller = AppController.create();
  // `late` = initialized on first use (here: once, when the state is created).

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<AppController>(
      future: _controller,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Scaffold(
            body: Center(
              child: Semantics(
                liveRegion: true, // VoiceOver reads it as soon as it appears
                child: Text('Clear Vision could not start:\n${snapshot.error}',
                    textAlign: TextAlign.center, style: bigText),
              ),
            ),
          );
        }
        if (!snapshot.hasData) {
          return Scaffold(
            body: Center(child: Semantics(liveRegion: true, child: Text('Starting Clear Vision…', style: bigText))),
          );
        }
        return HomeScreen(controller: snapshot.data!);
      },
    );
  }
}
