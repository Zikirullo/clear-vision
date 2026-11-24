# Clear Vision: phone app (Flutter)

The phone version of Clear Vision. The camera finds obstacles while you walk; short spoken
warnings and vibrations tell you what is where. iPhone first, Android from the same code.

```
YOLOView (camera + Core ML / LiteRT model, ultralytics_yolo plugin)
   │ onResult: class, confidence, box for every frame
   ▼
lib/navigation/   the rules, ported 1:1 from Python (docs/navigation_spec.md)
   │ at most one Message per frame
   ▼
speech (flutter_tts) + vibration          health checks: camera lost, slow, battery
```

| Folder | What is in it |
|---|---|
| `lib/navigation/` | Pure Dart rules: 3-of-5 confirmation, speak thresholds, speaking zone, cooldowns, the "post" naming rule, health checks, vibration patterns |
| `lib/services/` | Speech, vibration, settings, loading the bundled files |
| `lib/ui/` | The full-screen button, the debug view, settings |
| `lib/app_controller.dart` | Wires it all together (like Python's `src/main.py`) |
| `assets/` | `nav_config.json` (the rules' numbers and phrases), `models/` (the model files) |
| `test/` | `nav_golden_test.dart` replays `../tests/golden/nav_cases.json`; app logic tests |

Every Dart file explains the Flutter and Dart ideas it uses, compared with JavaScript and React.

---

## 1. Set up the Mac (once)

You need macOS with Xcode installed from the App Store (Xcode 27 on macOS Tahoe works).

```bash
# Flutter (the SDK + the `flutter` and `dart` commands)
brew install --cask flutter

# Check everything. Fix what it reports for "Xcode"; ignore "Android" until section 6.
flutter doctor

# Xcode's command-line part, first-launch setup and the iOS platform
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadPlatform iOS

# CocoaPods: one plugin (flutter_tts) still installs through it; the others use Swift Package Manager
brew install cocoapods
```

## 2. Put the model and config into the app

The model files are not in git (they are big). Download them from Google Drive
`MyDrive/clear_vision/exports/budget_v1/mobile/` into `exports/budget_v1/mobile/` in the repo:

- `clear_vision_budget_v1.mlpackage.zip` (iPhone)
- `clear_vision_budget_v1.tflite` (Android)

Then, from the repository root:

```bash
source .venv/bin/activate
python scripts/sync_mobile_assets.py          # copies nav_config.json, labels.txt and the models
```

It prints `MISSING` for any model it can't find. The iPhone build only needs the `.mlpackage.zip`.

**Changing to a newer model (v2, v3):** export it with notebook 07, download it the same way,
then run `python scripts/sync_mobile_assets.py --run v2`. The script replaces the old file
in `assets/models/`. The app loads whichever `.mlpackage.zip` (iPhone) or `.tflite`
(Android) it finds there, so no code changes. Keep the run name in the file name: the
plugin unpacks a model once per name.

## 3. Run it on your iPhone

1. **Get the Dart packages:**
   ```bash
   cd mobile_flutter
   flutter pub get
   ```
2. **Open the iOS project in Xcode:**
   ```bash
   open ios/Runner.xcworkspace
   ```
   Always open the `.xcworkspace`, not the `.xcodeproj`.
3. **Sign in:** Xcode > Settings… > Accounts > **+** > Apple ID, and sign in with your
   Apple ID (a free one is fine).
4. **Choose your team:** in the left sidebar click **Runner** (the blue icon), then the
   **Runner** target > **Signing & Capabilities**:
   - Tick **Automatically manage signing**.
   - **Team:** choose *Your Name (Personal Team)*.
   - **Bundle Identifier:** change `com.clearvision.clearVision` to something only you use,
     for example `com.yourname.clearvision`. Free accounts need a unique one.
5. **Connect the iPhone by USB cable.** Unlock it and tap **Trust This Computer**.
6. **Turn on Developer Mode on the iPhone:** Settings > Privacy & Security > Developer Mode
   > On. The iPhone restarts; confirm **Turn On**. The switch only appears after the iPhone
   has been connected to Xcode once; if it isn't there, pick your iPhone as the run target
   in Xcode's top bar and press ▶ once.
7. **Build and install** (from `mobile_flutter/`):
   ```bash
   flutter devices          # your iPhone should be listed
   flutter run --release    # release = full speed; debug builds are much slower
   ```
   The first build takes a few minutes.
8. **Trust your developer profile (first install only):** if the iPhone says *Untrusted
   Developer*, go to Settings > General > VPN & Device Management > your Apple ID >
   **Trust**. Then open Clear Vision.
9. Allow camera access when asked. The app starts by itself and says "Clear Vision started".

**The free Apple ID 7-day limit:** apps installed with a free Personal Team stop opening
after **7 days**. Connect the iPhone and run `flutter run --release` again to reinstall (your
settings are kept). A free account can also have at most 3 such apps on a device at once.
To give the app to testers with **TestFlight**, or to avoid the 7 days, you need the paid
**Apple Developer Program** (99 USD per year).

## 4. Using the app

The whole screen is one button (yellow on black):

| Gesture (VoiceOver off) | VoiceOver on | Does |
|---|---|---|
| Double-tap | Double-tap | Start / stop detection |
| Long press | Double-tap and hold | Say what is in front now ("Stairs ahead, close. Pole, left.") |
| Single tap | (select) | Say whether it is running |
| Triple-tap | Rotor > Actions > Debug view | Debug view: camera with boxes, FPS, last messages |
| Settings icon (top right) | Rotor > Actions > Settings | Settings |

- **Warnings:** the same rules as the Mac app. An object must be seen in 3 of 5 frames,
  pass its speak threshold, and be near enough. The same warning repeats after 4 s
  (2 s if it is close and straight ahead). Cones, bollards, hydrants and poles are said as
  "Obstacle" when the model isn't sure which one it is. Urgent warnings interrupt speech.
- **Vibration:** 1 pulse = left, 2 = right, 3 = ahead; faster pulses = closer.
- **Silent mode** (Settings): warnings by vibration only. "Camera lost", "Battery low"
  and the other system alerts are still spoken.
- **Speech plays even with the iPhone's silent switch on**, because the audio session is
  `playback`. Music is ducked while a warning plays.
- **Health checks:**
  - "Camera lost" after 1.5 s without frames (shorter hiccups stay silent), then
    "Camera back" once.
  - "Detection is slow" below 8 frames/s for 3 s.
  - "Battery low" at 20%, "Battery critical" at 10%.
  - "Detection model failed" if the model can't load.
- **The screen stays on** while running (wakelock). Mount the phone **upright** on the
  chest; the app is portrait only, because left/ahead/right are thirds of the upright
  picture.
- **Settings:** voice speed, language (English now; Korean and Uzbek once their phrase
  files exist), normal/silent, vibration, hazards only.

All numbers (thresholds, cooldowns, 1.5 s, 8 FPS, battery levels, vibration timing) come
from `configs/default.yaml` in the repo, through `assets/nav_config.json`.

### Acceptance check (street video)

1. On the Mac, make the 48 s video of the 24 street photos and play it full screen:
   ```bash
   python scripts/make_photo_video.py tests/assets/street_photos street.mp4
   open street.mp4
   ```
2. Hold the iPhone so the video fills most of the camera picture. Triple-tap for the debug
   view: FPS should be 15 or more.
3. Expect warnings for stairs, potholes, manholes, a curb and "Obstacle" for the posts.
   (`python scripts/safety_report.py` also checks these photos per class: budget_v1 found
   the class in 12 of 24, with no cones and only 1 of 4 poles.)
   For reference, the Mac replay of the same video (`python scripts/replay.py street.mp4
   --backend finetuned`) says: Fire hydrant, Obstacle (×2), Car, Manhole (×4), Stairs (×2),
   Pothole (×2), Fire hydrant (×2), Curb, Truck. Through a phone camera pointed at a screen,
   expect fewer and somewhat different messages.
4. Indoors (walk around a room) it should stay mostly quiet.

## 5. Tests

```bash
cd mobile_flutter
flutter test      # 35 golden cases (Dart = Python), health checks, vibration, taps, config
flutter analyze   # static checks
```

The golden test reads `../tests/golden/nav_cases.json` and `../models/nav_config.json`
directly. When a rule changes in Python, regenerate them (CLAUDE.md), run
`python scripts/sync_mobile_assets.py --config-only`, and port the change until
`flutter test` passes.

## 6. Android (same Dart code)

1. **Install Android Studio:** download it from developer.android.com/studio. On first
   start, let it install the Android SDK, then run:
   ```bash
   flutter doctor --android-licenses   # accept the licenses
   flutter doctor                      # "Android toolchain" should be ✓
   ```
2. **The model:** `clear_vision_<run>.tflite` from notebook 07 (LiteRT, no NMS, with its
   metadata). `python scripts/sync_mobile_assets.py` copies it to `assets/models/`.
   Flutter assets are one of the two places the plugin's docs allow; the other is
   `android/app/src/main/assets/`.
3. **On the phone:**
   - Settings > About phone > tap **Build number** 7 times. This turns on Developer options.
   - Settings > System > Developer options > turn on **USB debugging**.
   - Connect by USB and accept **Allow USB debugging** on the phone.
4. **Run:**
   ```bash
   flutter devices
   flutter run --release
   ```
5. **Share with testers (no 7-day limit on Android):**
   ```bash
   flutter build apk --release
   # -> build/app/outputs/flutter-apk/app-release.apk
   ```
   Send the APK (e.g. via Drive). Testers open it and allow "Install unknown apps" for
   the app they opened it from. The release APK is signed with the debug key, which is
   fine for testing but not for the Play Store.

Already set up for Android: camera and vibration permissions, a text-to-speech query, and
portrait mode in `AndroidManifest.xml`. Speech uses Android's *navigation guidance* audio
attributes and asks for audio focus on every phrase, so music ducks. `minSdk` is 24 and
`compileSdk` 36, as the plugin requires.

## 7. Notebook 07 and the plugin

No changes needed. The plugin reads the task and class names from the metadata that
Ultralytics embeds, and both exports have it:

- **Core ML** (`clear_vision_budget_v1.mlpackage`): `task: detect`, `names` (our 20
  classes in order), `imgsz: [512, 512]`, NMS in the model. The plugin's iOS core
  (UltralyticsYOLO 8.9.23) supports NMS-pipeline models (`VNRecognizedObjectObservation`)
  and letterboxes the whole camera frame.
- **LiteRT** (`clear_vision_<run>.tflite`, `nms=None`): metadata as an appended zip. The
  plugin's Android decoder runs NMS itself and accepts normalized or pixel boxes.

The app also compares the model's names with `assets/models/labels.txt` when it loads,
and says a warning if they differ.

## 8. License

`ultralytics_yolo` and models trained with Ultralytics (our YOLO11 fine-tunes) are
**AGPL-3.0**. That is fine for this personal/research prototype. A commercial product,
or an app whose source isn't published under AGPL, would need an **Ultralytics Enterprise
License** or a different model stack.

## Troubleshooting

- **"Detection model failed" / "No model in assets/models/":** run
  `python scripts/sync_mobile_assets.py` (section 2), then build again.
- **Pod or Swift package errors:** `flutter clean && flutter pub get`, then
  `cd ios && pod install --repo-update`.
- **"No profiles for … were found" / signing errors:** set the Team and a unique Bundle
  Identifier (section 3, step 4).
- **Slow (below 15 FPS):** make sure you used `--release`. The debug view shows FPS and
  the model's own time per frame.
