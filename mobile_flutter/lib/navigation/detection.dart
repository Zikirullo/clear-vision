// One detected object, in the same shape as Python's src/detection.py.
//
// Dart notes for a JavaScript developer:
// - `class` works much like a JS class, but fields have static types (`double`, `String`).
// - `final` = assigned once (like `const` for a JS variable, but per object field).
// - `const Detection(...)` constructor: objects can be compile-time constants.
// - `this.x1` in the constructor parameter list assigns the field directly
//   (no `this.x1 = x1;` boilerplate).
// - `get centerX => ...` is a getter: read as `det.centerX`, no parentheses.
//   `=>` is a one-expression function body, like a JS arrow function.

/// One detected object. Box coordinates are fractions of the frame:
/// x from 0 (left edge) to 1 (right edge), y from 0 (top) to 1 (bottom).
class Detection {
  const Detection(this.className, this.confidence, this.x1, this.y1, this.x2, this.y2);

  final String className;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  /// Horizontal center: `(x1 + x2) / 2`, exactly as in the spec (section 2).
  double get centerX => (x1 + x2) / 2;
  double get centerY => (y1 + y2) / 2;
  double get width => x2 - x1;

  /// Box height: the distance cue (taller box = closer object).
  double get height => y2 - y1;
  double get area => width * height;

  // toString() is what debugPrint / string interpolation shows (like JS toString()).
  @override
  String toString() => 'Detection($className ${confidence.toStringAsFixed(2)} '
      '[${x1.toStringAsFixed(3)}, ${y1.toStringAsFixed(3)}, ${x2.toStringAsFixed(3)}, ${y2.toStringAsFixed(3)}])';
}
