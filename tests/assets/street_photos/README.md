# Street photos (real-world check)

24 photos of real streets from **Wikimedia Commons**, 4 per class: bollards on sidewalks,
manhole covers, outdoor stairs, potholes, traffic cones and utility poles. Each file name
keeps its Commons title (after the class prefix). They were resized to 800 px for the repo.

Why they are here (2026-11-18): budget_v1 scored 0.98 on traffic cones in the merged test
split but found no cone in these photos. It called cones and a bollard "fire hydrant" and found poles
in 1 of 4. The test split comes from the same Roboflow sources as training; these photos don't.

Used by:
- `scripts/safety_report.py` / notebook 10: per photo, is its class (`expected.yaml`) found?
- `scripts/make_photo_video.py tests/assets/street_photos street.mp4`: the 48 s slideshow for the
  phone check (mobile_flutter/README.md) and for `scripts/replay.py`.

**Licenses:** Wikimedia Commons files have their own licenses (mostly CC BY-SA, CC BY or
public domain) that need attribution when shared. Look up each title on
commons.wikimedia.org (`File:<title>`) and add the author and license here before
redistributing them outside this project.
