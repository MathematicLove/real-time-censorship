# Real Time Censorship API

Real time NSFW detection and automatic blur. Point it at a webcam and anything explicit is covered
before the frame is ever shown, at 60 to 100 fps. Ships as a plain HTTP API, so it drops into any
app in a few lines.

Built on the NudeNet 320n detector, which returns boxes rather than a single score, so the blur
lands on the region instead of the whole frame.

## What it looks like

Every detected region is grown by 85 px on each side and blurred out. Boxes and labels below are
drawn for illustration, the API returns clean frames by default.

<img src="https://mathematiclove.github.io/my-cv/content/projects/REAL_TIME_CENSORSHIP/EXAMPLE_1.png" alt="single subject" width="520">

Two regions on one person, genitalia and breast. The blur box is deliberately larger than the
detection box, so a moving subject stays covered between frames.

<img src="https://mathematiclove.github.io/my-cv/content/projects/REAL_TIME_CENSORSHIP/EXAMPLE_2.png" alt="multiple regions" width="520">

Three regions on the same person, numbered as one. When a second body enters the frame it becomes
person 2, and the numbering stays stable while they are visible.

## Quick start

```sh
docker run -p 8000:8000 flugmaschine/real-time-censorship
```

Open <http://127.0.0.1:8000>. The model ships inside the image, nothing is downloaded on first run.

## API

| method | path | purpose |
| --- | --- | --- |
| POST | /detect | upload an image, get boxes, classes, scores and person numbers |
| POST | /censor | upload an image, get the blurred image back |
| GET | /camera/stream | mjpeg stream of the censored camera feed |
| GET | /camera/frame | single censored frame as jpeg |
| POST | /camera/start | open the camera |
| POST | /camera/stop | close the camera |
| GET | /settings | current blur strength, padding, threshold and classes |
| POST | /settings | change any of them at runtime |
| GET | /logs | stored nsfw detections |
| GET | /stats | totals and live fps |
| GET | /health | service state |

**by Salimli Ayzek (Салимли Айзек): https://mathematiclove.github.io/my-cv**