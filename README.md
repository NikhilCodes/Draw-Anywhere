# Draw Anywhere

## Intro
Not just any simple Digit Recognition. MNIST Digit Recognition got lot more Interesting!
- Run `python run_cam.py` to launch the camera app.
- Press `<SPACE>` bar to start tracing, and hit `<SPACE>` again to stop
  tracing and recognise the drawing.

## Command Line Options
- `--color` or `-c` to specify the color of pointer it should look for.
  SUPPORTED OPTIONS: [green, blue, red]
- `--canvas` or `-s` to specify whether to display a Window containing your drawing without Background.
  SUPPORTED OPTIONS: [True, False]
- `--area` or `-a` to specify minimum area in pixels.
- `--display` or `-d` to specify, How long is prediction shown in second(s).

## Demo
![gif Playback](DEMO/DEMO-1.gif)
<br><br>

## Dependencies
|   Library    |       pip command         |
|--------------|---------------------------|
| cv2          |`pip install opencv-python`|
| keras        |`pip install keras`        |
| numpy        |`pip install numpy`        |
| tensorflow   |`pip install tenorflow`    |

### DON'T FORGET TO HIT THE STAR BUTTON.
