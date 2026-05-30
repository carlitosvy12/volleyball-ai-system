# Volleyball Activity Analysis System

AI system that analyses volleyball footage: it detects and tracks the players, classifies their
team and role, detects the ball, predicts player movement, shows a 2D minimap, and includes a
natural-language assistant. Everything runs from a single Gradio interface.

**Course:** Neural Networks / Deep Learning — Universidad San Jorge
**Authors:** Carlos Vicente, Javier Revuelta, Ángel Moliné, Jorge Molia

---

## Notebooks (Colab)

- **`v12_final_VF`** — the main application. Loads every model and launches the GUI. It also
  trains the role CNN (from a tab) and the team MLP.
- **`TrainPersonVF.ipynb`** — trains the player detector → `volleyball_persons_best.pt`
- **`BallTrain_VF.ipynb`** — trains the ball detector → `volleyball_ball_best.pt`
- **`TrainLSTM_temporal_Final.ipynb`** — trains the trajectory model → `best_lstm_tracker.pth`
- **`train_cnn_classifier`** —  → `best_cnn_classifier`
## Weight files

- `volleyball_persons_best.pt` — player detector (YOLOv8)
- `volleyball_ball_best.pt` — ball detector (YOLOv8)
- `best_cnn_classifier.pth` — role classifier (CNN)
- `best_lstm_tracker.pth` — trajectory model (LSTM)
- `best_team_mlp.pt` — team classifier (MLP)

## Datasets

- **`player.yolov8`** — public Roboflow dataset of players labelled by team. Used for the player detector.
- **`volleyball-ball-a-jpn.yolov8`** — made by us: frames extracted from our match video with the
  ball annotated by hand. Used for the ball detector.
- **`volley_detection` (v5)** — annotated volleyball dataset; player crops are cut from it to train the role CNN.

## Video

Our own match clip (France vs Japan). It is the source for the LSTM trajectories and for the
K-means / MLP team-colour labels. Place it in the Colab session as `/content/video.mp4`.

---

## How to run

Open `v12_final_VF` in Colab, pick a GPU runtime, and run the cells in order (Sections 0 to 14):
install dependencies, define everything, load the models, apply the patches, the team MLP, the
single-frame test, and finally launch the interface.

For each model you can either **upload the weight** or **train it**:

- **Player detector / Ball detector / LSTM** → upload `volleyball_persons_best.pt`,
  `volleyball_ball_best.pt` and `best_lstm_tracker.pth`. These are trained only in their own
  notebooks (`TrainPersonVF`, `BallTrain_VF`, `TrainLSTM_temporal_Final`); the app only loads them.
- **Role CNN** → upload `best_cnn_classifier.pth` to use it, or train it from the "Train CNN" tab
  (needs the `volley_detection-5` dataset in `/content/`), also can entrain with his colab.
- **Team MLP** → upload `best_team_mlp.pt` to use it, or let Sections 11–12 train it from the video.
- **Gemma 4** → downloaded automatically on first run, nothing to upload (set `USE_GEMMA = False` to skip it).

In short: upload the five weight files and the video to `/content/`, run all the cells in order, and
the interface opens at the end.

**Re-training a module**
To re-train a module, open its Colab notebook, upload the file(s) it needs, and run it:

TrainPersonVF.ipynb (player detector) → upload the player.yolov8 dataset (as player.yolov8.zip). Produces volleyball_persons_best.pt.
BallTrain_VF.ipynb (ball detector) → upload the volleyball-ball-a-jpn.yolov8 dataset (as volleyball-ball-a-jpn.yolov8.zip). Produces volleyball_ball_best.pt.
TrainLSTM_temporal_Final.ipynb (trajectory LSTM) → upload the match video (video.mp4) and the player weights volleyball_persons_best.pt. Produces best_lstm_tracker.pth.

Then upload the generated weight file to v11_final.ipynb. The role CNN and the team MLP are re-trained inside v11_final.ipynb (the CNN needs the volley_detection-5 dataset in /content/; the MLP uses video.mp4