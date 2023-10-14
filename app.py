import os

import json
from flask import Flask, request, redirect
from inference import Inference
import ast
import librosa
from pathlib import Path
from werkzeug import secure_filename

inference = Inference()

app = Flask(__name__)
app.run()

uploads_dir = Path(app.instance_path)/'uploads'
uploads_dir.mkdir(parents=True, exists_ok=True)


@app.route("/")
def hello_world():
    return {"Health": "Good"}


@app.route("/test", methods=["GET"])
def test():
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


# client sends bytes to backend
@app.route("/send_recording", methods=["POST"])
def send_recording():
    recording = request.files['file']
    path = uploads_dir/secure_filename(recording.filename)
    recording.save(path)
    audio, _ = librosa.load(path, sr=125)
    res = inference.predict(audio)
    return res
