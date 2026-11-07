import io
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string
from PIL import Image

IMG_SIZE = (128, 128)

app = Flask(__name__)
model = tf.keras.models.load_model("blood_resnet.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

PAGE = """
<!doctype html>
<title>Blood Group Detection using Fingerprint</title>
<style>
  body{font-family:Arial,sans-serif;max-width:480px;margin:40px auto;padding:0 16px;text-align:center}
  .card{border:1px solid #ddd;border-radius:10px;padding:20px}
  button{padding:10px 20px;margin-top:12px;cursor:pointer}
  img{max-width:200px;margin-top:12px;border:1px solid #ccc}
  .res{font-size:1.4em;margin-top:14px}
  .note{font-size:.8em;color:#a33;margin-top:18px}
</style>
<h2>Blood Group Detection using Fingerprint</h2>
<div class="card">
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required><br>
    <button type="submit">Predict</button>
  </form>
  {% if label %}
    <img src="data:image/png;base64,{{ preview }}">
    <div class="res">Predicted: <b>{{ label }}</b> ({{ conf }}%)</div>
    <ul style="text-align:left">
      {% for n, p in top %}<li>{{ n }}: {{ p }}%</li>{% endfor %}
    </ul>
  {% endif %}
  {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
</div>
<p class="note">Experimental academic demo. Not a medical test; use a
laboratory serological test to determine blood group.</p>
"""


def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")[None, ..., None]
    return img, arr


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template_string(PAGE, error="Please choose an image.")
        try:
            img, arr = preprocess(f.read())
        except Exception:
            return render_template_string(PAGE, error="Invalid image file.")

        probs = model.predict(arr, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top = [(class_names[i], round(float(probs[i]) * 100, 1))
               for i in order[:3]]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        preview = base64.b64encode(buf.getvalue()).decode()

        return render_template_string(
            PAGE, label=top[0][0], conf=top[0][1], top=top, preview=preview)
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(debug=True)