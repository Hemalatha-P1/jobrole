from flask import Flask, render_template, request
from audiocraft.models import MusicGen
import torchaudio

app = Flask(__name__)

# Load model (download first time)
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=20)  # 20 sec music

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mood = request.form["mood"]
        try:
            # Generate music
            wav = model.generate([mood])  # wav is tensor
            output_file = "static/output.wav"
            torchaudio.save(output_file, wav[0].cpu(), 32000)

            return render_template("index.html", audio_file=output_file, mood=mood)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("index.html", audio_file=None)

if __name__ == "__main__":
    app.run(debug=False)
