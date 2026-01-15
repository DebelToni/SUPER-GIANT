from flask import Flask, render_template
import os

app = Flask(__name__)

# Read once when the container starts.
API_BASE = os.getenv("GIANT_API", "http://localhost:8000")

# Inject the value into every template so JS can grab it.
@app.context_processor
def inject_api_base():
    return {"api_base": API_BASE}

@app.route("/")
def index():
    return render_template("index.html")   # Jinja fills {{ api_base }}

if __name__ == "__main__":                 # dev-mode only
    app.run(host="0.0.0.0", port=5001, debug=True)

