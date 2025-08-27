# app.py — Command runner with SSE & cancel
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import subprocess, uuid, threading, queue, os, signal

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==== In-memory job store ====
jobs = {}      # job_id -> subprocess.Popen
buffers = {}   # job_id -> queue.Queue()
done = set()   # job_ids that have finished
# --- inside app.py ----------------------------------------------------------
def reader_thread(job_id, proc):
    buf = []
    while True:
        ch = proc.stdout.read(1)
        if not ch:                 # EOF
            if buf:
                buffers[job_id].put(''.join(buf))
            break

        if isinstance(ch, bytes):  # <— auto-decode if needed
            ch = ch.decode(errors='replace')

        if ch in '\r\n':           # flush on CR or LF
            buf.append(ch)
            buffers[job_id].put(''.join(buf))
            buf.clear()
        else:
            buf.append(ch)

    proc.wait()
    done.add(job_id)
    buffers[job_id].put(None)      # sentinel


# ==== POST /command ====
@app.route("/command", methods=["POST"])
def run_command():
    # support JSON {"cmd": "..."} or raw text body
    data = request.get_json(silent=True)
    cmd = None
    if data and "cmd" in data:
        cmd = data["cmd"]
    else:
        cmd = request.data.decode().strip()
    if not cmd:
        return jsonify({"error": "No command supplied"}), 400

    job_id = str(uuid.uuid4())

    # ensure Python child processes flush immediately
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=False, env=env
    )

    jobs[job_id] = proc
    buffers[job_id] = queue.Queue(maxsize=0)
    threading.Thread(target=reader_thread, daemon=True,
                     args=(job_id, proc)).start()

    return jsonify({"job_id": job_id})

# ==== GET /events/<job_id> ====
@app.route("/events/<job_id>")
def stream(job_id):
    if job_id not in buffers:
        return "Unknown job ID", 404

    def event_stream():
        q = buffers[job_id]
        while True:
            line = q.get()
            if line is None:
                # signal to client that job is done
                yield f"event: done\ndata: {job_id}\n\n"
                break
            # strip carriage returns so browser re-renders tqdm properly
            text = line.rstrip("\r")
            # SSE wants each chunk prefixed with `data: `
            yield f"data: {text}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # if you're behind nginx
    }
    return Response(
        stream_with_context(event_stream()),
        headers=headers,
        mimetype="text/event-stream"
    )

# ==== POST /command/<job_id>/cancel ====
@app.route("/command/<job_id>/cancel", methods=["POST"])
def cancel(job_id):
    proc = jobs.get(job_id)
    if not proc or proc.poll() is not None:
        return jsonify({"status": "not-running"})
    proc.send_signal(signal.SIGINT)
    return jsonify({"status": "sent"})

if __name__ == "__main__":
    # listen on all interfaces port 8000
    app.run(host="0.0.0.0", port=8000, threaded=True)

