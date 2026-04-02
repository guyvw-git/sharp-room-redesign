#!/usr/bin/env python3
"""
Room Redesign Webapp
Usage:  python server.py
Needs:  pip install flask requests
"""
import base64, json, uuid, threading, queue, time, mimetypes
import requests as req_lib
from pathlib import Path
from flask import Flask, request, Response, send_from_directory, jsonify

ROOT       = Path(__file__).parent
UPLOAD_DIR = ROOT / '.claude/uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / 'output').mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

# Per-session SSE queues
_sessions: dict[str, queue.Queue] = {}


# ── Headers required for SharedArrayBuffer (Gaussian Splat renderer) ──────────
@app.after_request
def add_headers(response):
    response.headers['Cross-Origin-Opener-Policy']   = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response


# ── Static routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(ROOT, 'webapp.html')

@app.route('/public/<path:f>')
def serve_public(f):
    return send_from_directory(ROOT / 'public', f)

@app.route('/output/<path:f>')
def serve_output(f):
    return send_from_directory(ROOT / 'output', f)

@app.route('/input/<path:f>')
def serve_input(f):
    return send_from_directory(ROOT / 'input', f)

@app.route('/uploads/<path:f>')
def serve_uploads(f):
    return send_from_directory(UPLOAD_DIR, f)


# ── Pipeline endpoint ─────────────────────────────────────────────────────────
@app.route('/api/redesign', methods=['POST'])
def api_redesign():
    image_file = request.files.get('image')
    prompt     = request.form.get('prompt', '').strip()
    if not image_file or not prompt:
        return jsonify({'error': 'image and prompt required'}), 400

    session_id = uuid.uuid4().hex[:8]
    _sessions[session_id] = queue.Queue()

    ext        = Path(image_file.filename).suffix or '.jpg'
    input_path = UPLOAD_DIR / f'{session_id}_input{ext}'
    image_file.save(str(input_path))

    threading.Thread(target=_run_pipeline,
                     args=(session_id, str(input_path), prompt),
                     daemon=True).start()

    return jsonify({'session_id': session_id})


# ── SSE stream ────────────────────────────────────────────────────────────────
@app.route('/api/events/<session_id>')
def api_events(session_id):
    def generate():
        q = _sessions.get(session_id)
        if not q:
            yield f'data: {json.dumps({"type":"error","message":"unknown session"})}\n\n'
            return
        while True:
            try:
                event = q.get(timeout=30)
                yield f'data: {json.dumps(event)}\n\n'
                if event['type'] in ('done', 'error'):
                    _sessions.pop(session_id, None)
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Pipeline ──────────────────────────────────────────────────────────────────
def _run_pipeline(session_id: str, input_path: str, prompt: str):
    def emit(step_id, label, status, detail=''):
        _sessions[session_id].put(
            {'type': 'step', 'id': step_id, 'label': label,
             'status': status, 'detail': detail}
        )

    try:
        emit('upload', 'Photo received', 'done', Path(input_path).name)

        # Step 1 — AI Makeover
        t0 = time.time()
        emit('makeover', 'AI Makeover', 'running',
             f'POST ai-makeover.../api/generate\n"{prompt}"')
        edited_path = _ai_makeover(input_path, prompt, session_id)
        _sessions[session_id].put({
            'type': 'step', 'id': 'makeover', 'label': 'AI Makeover',
            'status': 'done', 'detail': f'{time.time()-t0:.1f}s',
            'image': f'/uploads/{edited_path.name}',
        })

        # Step 2 — SHARP 3DGS reconstruction
        emit('sharp', 'SHARP processing', 'running', f'POST {SHARP_URL}')
        t0 = time.time()
        sharp_result = _run_sharp(str(edited_path))
        total_sec     = time.time() - t0
        inference_sec = sharp_result['inference_time']
        transfer_sec  = total_sec - inference_sec
        emit('sharp', 'SHARP processing', 'done', f'{inference_sec:.1f}s on GPU')
        emit('transfer', 'Transfer Vast→Mac', 'done', f'{transfer_sec:.1f}s')

        _sessions[session_id].put(
            {'type': 'done',
             'ply_url': sharp_result['ply_url'],
             'edited_image_url': f'/uploads/{edited_path.name}'}
        )

    except Exception as e:
        _sessions[session_id].put({'type': 'error', 'message': str(e)})


# ── AI Makeover ───────────────────────────────────────────────────────────────
AI_MAKEOVER_URL = 'https://ai-makeover-153178030687.us-central1.run.app/api/generate'
AI_MAKEOVER_KEY = 'dev-master-bypass'

def _ai_makeover(image_path: str, prompt: str, session_id: str) -> Path:
    mime = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {'customPrompt': prompt, 'imageBase64': f'data:{mime};base64,{b64[:40]}...'}
    print(f'[ai-makeover] payload (imageBase64 truncated): {payload}')
    resp = req_lib.post(
        AI_MAKEOVER_URL,
        headers={'Content-Type': 'application/json', 'x-api-key': AI_MAKEOVER_KEY},
        json={'customPrompt': prompt, 'imageBase64': f'data:{mime};base64,{b64}'},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    ai_url = data.get('aiUrl')
    if not ai_url:
        raise RuntimeError(f'No aiUrl in response: {list(data.keys())}')

    out = UPLOAD_DIR / f'{session_id}_edited.jpg'

    if ai_url.startswith('data:'):
        # Base64 data URI — decode directly
        img_b64 = ai_url.split(',', 1)[1]
        out.write_bytes(base64.b64decode(img_b64))
    else:
        # Regular URL — fetch it
        img_resp = req_lib.get(ai_url, timeout=60)
        img_resp.raise_for_status()
        out.write_bytes(img_resp.content)

    return out


# ── SHARP (local LAN splat server) ───────────────────────────────────────────
SHARP_URL = 'http://localhost:3001/splat'  # SSH tunnel — image upload only (small)

def _run_sharp(image_path: str, _retries: int = 2) -> dict:
    for attempt in range(1, _retries + 2):
        try:
            with open(image_path, 'rb') as f:
                resp = req_lib.post(SHARP_URL, files={'image': f}, timeout=600)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt <= _retries:
                print(f'[sharp] attempt {attempt} failed ({e}), retrying...')
            else:
                raise


if __name__ == '__main__':
    print('Starting Room Redesign Webapp at http://localhost:8080')
    app.run(host='0.0.0.0', port=8080, debug=False)
