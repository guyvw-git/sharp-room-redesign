#!/usr/bin/env python3
"""
SHARP HTTP server — model loaded once at startup, warm for every request.
POST /splat  multipart image -> JSON {ply, ply_url, inference_time, ply_size}
GET  /output/<filename> -> PLY binary (CORS + CORP headers for direct browser download)
"""
import gzip, io, logging, re, tempfile, time, uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, Response, jsonify, send_file

from sharp.models import PredictorParams, create_predictor
from sharp.cli.predict import predict_image
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import save_ply

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

MODEL_PATH  = '/root/sharp_model.pt'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR  = Path('/root/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Read cloudflared tunnel URL written by startup script
_tunnel_file = Path('/root/tunnel_url')
TUNNEL_URL = _tunnel_file.read_text().strip() if _tunnel_file.exists() else None
log.info('Tunnel URL: %s', TUNNEL_URL or '(none — fallback to streaming)')

# ── Load model once at startup ────────────────────────────────────────────────
log.info('Loading SHARP model from %s on %s...', MODEL_PATH, DEVICE)
_state = torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE)
_predictor = create_predictor(PredictorParams())
_predictor.load_state_dict(_state)
_predictor.eval()
_predictor.to(DEVICE)
log.info('Model ready.')

app = Flask(__name__)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': DEVICE, 'tunnel_url': TUNNEL_URL})


@app.route('/output/<filename>')
def serve_output(filename):
    """Serve saved PLY files directly to the browser with CORS/CORP headers."""
    if not re.match(r'^[a-f0-9]+\.ply$', filename):
        return jsonify({'error': 'invalid filename'}), 400
    path = OUTPUT_DIR / filename
    if not path.exists():
        return jsonify({'error': 'not found'}), 404
    response = send_file(str(path), mimetype='application/octet-stream')
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/splat', methods=['POST'])
def splat():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'missing image field'}), 400

    log.info('Request: %s', f.filename)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        f.save(str(tmp_path))

    try:
        image, _, f_px = sharp_io.load_rgb(tmp_path)
        height, width  = image.shape[:2]

        log.info('Running inference (%dx%d f_px=%.1f)...', width, height, f_px)
        t_infer = time.time()
        with torch.no_grad():
            gaussians = predict_image(_predictor, image, f_px, torch.device(DEVICE))
        inference_sec = time.time() - t_infer
        log.info('Inference done in %.2fs', inference_sec)

        with tempfile.TemporaryDirectory() as out_dir:
            ply_path = Path(out_dir) / 'output.ply'
            save_ply(gaussians, f_px, (height, width), ply_path)
            ply_bytes = ply_path.read_bytes()

        # Save to persistent output dir for direct browser download
        ply_id   = uuid.uuid4().hex[:12]
        out_path = OUTPUT_DIR / f'{ply_id}.ply'
        out_path.write_bytes(ply_bytes)
        log.info('Saved PLY %.1fMB → %s', len(ply_bytes) / 1e6, out_path)

    finally:
        tmp_path.unlink(missing_ok=True)

    ply_url = f'{TUNNEL_URL}/output/{ply_id}.ply' if TUNNEL_URL else None

    return jsonify({
        'ply':            f'{ply_id}.ply',
        'ply_url':        ply_url,
        'inference_time': inference_sec,
        'ply_size':       len(ply_bytes),
    })


if __name__ == '__main__':
    log.info('Server on :3000')
    app.run(host='0.0.0.0', port=3000, threaded=False)
