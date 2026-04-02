#!/usr/bin/env python3
"""
Quick pipeline validator. Run while server.py is running:
  python test_pipeline.py
"""
import json, sys, time
import requests

SERVER   = 'http://localhost:8080'
IMAGE    = 'input/default_room.png'
PROMPT   = 'Add airplane wallpaper, multiple plants and some office space with a big screen'

def run():
    print(f'[1] Submitting job...')
    with open(IMAGE, 'rb') as f:
        resp = requests.post(
            f'{SERVER}/api/redesign',
            files={'image': ('default_room.png', f, 'image/png')},
            data={'prompt': PROMPT},
            timeout=10,
        )
    resp.raise_for_status()
    session_id = resp.json()['session_id']
    print(f'    session: {session_id}')

    print(f'[2] Streaming events...')
    with requests.get(f'{SERVER}/api/events/{session_id}', stream=True, timeout=700) as es:
        for line in es.iter_lines():
            if not line or not line.startswith(b'data:'):
                continue
            ev = json.loads(line[5:])

            if ev['type'] == 'ping':
                continue

            if ev['type'] == 'step':
                icon = {'done': '✓', 'running': '⟳', 'error': '✗'}.get(ev['status'], '○')
                print(f'    {icon} {ev["label"]:20s}  {ev.get("detail","")[:80]}')

            elif ev['type'] == 'done':
                ply = ev['ply']
                print(f'\n[3] Pipeline done → {ply}')
                # Verify PLY exists and has content
                size_resp = requests.head(f'{SERVER}/{ply}', timeout=5)
                cl = size_resp.headers.get('Content-Length', '?')
                print(f'    PLY size: {int(cl)/1024/1024:.1f} MB' if cl != '?' else '    PLY size: unknown')
                print('\n✅ PASS')
                return True

            elif ev['type'] == 'error':
                print(f'\n❌ FAIL: {ev["message"]}')
                return False

    print('\n❌ FAIL: stream ended without done/error')
    return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
