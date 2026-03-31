const { chromium } = require('@playwright/test');
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const PORT = 7654;
const PLY_FILE = process.argv[2] || '/references/aidemo1.ply';
const OUT = process.argv[3] || path.join(ROOT, '.claude/golden/splat_verify_screenshot.png');
const TIMEOUT_MS = 90_000;

// Minimal static file server
const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.ply': 'application/octet-stream',
  '.map': 'application/json',
  '.cjs': 'application/javascript',
};

const CORS_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

const server = http.createServer((req, res) => {
  const filePath = path.join(ROOT, req.url.split('?')[0]);
  if (!filePath.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.error('404:', filePath);
      res.writeHead(404, CORS_HEADERS); res.end('Not found: ' + filePath); return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', ...CORS_HEADERS });
    res.end(data);
  });
});

(async () => {
  await new Promise(r => server.listen(PORT, r));
  console.log(`Server: http://localhost:${PORT}`);

  const browser = await chromium.launch({ headless: false }); // headless:false so you can watch
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1920, height: 1080 });

  page.on('console', m => console.log('[browser]', m.text()));
  page.on('pageerror', e => console.error('[page error]', e.message));

  const side = process.argv[4] || '1'; // default to -Z (flipped)
  const url = `http://localhost:${PORT}/viewer.html?file=${PLY_FILE}&side=${side}`;
  console.log('Navigating to:', url);
  await page.goto(url);

  console.log(`Waiting for render (up to ${TIMEOUT_MS / 1000}s)...`);
  try {
    await page.waitForFunction(() => window.renderComplete === true, { timeout: TIMEOUT_MS });
    console.log('Render complete — taking screenshot');

    // Extra half-second for final sort pass to settle
    await page.waitForTimeout(500);
    await page.screenshot({ path: OUT, fullPage: false });
    console.log('Screenshot saved:', OUT);
  } catch (e) {
    const errMsg = await page.evaluate(() => window.renderError);
    console.error('Timed out or errored. renderError:', errMsg);
    await page.screenshot({ path: OUT });
    console.log('Partial screenshot saved:', OUT);
  }

  await browser.close();
  server.close();
})();
