/* Print the report only after every figure has loaded. */
const assert = require('node:assert/strict');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require('playwright');

async function main() {
  const root = path.resolve(__dirname, '..');
  const url = process.env.REPORT_URL || 'http://127.0.0.1:8766/docs/report.html';
  let server, browser;
  try {
    if (!process.env.REPORT_URL) {
      server = spawn(process.env.PYTHON || 'python3', ['-m', 'http.server', '8766', '--bind', '127.0.0.1'], {cwd: root, stdio: 'ignore'});
      server.on('error', error => console.error(error.message));
    }
    let ready = false;
    for (let i = 0; i < 50; i++) {
      try {ready = (await fetch(url)).ok;} catch {}
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert(ready, 'The formatted report could not be served.');
    browser = await chromium.launch({executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined});
    const page = await browser.newPage();
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    assert.equal(await page.locator('figure img').count(), 6, 'The paper must include all six scientific figures.');
    const output = path.join(root, 'docs/report.pdf');
    await page.pdf({path: output, format: 'A4', preferCSSPageSize: true, printBackground: true, tagged: true});
    console.log(`Exported ${output} with all six figures.`);
  } finally {
    if (browser) await browser.close();
    if (server) server.kill();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
