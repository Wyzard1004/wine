/* Optional browser checks. Run npm ci && npx playwright install chromium first. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require('playwright');

const root = path.resolve(__dirname, '..');
const results = JSON.parse(fs.readFileSync(path.join(root, 'results/analysis.json'), 'utf8'));
const artifacts = process.env.DASHBOARD_ARTIFACT_DIR || path.join(root, 'test-results');
const url = process.env.DASHBOARD_URL || 'http://127.0.0.1:8765/dashboard/';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  let server, browser;
  try {
    if (!process.env.DASHBOARD_URL) {
      server = spawn(process.env.PYTHON || 'python3', ['-m', 'http.server', '8765', '--bind', '127.0.0.1'], {cwd: root, stdio: 'ignore'});
      server.on('error', error => {console.error(error.message);});
    }
    let ready = false;
    for (let i = 0; i < 50; i++) {
      try {ready = (await fetch(url)).ok;} catch {}
      if (ready) break;
      await sleep(100);
    }
    assert(ready, `Dashboard did not start: ${url}`);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE_PATH || undefined});
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    await page.goto(url);
    await page.locator('#app').waitFor({state: 'visible'});
    const text = selector => page.locator(selector).innerText();
    assert.equal((await text('#sample-count')).replaceAll(',', ''), String(results.dataset.n_samples));
    assert.equal(await text('#baseline-accuracy'), `${(results.models[0].mean_accuracy * 100).toFixed(2)}%`);
    assert.match(await text('#best-note'), /same total correct/);
    assert.equal(await page.locator('#scatter-chart circle').count(), 178);
    assert.equal(await page.locator('#results-body tr').count(), 4);

    // Sensitivity changes the plot while preserving the fixed-k reference table.
    const reference = await text('#results-body');
    const initialChart = await page.locator('#accuracy-chart').innerHTML();
    await page.selectOption('#k-select', '11');
    assert.notEqual(await page.locator('#accuracy-chart').innerHTML(), initialChart);
    assert.match(await text('#accuracy-note'), /Sensitivity analysis: k = 11/);
    assert.equal(await text('#results-body'), reference);
    await page.selectOption('#k-select', '5');
    for (const model of results.models) {
      await page.selectOption('#model-select', model.id);
      const cells = await page.locator('#confusion-matrix td').allTextContents();
      assert.deepEqual(cells.map(Number), model.confusion_matrix.flat());
    }
    await page.selectOption('#model-select', 'baseline');

    // Class toggles, their empty state, and keyboard-driven tooltips.
    for (const id of [1, 2, 3]) await page.locator(`[data-class="${id}"]`).click();
    assert.equal(await page.locator('#scatter-chart circle').count(), 0);
    assert.match(await text('#scatter-chart'), /Select a cultivar/);
    for (const id of [1, 2, 3]) await page.locator(`[data-class="${id}"]`).click();
    await page.locator('#scatter-chart circle').first().focus();
    assert(await page.locator('#tooltip').isVisible());
    await page.keyboard.press('Escape');
    assert(await page.locator('#tooltip').isHidden());

    // Data pagination, filtering, no-match recovery, and feature selection.
    await page.locator('[data-tab="explore"]').click();
    assert.match(await text('#page-status'), /of 178 samples/);
    const firstRow = await text('#data-body tr:first-child');
    await page.locator('#next-page').click();
    assert.notEqual(await text('#data-body tr:first-child'), firstRow);
    await page.selectOption('#class-filter', '3');
    assert.match(await text('#page-status'), /of 48 samples/);
    for (const row of await page.locator('#data-body tr').allTextContents()) assert.match(row, /Cultivar 3/);
    await page.locator('#sample-search').fill('178');
    assert.equal(await page.locator('#data-body tr').count(), 1);
    assert.match(await text('#data-body'), /#178/);
    await page.locator('#sample-search').fill('no-such-sample');
    assert.equal(await text('#page-status'), '0 matching samples');
    assert(await page.locator('#prev-page').isDisabled());
    assert(await page.locator('#next-page').isDisabled());
    await page.locator('#sample-search').fill('');
    await page.selectOption('#class-filter', 'all');
    await page.selectOption('#x-feature', 'proline');
    assert.equal(await text('#profile-name'), 'Proline');
    assert.equal(await page.locator('#feature-chart circle').count(), 178);

    // Methods and all local report/data/download destinations.
    await page.locator('[data-tab="methods"]').click();
    assert.equal(await page.locator('#nested-body tr').count(), 4);
    const links = await page.locator('a[href]').evaluateAll(nodes => [...new Set(nodes.map(node => node.href))]);
    for (const link of links.filter(link => new URL(link).origin === new URL(url).origin)) {
      const response = await page.request.get(link);
      assert(response.ok(), `Broken destination: ${link}`);
    }
    const report = await browser.newPage();
    await report.goto(new URL('../docs/report.html', url).href, {waitUntil: 'networkidle'});
    await report.evaluate(() => Promise.all([...document.images].map(image => image.decode())));
    assert.equal(await report.locator('figure img').count(), 6, 'All report figures must load.');
    assert.equal(await report.locator('h1').count(), 1);
    await report.close();
    await page.locator('[data-tab="overview"]').click();
    await page.evaluate(() => window.scrollTo(0, 0));
    fs.mkdirSync(artifacts, {recursive: true});
    await page.screenshot({path: path.join(artifacts, 'dashboard-desktop.png'), fullPage: true});

    // Narrow-screen navigation and containment of the wide data table.
    await page.setViewportSize({width: 390, height: 844});
    for (const tab of ['overview', 'explore', 'methods']) {
      await page.locator(`[data-tab="${tab}"]`).click();
      assert(await page.locator(`#${tab}`).isVisible());
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow, false, `Horizontal page overflow on mobile: ${tab}`);
    }
    await page.locator('[data-tab="overview"]').click();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({path: path.join(artifacts, 'dashboard-mobile.png'), fullPage: true});
    assert.deepEqual(errors, [], 'Unexpected browser errors');

    const failed = await browser.newPage();
    await failed.route('**/results/analysis.json', route => route.fulfill({status: 404, body: 'Missing'}));
    await failed.goto(url);
    await failed.locator('#error').waitFor({state: 'visible'});
    assert(await failed.locator('#loading').isHidden());
    assert(await failed.locator('#app').isHidden());
    console.log('Dashboard checks passed: result accuracy, sensitivity, matrices, class toggles, tooltips, data explorer, links, mobile layout, and error state.');
  } finally {
    if (browser) await browser.close();
    if (server) server.kill();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
