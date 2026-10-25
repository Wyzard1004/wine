'use strict';

const $ = (selector) => document.querySelector(selector);
const COLORS = {1: '#7c3c50', 2: '#829880', 3: '#c3a265'};
const state = {data: null, page: 0, pageSize: 12, visibleClasses: new Set([1, 2, 3])};
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const percent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
const number = (value, digits = 2) => Number(value).toLocaleString('en', {maximumFractionDigits: digits});
const shortName = (model) => model.id === 'baseline' ? 'All 13 features' : `${model.components} components`;
const deltaText = (value) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} pp`;
const svgText = (x, y, text, options = '') => `<text x="${x}" y="${y}" fill="#626c5c" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="9" ${options}>${escapeHTML(text)}</text>`;

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach((panel) => {panel.hidden = panel.id !== name;});
  document.querySelectorAll('.tab').forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('active', active);
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  history.replaceState(null, '', `#${name}`);
  hideTooltip();
}

function renderSummary() {
  const {dataset, models} = state.data;
  const baseline = models.find((model) => model.id === 'baseline');
  const best = models.filter((model) => model.id !== 'baseline').reduce((a, b) => a.mean_accuracy >= b.mean_accuracy ? a : b);
  $('#sample-count').textContent = number(dataset.n_samples, 0);
  $('#feature-count').textContent = number(dataset.n_features, 0);
  $('#sample-note').textContent = `${dataset.n_classes} cultivars · ${dataset.missing_values} missing values`;
  $('#baseline-accuracy').textContent = percent(baseline.mean_accuracy);
  $('#best-accuracy').textContent = percent(best.mean_accuracy);
  const correct = (model) => model.confusion_matrix.reduce((sum, row, i) => sum + row[i], 0);
  $('#best-note').textContent = correct(best) === correct(baseline)
    ? `${best.components} PCs · same total correct as baseline`
    : `${best.components} PCs · ${deltaText(best.delta_vs_baseline)} vs. baseline`;
  const retained = state.data.pca.cumulative_variance[best.components - 1];
  $('#finding').innerHTML = correct(best) === correct(baseline)
    ? `<strong>${best.components} components, ${percent(retained, 1)} variance retained.</strong> Both this model and the baseline classify ${correct(best)} of ${dataset.n_samples} samples correctly. A near tie, with fewer dimensions.`
    : `<strong>${best.components} components, ${percent(retained, 1)} variance retained.</strong> ${deltaText(best.delta_vs_baseline)} versus the baseline in mean fold accuracy. Small differences need cautious interpretation.`;
  const ks = [...new Set(state.data.knn_sensitivity.map((row) => row.k))].sort((a, b) => a - b);
  $('#k-select').innerHTML = ks.map((k) => `<option value="${k}" ${k === 5 ? 'selected' : ''}>k = ${k}</option>`).join('');
  $('#model-select').innerHTML = models.map((model) => `<option value="${model.id}">${escapeHTML(shortName(model))}</option>`).join('');
  $('#class-toggles').innerHTML = dataset.class_counts.map((item) => `<button class="class-toggle" data-class="${item.class_id}" aria-pressed="true"><i class="swatch" style="background:${COLORS[item.class_id]}"></i>${escapeHTML(item.label)}</button>`).join('');
  $('#results-body').innerHTML = models.map((model) => `<tr><td>${escapeHTML(shortName(model))}${model.id === 'baseline' ? '<span class="reference-tag">BASELINE</span>' : ''}</td><td>${model.components}</td><td>${percent(model.mean_accuracy)}</td><td>±${(model.std_accuracy * 100).toFixed(2)} pp</td><td class="${model.delta_vs_baseline >= 0 ? 'positive' : 'negative'}">${model.id === 'baseline' ? 'Reference' : deltaText(model.delta_vs_baseline)}</td></tr>`).join('');
}

function renderAccuracy() {
  const k = Number($('#k-select').value);
  const models = [...state.data.models].sort((a, b) => a.components - b.components);
  const rows = models.map((model) => {
    const score = state.data.knn_sensitivity.find((row) => row.model_id === model.id && row.k === k);
    return {...model, ...score};
  });
  const width = 540, height = 274, left = 47, right = 25, top = 29, bottom = 48;
  const low = Math.max(0, Math.floor((Math.min(...rows.map((row) => row.mean_accuracy - row.std_accuracy)) - 0.02) * 20) / 20);
  const high = Math.max(1.01, ...rows.map((row) => row.mean_accuracy + row.std_accuracy + 0.01));
  const x = (i) => left + 35 + i * (width - left - right - 70) / (rows.length - 1);
  const y = (value) => top + (high - value) / (high - low) * (height - top - bottom);
  let contents = `<title>Mean accuracy by representation, k equals ${k}</title><desc>${rows.map((row) => `${shortName(row)}: ${percent(row.mean_accuracy)}, sample standard deviation ${(row.std_accuracy * 100).toFixed(2)} percentage points`).join('. ')}. Vertical axis starts at ${percent(low, 0)}.</desc>`;
  for (let i = 0; i <= 4; i++) {
    const value = low + (1 - low) * i / 4;
    contents += `<line x1="${left}" x2="${width - right}" y1="${y(value)}" y2="${y(value)}" stroke="#edeee7" stroke-dasharray="3 4"/>${svgText(left - 10, y(value) + 3, percent(value, 0), 'text-anchor="end"')}`;
  }
  const baseline = rows.find((row) => row.id === 'baseline');
  contents += `<line x1="${left}" x2="${width - right}" y1="${y(baseline.mean_accuracy)}" y2="${y(baseline.mean_accuracy)}" stroke="#9aaa91" stroke-dasharray="5 5" opacity=".7"/>`;
  const line = rows.filter((row) => row.id !== 'baseline').map((row) => `${x(rows.indexOf(row))},${y(row.mean_accuracy)}`).join(' ');
  contents += `<polyline points="${line}" fill="none" stroke="#a37483" stroke-width="1.8"/>`;
  rows.forEach((row, i) => {
    const center = x(i), color = row.id === 'baseline' ? '#4f695c' : '#78384b';
    const y0 = y(row.mean_accuracy - row.std_accuracy);
    const y1 = y(row.mean_accuracy + row.std_accuracy);
    const hint = `${shortName(row)}|Mean ${percent(row.mean_accuracy)} · SD ${(row.std_accuracy * 100).toFixed(2)} pp|k = ${k}`;
    contents += `<g data-tooltip="${escapeHTML(hint)}" tabindex="0" aria-label="${escapeHTML(hint.replaceAll('|', '. '))}"><line x1="${center}" x2="${center}" y1="${y0}" y2="${y1}" stroke="${color}" stroke-width="1.5"/><line x1="${center - 5}" x2="${center + 5}" y1="${y0}" y2="${y0}" stroke="${color}"/><line x1="${center - 5}" x2="${center + 5}" y1="${y1}" y2="${y1}" stroke="${color}"/><circle cx="${center}" cy="${y(row.mean_accuracy)}" r="6" fill="${color}" stroke="white" stroke-width="2"/><circle cx="${center}" cy="${y(row.mean_accuracy)}" r="15" fill="transparent"/>${svgText(center, y1 - 10, percent(row.mean_accuracy, 2), `text-anchor="middle" style="fill:${color};font-size:11px;font-weight:600"`)}</g>`;
    contents += svgText(center, height - 25, row.id === 'baseline' ? '13 features' : `${row.components} PCs`, 'text-anchor="middle"');
    contents += svgText(center, height - 11, row.id === 'baseline' ? 'BASELINE' : 'PCA', 'text-anchor="middle" style="font-size:7px;letter-spacing:1px;fill:#626c5c"');
  });
  $('#accuracy-chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="group" aria-label="Accuracy comparison, k equals ${k}">${contents}</svg>`;
  $('#accuracy-note').textContent = `Mean ±1 sample SD across the same five folds. Axis starts at ${percent(low, 0)}. ${k === 5 ? 'Reference experiment: k = 5.' : `Sensitivity analysis: k = ${k}; reference table and confusion matrices remain k = 5.`}`;
}

function scatterSVG(points, xLabel, yLabel, {showPC = false} = {}) {
  const width = 470, height = 298, left = 49, right = 14, top = 17, bottom = 47;
  const allPoints = points;
  const minMax = (key) => {const values = allPoints.map((point) => point[key]); const min = Math.min(...values), max = Math.max(...values); const pad = (max - min) * 0.1 || 1; return [min - pad, max + pad];};
  const [xmin, xmax] = minMax('x'), [ymin, ymax] = minMax('y');
  const x = (value) => left + (value - xmin) / (xmax - xmin) * (width - left - right);
  const y = (value) => top + (ymax - value) / (ymax - ymin) * (height - top - bottom);
  let contents = `<title>${escapeHTML(xLabel)} versus ${escapeHTML(yLabel)}</title><desc>${allPoints.length} wine samples coloured by cultivar. Inspect every sample in Explore data.</desc>`;
  for (let i = 0; i <= 4; i++) {
    const xv = xmin + (xmax - xmin) * i / 4, yv = ymin + (ymax - ymin) * i / 4;
    contents += `<line x1="${left}" x2="${width - right}" y1="${y(yv)}" y2="${y(yv)}" stroke="#f0f1eb"/>${svgText(left - 10, y(yv) + 3, number(yv, Math.abs(ymax) < 10 ? 1 : 0), 'text-anchor="end"')}`;
    contents += svgText(x(xv), height - bottom + 18, number(xv, Math.abs(xmax) < 10 ? 1 : 0), 'text-anchor="middle"');
  }
  if (showPC) {
    contents += `<line x1="${x(0)}" x2="${x(0)}" y1="${top}" y2="${height - bottom}" stroke="#dadfd2" stroke-dasharray="3 4"/><line x1="${left}" x2="${width - right}" y1="${y(0)}" y2="${y(0)}" stroke="#dadfd2" stroke-dasharray="3 4"/>`;
  }
  points.filter((point) => !showPC || state.visibleClasses.has(point.class_id)).forEach((point) => {
    const label = `Sample ${point.sample_id} · Cultivar ${point.class_id}|${xLabel}: ${number(point.x, 3)}|${yLabel}: ${number(point.y, 3)}`;
    contents += `<circle cx="${x(point.x)}" cy="${y(point.y)}" r="3.8" fill="${COLORS[point.class_id]}" fill-opacity=".78" stroke="white" stroke-width=".7" tabindex="0" data-tooltip="${escapeHTML(label)}" aria-label="${escapeHTML(label.replaceAll('|', '. '))}"><title>${escapeHTML(label.replaceAll('|', '. '))}</title></circle>`;
  });
  if (showPC && state.visibleClasses.size === 0) contents += svgText(width / 2, height / 2, 'Select a cultivar above to show its samples.', 'text-anchor="middle"');
  contents += svgText((left + width - right) / 2, height - 5, xLabel, 'text-anchor="middle" style="font-size:9px;fill:#626c5c"');
  contents += svgText(11, (top + height - bottom) / 2, yLabel, `text-anchor="middle" transform="rotate(-90 11 ${(top + height - bottom) / 2})" style="font-size:9px;fill:#626c5c"`);
  return `<svg viewBox="0 0 ${width} ${height}" role="group" aria-label="${escapeHTML(xLabel)} and ${escapeHTML(yLabel)} wine sample scatter plot">${contents}</svg>`;
}

function renderPCA() {
  const {pca} = state.data;
  $('#scatter-chart').innerHTML = scatterSVG(pca.projection.map((point) => ({...point, x: point.pc1, y: point.pc2})), `PC 1 · ${percent(pca.explained_variance_ratio[0], 1)} variance`, `PC 2 · ${percent(pca.explained_variance_ratio[1], 1)} variance`, {showPC: true});
}

function renderVariance() {
  const {cumulative_variance: values} = state.data.pca;
  $('#variance-chart').innerHTML = [2, 5, 10].map((count) => `<div class="variance-row"><span class="variance-label">${count} PCs</span><div class="variance-track" role="img" aria-label="${count} principal components retain ${percent(values[count - 1], 1)} variance"><div class="variance-fill" style="width:${values[count - 1] * 100}%"></div></div><span class="variance-number">${percent(values[count - 1], 1)}</span></div>`).join('') + `<div class="variance-foot"><span>5 PCs reduce dimensions by</span><strong>${percent(1 - 5 / state.data.dataset.n_features, 1)}</strong></div>`;
}

function renderConfusion() {
  const model = state.data.models.find((item) => item.id === $('#model-select').value);
  const max = Math.max(...model.confusion_matrix.flat());
  $('#confusion-matrix').innerHTML = `<div class="matrix-wrap"><span class="matrix-label" aria-hidden="true">Actual cultivar</span><table class="matrix"><caption class="sr-only">${escapeHTML(model.label)}, k equals 5. Rows are actual classes and columns are predicted classes.</caption><thead><tr><th><span class="sr-only">Actual / predicted</span></th>${[1, 2, 3].map((id) => `<th scope="col">Cultivar ${id}</th>`).join('')}</tr></thead><tbody>${model.confusion_matrix.map((row, i) => `<tr><th scope="row">Cultivar ${i + 1}</th>${row.map((value) => {const alpha = .06 + value / max * .84; const luminance = [120, 56, 75].map((channel) => (channel * alpha + 255 * (1 - alpha)) / 255).map((channel) => channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4).reduce((sum, channel, index) => sum + channel * [.2126, .7152, .0722][index], 0); return `<td style="background:rgba(120,56,75,${alpha});color:${luminance <= .179 ? '#fff' : '#000'}">${value}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div><div class="matrix-bottom">Predicted cultivar</div>`;
}

function renderExplore() {
  const {dataset} = state.data;
  const options = dataset.feature_names.map((name, i) => `<option value="${escapeHTML(name)}">${escapeHTML(dataset.feature_labels[i])}</option>`).join('');
  $('#x-feature').innerHTML = options;
  $('#y-feature').innerHTML = options;
  $('#x-feature').value = 'alcohol';
  $('#y-feature').value = 'flavanoids';
  $('#class-filter').innerHTML += dataset.class_counts.map((item) => `<option value="${item.class_id}">${escapeHTML(item.label)}</option>`).join('');
  $('#data-head').innerHTML = `<tr><th scope="col">Sample</th><th scope="col">Cultivar</th>${dataset.feature_labels.map((label) => `<th scope="col">${escapeHTML(label)}</th>`).join('')}</tr>`;
  $('#class-breakdown').innerHTML = dataset.class_counts.map((item) => `<div><span><i class="swatch" style="background:${COLORS[item.class_id]}"></i>${escapeHTML(item.label)}</span><strong>${item.count} samples</strong></div>`).join('');
  renderFeatureChart();
  renderData();
}

function renderFeatureChart() {
  const {dataset} = state.data;
  const xFeature = $('#x-feature').value, yFeature = $('#y-feature').value;
  const label = (feature) => dataset.feature_labels[dataset.feature_names.indexOf(feature)];
  $('#feature-chart').innerHTML = scatterSVG(dataset.records.map((record) => ({...record, x: record[xFeature], y: record[yFeature]})), label(xFeature), label(yFeature));
  const summary = dataset.feature_summary.find((row) => row.name === xFeature);
  $('#profile-name').textContent = summary.label;
  $('#feature-profile').innerHTML = [['Mean', summary.mean], ['Standard deviation', summary.std], ['Minimum', summary.min], ['Maximum', summary.max]].map(([name, value]) => `<div><dt>${name}</dt><dd>${number(value, 3)}</dd></div>`).join('');
}

function renderData() {
  const {dataset} = state.data;
  const query = $('#sample-search').value.trim().toLowerCase();
  const filter = $('#class-filter').value;
  const rows = dataset.records.filter((record) => (filter === 'all' || record.class_id === Number(filter)) && `sample ${record.sample_id} #${String(record.sample_id).padStart(3, '0')} cultivar ${record.class_id} class ${record.class_id}`.includes(query));
  const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * state.pageSize;
  const pageRows = rows.slice(start, start + state.pageSize);
  $('#data-body').innerHTML = pageRows.length ? pageRows.map((record) => `<tr><td>#${String(record.sample_id).padStart(3, '0')}</td><td><i class="swatch" style="background:${COLORS[record.class_id]};margin-right:6px"></i>Cultivar ${record.class_id}</td>${dataset.feature_names.map((name) => `<td>${number(record[name], 3)}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="15" class="empty-cell">No samples match your search. Try another sample ID or cultivar.</td></tr>`;
  $('#page-status').textContent = rows.length ? `Showing ${start + 1}–${Math.min(start + state.pageSize, rows.length)} of ${rows.length} samples` : '0 matching samples';
  $('#prev-page').disabled = state.page === 0;
  $('#next-page').disabled = state.page >= pages - 1;
}

function renderMethods() {
  const {metadata, nested_cv, models} = state.data;
  const settings = [['Evaluation', `${metadata.n_splits}-fold stratified CV`], ['Shuffle seed', metadata.seed], ['Reference neighbours', `k = ${metadata.k}`], ['Distance', 'Euclidean'], ['Neighbour weights', 'Uniform'], ['Standardisation', 'Within each training fold'], ['PCA whitening', 'Off'], ['Missing values', state.data.dataset.missing_values], ['scikit-learn', metadata.versions.scikit_learn]];
  $('#method-settings').innerHTML = settings.map(([name, value]) => `<div><dt>${escapeHTML(name)}</dt><dd>${escapeHTML(value)}</dd></div>`).join('');
  $('#nested-description').textContent = nested_cv.description;
  $('#nested-body').innerHTML = nested_cv.models.map((model) => `<tr><td>${escapeHTML(shortName(models.find((item) => item.id === model.id)))}</td><td>${percent(model.mean_accuracy)}</td><td>±${(model.std_accuracy * 100).toFixed(2)} pp</td><td>${model.selected_k.join(' · ')}</td></tr>`).join('');
}

function hideTooltip() {$('#tooltip').hidden = true;}
function showTooltip(target, event) {
  const tooltip = $('#tooltip');
  const parts = target.dataset.tooltip.split('|');
  tooltip.innerHTML = parts.map((part, i) => i === 0 ? `<strong>${escapeHTML(part)}</strong>` : escapeHTML(part)).join('<br>');
  tooltip.hidden = false;
  const rect = target.getBoundingClientRect();
  const x = event.clientX || rect.left + rect.width / 2;
  const y = event.clientY || rect.top + rect.height / 2;
  tooltip.style.left = `${Math.max(8, Math.min(x + 13, window.innerWidth - tooltip.offsetWidth - 12))}px`;
  tooltip.style.top = `${Math.max(8, Math.min(y + 13, window.innerHeight - tooltip.offsetHeight - 12))}px`;
}

function bindEvents() {
  document.querySelectorAll('[data-tab]').forEach((button) => button.addEventListener('click', () => switchTab(button.dataset.tab)));
  document.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => {switchTab(button.dataset.go); window.scrollTo({top: 0, behavior: 'smooth'});}));
  $('#k-select').addEventListener('change', renderAccuracy);
  $('#model-select').addEventListener('change', renderConfusion);
  $('#class-toggles').addEventListener('click', (event) => {
    const button = event.target.closest('[data-class]');
    if (!button) return;
    const id = Number(button.dataset.class);
    if (state.visibleClasses.has(id)) state.visibleClasses.delete(id); else state.visibleClasses.add(id);
    button.setAttribute('aria-pressed', String(state.visibleClasses.has(id)));
    renderPCA();
  });
  ['#x-feature', '#y-feature'].forEach((id) => $(id).addEventListener('change', renderFeatureChart));
  ['#sample-search', '#class-filter'].forEach((id) => $(id).addEventListener(id === '#sample-search' ? 'input' : 'change', () => {state.page = 0; renderData();}));
  $('#prev-page').addEventListener('click', () => {state.page -= 1; renderData();});
  $('#next-page').addEventListener('click', () => {state.page += 1; renderData();});
  document.addEventListener('pointerover', (event) => {const target = event.target.closest('[data-tooltip]'); if (target) showTooltip(target, event);});
  document.addEventListener('pointerout', (event) => {if (event.target.closest('[data-tooltip]')) hideTooltip();});
  document.addEventListener('focusin', (event) => {const target = event.target.closest('[data-tooltip]'); if (target) showTooltip(target, event);});
  document.addEventListener('focusout', hideTooltip);
  document.addEventListener('keydown', (event) => {if (event.key === 'Escape') hideTooltip();});
  window.addEventListener('scroll', hideTooltip, {passive: true});
  window.addEventListener('hashchange', () => {const tab = location.hash.slice(1); if (['overview', 'explore', 'methods'].includes(tab)) switchTab(tab);});
}

async function init() {
  try {
    const response = await fetch('../results/analysis.json');
    if (!response.ok) throw new Error(`Analysis HTTP ${response.status}`);
    state.data = await response.json();
    if (!state.data.dataset?.records?.length || state.data.models?.length !== 4) throw new Error('Analysis is incomplete');
    renderSummary();
    renderAccuracy();
    renderPCA();
    renderVariance();
    renderConfusion();
    renderExplore();
    renderMethods();
    bindEvents();
    const tab = location.hash.slice(1);
    switchTab(['overview', 'explore', 'methods'].includes(tab) ? tab : 'overview');
    $('#loading').hidden = true;
    $('#app').hidden = false;
  } catch (error) {
    $('#loading').hidden = true;
    $('#error').hidden = false;
    console.error('Unable to load wine analysis:', error);
  }
}

init();
