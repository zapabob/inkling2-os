    const $ = (id) => document.getElementById(id);

    // Theme switch
    $('btn-theme').addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('inkling2.theme', next); } catch {}
    });

    // Mobile sidebar toggle
    $('btn-sidebar').addEventListener('click', () => {
      $('sidebar').classList.toggle('open');
    });

    // Segmented view switcher
    document.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.view;
        document.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + target));
        $('sidebar').classList.remove('open');
      });
    });

    let consoleToken = '';
    let localConsole = false;
    let usingUpstream = false;
    async function api(path, opts = {}) {
      const headers = new Headers(opts.headers);
      if (opts.body) headers.set('Content-Type', 'application/json');
      if (consoleToken) headers.set('Authorization', 'Bearer ' + consoleToken);
      let res;
      try {
        res = await fetch(path, { ...opts, headers, redirect: 'error', cache: 'no-store' });
      } catch {
        return { ok: false, status: 0, data: { error: 'Console connection lost' } };
      }
      const text = await res.text();
      try { return { ok: res.ok, status: res.status, data: JSON.parse(text) }; }
      catch { return { ok: res.ok, status: res.status, data: { raw: text } }; }
    }

    const accessDialog = $('access-dialog');
    $('btn-access').addEventListener('click', () => {
      if (consoleToken) {
        consoleToken = '';
        location.reload();
        return;
      }
      accessDialog.showModal();
      $('access-token').focus();
    });
    $('access-cancel').addEventListener('click', () => accessDialog.close());
    accessDialog.addEventListener('close', () => { $('access-token').value = ''; });
    $('access-confirm').addEventListener('click', async () => {
      const btn = $('access-confirm');
      btn.disabled = true;
      consoleToken = $('access-token').value.trim();
      try {
        const result = await api('/v1/session');
        if (!result.ok) {
          consoleToken = '';
          $('access-status').textContent = 'The console credential was rejected.';
          return;
        }
        usingUpstream = result.data.upstream;
        $('btn-lora-step').disabled = !result.data.experimental_lora;
        $('btn-access').textContent = 'Lock';
        accessDialog.close();
        refreshHealthAndState();
        refreshGpu();
        refreshModels();
        loadFsDir('');
      } catch {
        consoleToken = '';
        $('access-status').textContent = 'The console service is unavailable.';
      } finally {
        btn.disabled = false;
      }
    });
    $('access-token').addEventListener('keydown', event => {
      if (event.key === 'Enter') $('access-confirm').click();
    });

    async function refreshHealthAndState() {
      try {
        const [h, mem, lora] = await Promise.all([
          api('/health'),
          api('/v1/memory/stats'),
          api('/v1/lora/stats'),
        ]);
        const pill = $('health-pill');
        if (h.ok && h.data.status === 'ok') {
          pill.classList.remove('offline');
          const needsToken = h.data.console_mode === 'locked';
          localConsole = h.data.console_mode === 'local';
          $('health-label').textContent = localConsole || consoleToken ? 'Console online' : needsToken ? 'Console locked' : 'Preview · access unconfigured';
          $('btn-access').hidden = !needsToken;
        } else {
          pill.classList.add('offline');
          $('health-label').textContent = 'Offline';
        }
        if (mem.ok) {
          const usedMib = (mem.data.serialized_bytes / (1024 * 1024)).toFixed(2);
          const capMib = (mem.data.budget_bytes / (1024 * 1024)).toFixed(0);
          const pct = Math.min(100, (mem.data.serialized_bytes / Math.max(1, mem.data.budget_bytes)) * 100);
          $('sb-mem-entries').textContent = mem.data.entries + ' entries';
          $('sb-mem-bytes').textContent = `${usedMib} / ${capMib} MiB`;
          $('sb-mem-bar').style.width = Math.max(2, pct) + '%';
        }
        if (lora.ok) {
          $('sb-lora-steps').textContent = 'step ' + lora.data.steps;
          $('lora-steps').textContent = lora.data.steps;
          $('lora-reward').textContent = Number(lora.data.cumulative_reward || 0).toFixed(4);
          $('lora-norms').textContent = `${Number(lora.data.norm_a || 0).toFixed(4)} / ${Number(lora.data.norm_b || 0).toFixed(4)}`;
        }
      } catch {
        $('health-pill').classList.add('offline');
        $('health-label').textContent = 'Offline';
      }
    }

    async function refreshGpu() {
      const r = await api('/v1/gpu/info');
      if (!r.ok || !r.data.available) {
        $('sb-gpu-name').textContent = r.status === 401 ? 'Locked' : 'Unavailable';
        return;
      }
      const g = r.data;
      const shortName = (g.name || 'GPU').replace('NVIDIA GeForce ', '');
      const vramPct = Math.min(100, Math.round((g.memory_used_mib / Math.max(1, g.memory_total_mib)) * 100));
      $('sb-gpu-name').textContent = shortName;
      $('sb-gpu-vram').textContent = `${g.memory_used_mib} / ${g.memory_total_mib} MiB`;
      $('sb-vram-bar').style.width = vramPct + '%';
      $('sb-gpu-util').textContent = `${g.utilization_gpu}% · ${g.temperature_c}°C`;
      $('sb-gpu-pwr').textContent = `${g.power_draw_w.toFixed(1)}W · ${g.clock_sm_mhz} MHz`;

      $('tc-vram-val').innerHTML = `${(g.memory_used_mib / 1024).toFixed(2)}<span class="metric-unit">/ ${(g.memory_total_mib / 1024).toFixed(1)} GiB</span>`;
      $('tc-vram-sub').textContent = `${shortName} · Driver ${g.driver_version}`;
      $('tc-temp-val').innerHTML = `${g.temperature_c}<span class="metric-unit">°C</span>`;
      $('tc-util-sub').textContent = `SM Utilization: ${g.utilization_gpu}%`;
      $('tc-pwr-val').innerHTML = `${g.power_draw_w.toFixed(1)}<span class="metric-unit">W</span>`;
      $('tc-clk-sub').textContent = `Limit ${g.power_limit_w.toFixed(0)}W · SM ${g.clock_sm_mhz} MHz`;
    }

    async function refreshModels() {
      const r = await api('/v1/models');
      const sel = $('model-select');
      const list = $('models-list');
      if (!r.ok || !Array.isArray(r.data.data) || r.data.data.length === 0) {
        sel.replaceChildren(new Option(r.status === 401 ? 'Unlock to view models' : 'No local .strata found', ''));
        list.textContent = 'No .strata models discovered under INKLING2_MODELS_DIR.';
        return;
      }
      sel.innerHTML = '';
      list.innerHTML = '';
      r.data.data.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        sel.appendChild(opt);

        const gib = (m.bytes / (1024 ** 3)).toFixed(2);
        const div = document.createElement('div');
        div.className = 'list-item';
        div.innerHTML = `<div><strong>${escapeHtml(m.id)}</strong><div style="font-size:12px;color:var(--text-tertiary);">${escapeHtml(m.owned_by || 'SC117')}</div></div><span class="kv-val">${gib} GiB</span>`;
        list.appendChild(div);
      });
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    // Chat
    function appendMessage(role, content, reasoning, meta) {
      const empty = $('chat-empty');
      if (empty) empty.remove();
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + role;

      if (reasoning) {
        const det = document.createElement('details');
        det.className = 'thinking';
        const sum = document.createElement('summary');
        sum.textContent = 'Reasoning Trace';
        const pre = document.createElement('pre');
        pre.textContent = reasoning;
        det.appendChild(sum);
        det.appendChild(pre);
        wrap.appendChild(det);
      }

      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      bubble.textContent = content || '(empty response)';
      wrap.appendChild(bubble);

      if (meta) {
        const m = document.createElement('div');
        m.className = 'msg-meta';
        m.textContent = meta;
        wrap.appendChild(m);
      }

      const sc = $('chat-scroll');
      sc.appendChild(wrap);
      sc.scrollTop = sc.scrollHeight;
    }

    async function sendChat(textOverride) {
      const input = $('chat-input');
      const text = (textOverride ?? input.value).trim();
      if (!text) return;
      if (!localConsole && !consoleToken) { accessDialog.showModal(); return; }
      if (!usingUpstream && $('opt-effort').value !== 'none') {
        $('chat-status').textContent = 'Set Reasoning to none for the native engine.';
        return;
      }
      if (!usingUpstream && Number($('opt-max-tokens').value) > 128) {
        $('chat-status').textContent = 'The native engine supports up to 128 output tokens.';
        return;
      }
      if (!textOverride) input.value = '';
      appendMessage('user', text, null, null);

      const btn = $('btn-send');
      btn.disabled = true;
      const t0 = performance.now();
      $('chat-status').textContent = 'Waiting for inference…';

      try {
        const res = await api('/v1/chat/completions', {
          method: 'POST',
          body: JSON.stringify({
            model: $('model-select').value,
            messages: [{ role: 'user', content: text }],
            max_tokens: Number($('opt-max-tokens').value),
            reasoning_effort: $('opt-effort').value,
          }),
        });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        if (res.ok && res.data.choices && res.data.choices[0]) {
          const msg = res.data.choices[0].message || {};
          const usage = res.data.usage;
          const meta = usage
            ? `${dt}s · prompt ${usage.prompt_tokens || 0} / comp ${usage.completion_tokens || 0} tok`
            : `${dt}s · ${res.data.model || 'Inkling-2-Mini'}`;
          appendMessage('assistant', msg.content || '(reasoning only — increase max_tokens)', msg.reasoning_content || null, meta);
          $('chat-status').textContent = `Done in ${dt}s`;
        } else {
          appendMessage('assistant', 'Error: ' + JSON.stringify(res.data), null, `HTTP ${res.status}`);
          $('chat-status').textContent = 'Request failed';
        }
      } catch (e) {
        appendMessage('assistant', 'Network error: ' + e.message, null, null);
        $('chat-status').textContent = 'Disconnected';
      } finally {
        btn.disabled = false;
        refreshHealthAndState();
      }
    }

    $('btn-send').addEventListener('click', () => sendChat());
    $('chat-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
      }
    });
    document.querySelectorAll('.chip').forEach((c) => {
      c.addEventListener('click', () => sendChat(c.dataset.prompt));
    });

    // Tensor Core Bench
    $('btn-refresh-gpu').addEventListener('click', refreshGpu);
    $('btn-run-tc').addEventListener('click', async () => {
      const btn = $('btn-run-tc');
      btn.disabled = true;
      $('tc-log').textContent = 'Running CUDA 13.5 sm_120a Tensor Core benchmark (BF16 / INT8 / NVFP4)…';
      try {
        const r = await api('/v1/gpu/bench-tc', { method: 'POST', body: '{}' });
        if (r.ok) {
          const bf16 = Number(r.data.bf16_tflops || 0);
          const int8 = Number(r.data.int8_tops || 0);
          const nvfp4 = Number(r.data.nvfp4_tflops || 0);
          const peak = Math.max(bf16, int8, nvfp4, 1);
          $('val-bf16').textContent = bf16 > 0 ? `${bf16.toFixed(2)} TFLOPS` : '—';
          $('val-int8').textContent = int8 > 0 ? `${int8.toFixed(2)} TOPS` : '—';
          $('val-nvfp4').textContent = nvfp4 > 0 ? `${nvfp4.toFixed(2)} TFLOPS` : '—';
          $('bar-bf16').style.width = `${Math.round((bf16 / peak) * 100)}%`;
          $('bar-int8').style.width = `${Math.round((int8 / peak) * 100)}%`;
          $('bar-nvfp4').style.width = `${Math.round((nvfp4 / peak) * 100)}%`;
          $('tc-log').textContent = r.data.log || r.data.result || JSON.stringify(r.data, null, 2);
        } else {
          $('tc-log').textContent = 'Benchmark error: ' + JSON.stringify(r.data);
        }
      } finally {
        btn.disabled = false;
        refreshGpu();
      }
    });

    // Memory45 & LoRA
    $('btn-refresh-mem').addEventListener('click', refreshHealthAndState);
    $('btn-mem-add').addEventListener('click', async () => {
      const text = $('mem-add-text').value.trim();
      if (!text) return;
      const importance = parseFloat($('mem-add-imp').value) || 0.85;
      const r = await api('/v1/memory/add', { method: 'POST', body: JSON.stringify({ text, importance }) });
      if (r.ok) {
        $('mem-add-text').value = '';
        refreshHealthAndState();
      }
    });
    $('btn-mem-recall').addEventListener('click', async () => {
      const query = $('mem-recall-q').value.trim();
      const r = await api('/v1/memory/recall', { method: 'POST', body: JSON.stringify({ query }) });
      const box = $('mem-hits');
      if (!r.ok || !Array.isArray(r.data.hits) || r.data.hits.length === 0) {
        box.innerHTML = '<div class="kv-label">No matching memories found.</div>';
        return;
      }
      box.innerHTML = r.data.hits.map((h) =>
        `<div class="list-item"><div><div>${escapeHtml(h.text)}</div><div style="font-size:11.5px;color:var(--text-tertiary);font-family:var(--font-mono);">id=${h.id} · imp=${h.importance} · str=${h.strength}</div></div></div>`
      ).join('');
    });
    $('btn-lora-step').addEventListener('click', async () => {
      const text = $('lora-text').value.trim() || 'positive reinforcement';
      const reward = parseFloat($('lora-r').value) || 0.5;
      await api('/v1/lora/reinforce', { method: 'POST', body: JSON.stringify({ text, reward }) });
      $('lora-text').value = '';
      refreshHealthAndState();
    });
    $('btn-lora-rollback').addEventListener('click', async () => {
      await api('/v1/lora/rollback', { method: 'POST', body: '{}' });
      refreshHealthAndState();
    });
    $('btn-voice-say').addEventListener('click', async () => {
      const text = $('voice-text').value.trim();
      if (!text) return;
      $('voice-status').textContent = 'Synthesizing…';
      const r = await api('/v1/voice/say', { method: 'POST', body: JSON.stringify({ text }) });
      $('voice-status').textContent = r.ok ? (r.data.detail || 'OK') : ('Error: ' + (r.data.error || r.status));
    });
    $('btn-refresh-models').addEventListener('click', refreshModels);

    // Hardened Finder FS
    let fsCurrentPath = '';
    let fsRequest = 0;
    let previewRequest = 0;
    async function loadFsDir(relPath = '') {
      const request = ++fsRequest;
      previewRequest++;
      $('fs-preview').textContent = 'Select a text file to preview.';
      $('fs-meta').textContent = '';
      fsCurrentPath = relPath;
      const root = $('fs-root').value;
      $('fs-breadcrumb').textContent = '/' + relPath;
      const r = await api('/v1/fs/list', { method: 'POST', body: JSON.stringify({ root, path: relPath }) });
      if (request !== fsRequest) return;
      const box = $('fs-list');
      if (!r.ok) {
        box.innerHTML = `<div class="kv-label" style="color:var(--danger);">${escapeHtml(r.data.error || 'Access denied')}</div>`;
        return;
      }
      if (!r.data.entries || r.data.entries.length === 0) {
        box.innerHTML = '<div class="kv-label">(empty directory)</div>';
        return;
      }
      box.innerHTML = '';
      r.data.entries.forEach((e) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.disabled = e.blocked;
        item.className = 'list-item btn-secondary';
        item.style.width = '100%';
        item.style.cursor = e.blocked ? 'not-allowed' : 'pointer';
        const icon = e.is_dir ? '📁' : (e.blocked ? '🔒' : '📄');
        const sz = e.is_dir ? 'DIR' : (e.blocked ? 'BLOCKED' : `${e.size} B`);
        item.innerHTML = `<span>${icon} ${escapeHtml(e.name)}</span><span class="kv-val">${sz}</span>`;
        if (!e.blocked) {
          item.addEventListener('click', () => {
            const next = fsCurrentPath ? `${fsCurrentPath}/${e.name}` : e.name;
            if (e.is_dir) loadFsDir(next);
            else readFsFile(next);
          });
        }
        box.appendChild(item);
      });
    }

    async function readFsFile(relPath) {
      const request = ++previewRequest;
      const root = $('fs-root').value;
      $('fs-preview').textContent = 'Reading safely…';
      const r = await api('/v1/fs/read', { method: 'POST', body: JSON.stringify({ root, path: relPath }) });
      if (request !== previewRequest) return;
      if (!r.ok) {
        $('fs-meta').textContent = 'BLOCKED';
        $('fs-preview').textContent = r.data.error || 'Read denied by policy';
        return;
      }
      $('fs-meta').textContent = `${r.data.size} B · FNV64:${r.data.fnv64}${r.data.truncated ? ' (64 KiB cap)' : ''}`;
      $('fs-preview').textContent = r.data.content;
    }

    $('fs-root').addEventListener('change', () => loadFsDir(''));
    $('btn-fs-refresh').addEventListener('click', () => loadFsDir(fsCurrentPath));
    $('btn-fs-up').addEventListener('click', () => {
      if (!fsCurrentPath) return;
      const parts = fsCurrentPath.split('/').filter(Boolean);
      parts.pop();
      loadFsDir(parts.join('/'));
    });

    // Hardened Browser
    let browserRequest = 0;
    const browserPage = $('browser-page');
    const browserBody = $('browser-body');
    const browserResults = $('browser-results');
    const linkPrefix = '#inkling-page-';
    const readerMarkup = '<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; form-action \'none\'; base-uri \'none\'"><style>body{font:16px/1.65 system-ui;color:#162234;margin:32px auto;padding:0 22px;max-width:820px;overflow-wrap:anywhere}a{color:#155fb8}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #d4dbe4}</style>';

    browserPage.addEventListener('load', () => {
      browserPage.contentDocument?.addEventListener('click', event => {
        const link = event.target.closest?.('a');
        if (!link) return;
        event.preventDefault();
        const href = link.getAttribute('href');
        if (href?.startsWith(linkPrefix)) fetchHardenedUrl(decodeURIComponent(href.slice(linkPrefix.length)));
      });
    });

    function resetBrowser(title, message) {
      browserPage.hidden = true;
      browserPage.removeAttribute('srcdoc');
      browserResults.hidden = true;
      browserResults.replaceChildren();
      browserBody.hidden = false;
      browserBody.textContent = message;
      $('browser-title').textContent = title;
      $('browser-meta').textContent = '';
    }

    async function fetchHardenedUrl(urlOverride) {
      const url = (urlOverride ?? $('browser-url').value).trim();
      if (!url) return;
      const request = ++browserRequest;
      $('browser-url').value = url;
      resetBrowser('Opening page…', 'Connecting through Inkling2.');
      const r = await api('/v1/net/fetch', { method: 'POST', body: JSON.stringify({ url }) });
      if (request !== browserRequest) return;
      if (!r.ok) {
        $('browser-title').textContent = 'Page unavailable';
        $('browser-meta').textContent = `HTTP ${r.status}`;
        browserBody.textContent = r.data.error || 'Could not open this page.';
        return;
      }
      $('browser-url').value = r.data.url;
      $('browser-title').textContent = r.data.title || url;
      $('browser-meta').textContent = `TLS ${r.data.tls} · ${r.data.bytes} B`;
      if (r.data.html) {
        browserPage.srcdoc = readerMarkup + r.data.html;
        browserBody.hidden = true;
        browserPage.hidden = false;
      } else {
        browserBody.textContent = r.data.text || 'This page has no readable text.';
      }
    }

    async function searchWeb() {
      const query = $('browser-url').value.trim();
      if (!query) return;
      const request = ++browserRequest;
      resetBrowser('Searching…', 'Waiting for RustXNG.');
      const r = await api('/v1/net/search', { method: 'POST', body: JSON.stringify({ query }) });
      if (request !== browserRequest) return;
      if (!r.ok) {
        resetBrowser('Search unavailable', r.data.error || 'Could not search the web.');
        return;
      }
      $('browser-title').textContent = 'Web search';
      $('browser-meta').textContent = 'RustXNG';
      for (const result of r.data.results) {
        const item = document.createElement('li');
        const button = document.createElement('button');
        const snippet = document.createElement('p');
        button.type = 'button';
        button.className = 'chip';
        button.textContent = result.title;
        button.title = result.url;
        button.addEventListener('click', () => fetchHardenedUrl(result.url));
        snippet.textContent = result.snippet || result.url;
        item.append(button, snippet);
        browserResults.append(item);
      }
      browserBody.hidden = r.data.results.length > 0;
      browserBody.textContent = 'No results with a public HTTPS address.';
      browserResults.hidden = r.data.results.length === 0;
    }

    $('btn-browser-go').addEventListener('click', () => fetchHardenedUrl());
    $('btn-browser-search').addEventListener('click', searchWeb);
    $('browser-url').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      if (/^https?:\/\//i.test(e.target.value.trim())) fetchHardenedUrl();
      else searchWeb();
    });
    document.querySelectorAll('.browser-quick').forEach((b) => {
      b.addEventListener('click', () => fetchHardenedUrl(b.dataset.url));
    });

    // Boot
    $('sb-endpoint').textContent = location.host;
    api('/v1/session').then(result => {
      if (!result.ok) return;
      usingUpstream = result.data.upstream;
      $('btn-lora-step').disabled = !result.data.experimental_lora;
    });
    refreshHealthAndState();
    refreshGpu();
    refreshModels();
    loadFsDir('');
    setInterval(refreshGpu, 5000);
    setInterval(refreshHealthAndState, 10000);
