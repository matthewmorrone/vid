class Router {
  constructor(rootId = 'view') {
    this.root = document.getElementById(rootId);
    this.routes = [];
    this.handle = this.handle.bind(this);
    // Use popstate for history navigation and initial load for deep links.
    window.addEventListener('popstate', this.handle);
    window.addEventListener('load', this.handle);
  }

  // Register a new route. `path` can be a regex or a ":param" pattern.
  register(path, handler) {
    let regex;
    let keys = [];
    if (path instanceof RegExp) {
      regex = path;
    } else {
      const pattern = path.replace(/:([^\/]+)/g, (_, key) => {
        keys.push(key);
        return '([^/]+)';
      });
      regex = new RegExp('^' + pattern + '$');
    }
    this.routes.push({ regex, keys, handler });
    return this;
  }

  // Optional programmatic navigation helper
  navigate(path) {
    // Use History API so URLs like /video/name are navigable and shareable.
    if (path !== window.location.pathname) {
      window.history.pushState({}, '', path);
      this.handle();
    }
  }

  // Find the first matching route for the current path and dispatch.
  handle() {
    const fragment = window.location.pathname || '/';
    for (const route of this.routes) {
      const match = fragment.match(route.regex);
      if (match) {
        const params = {};
        route.keys.forEach((k, i) => {
          params[k] = decodeURIComponent(match[i + 1]);
        });
        route.handler(params);
        return;
      }
    }
    // No route matched; clear the view.
    if (this.root) {
      this.root.textContent = '';
    }
  }
}

window.Router = Router;

// Simple toast helper
function showToast(message, { duration = 3000, actionText, onAction } = {}) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.position = 'fixed';
    container.style.bottom = '10px';
    container.style.right = '10px';
    container.style.zIndex = '1000';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.textContent = message;
  toast.style.background = 'rgba(0,0,0,0.7)';
  toast.style.color = 'white';
  toast.style.padding = '8px 12px';
  toast.style.marginTop = '5px';
  toast.style.borderRadius = '4px';
  if (actionText && typeof onAction === 'function') {
    const btn = document.createElement('button');
    btn.textContent = actionText;
    btn.style.marginLeft = '8px';
    btn.addEventListener('click', () => {
      onAction();
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    });
    toast.appendChild(btn);
  }
  container.appendChild(toast);
  if (duration > 0) {
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, duration);
  }
  return toast;
}

function detachPlayerHotkeys() {
  if (window.__playerKeyHandler) {
    window.removeEventListener('keydown', window.__playerKeyHandler);
    window.__playerKeyHandler = null;
  }
}

function openSidebar(video) {
  let sidebar = document.getElementById('sidebar');
  if (!sidebar) {
    sidebar = document.createElement('div');
    sidebar.id = 'sidebar';
    sidebar.style.position = 'fixed';
    sidebar.style.top = '0';
    sidebar.style.right = '0';
    sidebar.style.width = '300px';
    sidebar.style.height = '100%';
    sidebar.style.background = 'rgba(0,0,0,0.8)';
    sidebar.style.color = 'white';
    sidebar.style.overflow = 'auto';
    sidebar.style.padding = '10px';
    document.body.appendChild(sidebar);
  }
  sidebar.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <h3 style="margin: 0;">${video.name}</h3>
      <button id="sidebar-close-btn" style="background: transparent; color: white; border: none; font-size: 1.5em; cursor: pointer;">&times;</button>
    </div>
  `;
  const closeBtn = sidebar.querySelector('#sidebar-close-btn');
  closeBtn.addEventListener('click', () => {
    sidebar.remove();
  });
}

window.openSidebar = openSidebar;

// ---------------------------------------------------------------------------
// Settings handling
// ---------------------------------------------------------------------------
const DEFAULT_SETTINGS = {
  theme: 'system',
  paging: 'grid',
  listPageSize: 16,
  autoAdvance: true,
  infiniteScroll: false,
};

function loadSettings() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem('settings') || '{}');
  } catch (_) {
    stored = {};
  }
  window.Settings = Object.assign({}, DEFAULT_SETTINGS, stored);
}

function saveSettings() {
  try {
    localStorage.setItem('settings', JSON.stringify(window.Settings));
  } catch (_) {
    // ignore
  }
}

function applyTheme(theme) {
  const root = document.documentElement;
  root.classList.remove('theme-dark', 'theme-light');
  let mode = theme;
  if (mode === 'system') {
    mode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  if (mode === 'dark') {
    root.classList.add('theme-dark');
  } else {
    root.classList.add('theme-light');
  }
}

loadSettings();
applyTheme(window.Settings.theme);

if (window.matchMedia) {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const handle = () => {
    if (window.Settings.theme === 'system') applyTheme('system');
  };
  if (mq.addEventListener) mq.addEventListener('change', handle); else if (mq.addListener) mq.addListener(handle);
}

function renderSettings(options = {}) {
  detachPlayerHotkeys();
  const { containerId = 'view' } = options;
  const container = document.getElementById(containerId);
  if (!container) return;
  const settings = window.Settings;

  container.innerHTML = '';

  function addSection(labelText, input) {
    const label = document.createElement('label');
    label.textContent = labelText + ' ';
    label.appendChild(input);
    const wrap = document.createElement('div');
    wrap.appendChild(label);
    container.appendChild(wrap);
  }

  const themeSel = document.createElement('select');
  [
    { value: 'system', text: 'System' },
    { value: 'light', text: 'Light' },
    { value: 'dark', text: 'Dark' },
  ].forEach(opt => {
    const o = document.createElement('option');
    o.value = opt.value;
    o.textContent = opt.text;
    themeSel.appendChild(o);
  });
  themeSel.value = settings.theme;
  themeSel.addEventListener('change', () => {
    settings.theme = themeSel.value;
    saveSettings();
    applyTheme(settings.theme);
  });
  addSection('Theme:', themeSel);

  const pagingSel = document.createElement('select');
  [
    { value: 'grid', text: 'Grid' },
    { value: 'list', text: 'List' },
  ].forEach(opt => {
    const o = document.createElement('option');
    o.value = opt.value;
    o.textContent = opt.text;
    pagingSel.appendChild(o);
  });
  pagingSel.value = settings.paging;
  pagingSel.addEventListener('change', () => {
    settings.paging = pagingSel.value;
    saveSettings();
  });
  addSection('Paging mode:', pagingSel);

  const sizeInput = document.createElement('input');
  sizeInput.type = 'number';
  sizeInput.min = '1';
  sizeInput.value = settings.listPageSize;
  let prevValidPageSize = settings.listPageSize;

  sizeInput.addEventListener('change', () => {
    const v = parseInt(sizeInput.value, 10);
    if (!isNaN(v) && v > 0) {
      settings.listPageSize = v;
      prevValidPageSize = v;
      saveSettings();
    } else {
      alert('Please enter a valid page size (number greater than 0).');
      sizeInput.value = prevValidPageSize;
    }
    }
  });
  addSection('List page size:', sizeInput);

  const autoChk = document.createElement('input');
  autoChk.type = 'checkbox';
  autoChk.checked = !!settings.autoAdvance;
  autoChk.addEventListener('change', () => {
    settings.autoAdvance = autoChk.checked;
    saveSettings();
  });
  addSection('Auto-advance:', autoChk);

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset to defaults';
  resetBtn.addEventListener('click', () => {
    Object.assign(settings, DEFAULT_SETTINGS);
    saveSettings();
    applyTheme(settings.theme);
    renderSettings(options);
  });
  container.appendChild(resetBtn);
}

window.renderSettings = renderSettings;

// ---------------------------------------------------------------------------
// Grid rendering helper
// ---------------------------------------------------------------------------
// Options: { containerId: 'content', limit: 50, sort: 'date_added desc', ...filters }
// Currently supports lazy loading batches from `/videos` endpoint. The API
// does not yet implement filtering or sort, but query parameters are always
// sent so future backend features can hook in without client changes.
async function renderGrid(options = {}) {
  detachPlayerHotkeys();
  const {
    containerId = 'content',
    limit = 50,
    sort = 'date_added desc',
    ...filters
  } = options;

  const container = document.getElementById(containerId);
  if (!container) return;

  let offset = 0;
  let loading = false;
  let done = false;

  const sentinel = document.createElement('div');
  container.appendChild(sentinel);
  let observer;

  // Create play overlay element for a tile
  function createOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'play-overlay';
    overlay.textContent = '▶';
    // Basic inline styles so the icon sits centered and hidden by default.
    overlay.style.position = 'absolute';
    overlay.style.top = '50%';
    overlay.style.left = '50%';
    overlay.style.transform = 'translate(-50%, -50%)';
    overlay.style.fontSize = '2rem';
    overlay.style.color = 'white';
    overlay.style.pointerEvents = 'none';
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.2s';
    return overlay;
  }

  // Append a batch of videos to the grid
  async function appendBatch() {
    if (loading || done) return;
    loading = true;

    const params = new URLSearchParams({ offset, limit, sort });
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') params.append(k, v);
    });

    try {
      const resp = await fetch(`/videos?${params.toString()}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const videos = data.videos || [];
      offset += videos.length;
      if (videos.length < limit) done = true;

      videos.forEach(v => {
        const tile = document.createElement('div');
        tile.className = 'grid-item';
        tile.tabIndex = 0; // allow keyboard focus
        tile.style.position = 'relative';

        // Base thumbnail
        const img = document.createElement('img');
        if (v.artifacts && v.artifacts.thumbs && v.artifacts.thumbs.exists) {
          img.src = v.artifacts.thumbs.url;
        }
        tile.appendChild(img);

        // Hover preview (if preview clip exists)
        let vid = null;
        if (v.artifacts && v.artifacts.previews && v.artifacts.previews.exists) {
          vid = document.createElement('video');
          vid.src = v.artifacts.previews.url;
          vid.muted = true;
          vid.loop = true;
          vid.style.display = 'none';
          tile.appendChild(vid);
        }

        // Play overlay
        const overlay = createOverlay();
        tile.appendChild(overlay);

        // Hover handlers for preview/overlay
        const handleEnter = () => {
          overlay.style.opacity = '1';
          if (vid) {
            img.style.display = 'none';
            vid.style.display = 'block';
            vid.play().catch(() => {});
          }
        };
        const handleLeave = () => {
          overlay.style.opacity = '0';
          if (vid) {
            vid.pause();
            vid.style.display = 'none';
            img.style.display = 'block';
          }
        };
        tile.addEventListener('mouseenter', handleEnter);
        tile.addEventListener('mouseleave', handleLeave);
        tile.addEventListener('focus', handleEnter);
        tile.addEventListener('blur', handleLeave);

        const navigate = () => {
          const target = `/video/${encodeURIComponent(v.name)}`;
          if (window.router instanceof Router) {
            window.router.navigate(target);
          } else {
            window.location.pathname = target;
          }
        };

        tile.addEventListener('click', () => openSidebar(v));
        tile.addEventListener('dblclick', navigate);
        tile.addEventListener('keydown', e => {
          if (e.key === 'Enter') navigate();
        });

        container.insertBefore(tile, sentinel);
      });
      if (videos.length < limit) {
        done = true;
        if (observer) observer.disconnect();
        sentinel.remove();
      }
    } finally {
      loading = false;
    }
  }

  observer = new IntersectionObserver(async entries => {
    if (entries.some(e => e.isIntersecting)) {
      await appendBatch();
    }
  }, { root: null, rootMargin: '200px' });
  observer.observe(sentinel);

  await appendBatch();
  // Instead of a synchronous while loop, use a non-blocking batch loader.
  async function loadBatches() {
    if (!done && sentinel.getBoundingClientRect().top <= window.innerHeight + 200) {
      await appendBatch();
      setTimeout(loadBatches, 0); // Yield control to the browser between batches
    }
  }
  loadBatches();
}

window.renderGrid = renderGrid;

// ---------------------------------------------------------------------------
// List rendering helper
// ---------------------------------------------------------------------------
// Fetches `/videos` with limit and page parameters and renders a sortable,
// paginated table.
async function renderList(options = {}) {
  detachPlayerHotkeys();
  const settings = window.Settings || {};
  const {
    containerId = 'content',
    limit = settings.listPageSize || 16,
    sort = 'title asc',
    page = 1,
    ...filters
  } = options;

  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';

  let currentPage = page;
  let [sortKey, sortDir] = sort.split(/\s+/);
  sortDir = sortDir || 'asc';

  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const columns = [
    { key: 'title', label: 'Title' },
    { key: 'duration', label: 'Duration' },
    { key: 'resolution', label: 'Resolution' },
    { key: 'date_added', label: 'Date Added' },
    { key: 'plays', label: 'Plays' },
    { key: 'codec', label: 'Codec' },
    { key: 'size', label: 'File Size' },
  ];
  columns.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      if (sortKey === col.key) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = col.key;
        sortDir = 'asc';
      }
      currentPage = 1;
      load();
    });
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  container.appendChild(table);

  const pager = document.createElement('div');
  pager.className = 'pager';
  container.appendChild(pager);

  function formatSize(bytes) {
    if (bytes == null) return '';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    let n = bytes;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatDuration(sec) {
    if (sec == null) return '';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const parts = [m, s].map(v => String(v).padStart(2, '0'));
    if (h > 0) parts.unshift(String(h));
    return parts.join(':');
  }

  const formatDate = str => {
    if (!str) return '';
    const d = new Date(str);
    return isNaN(d.getTime()) ? '' : d.toLocaleString();
  };

  const getResolution = v => {
    if (v.width != null && v.height != null) return `${v.width}x${v.height}`;
    return v.resolution || '';
  };

  const getCodec = v => v.codec || v.vcodec || '';

  function renderPager(total) {
    const totalPages = Math.max(1, Math.ceil(total / limit));
    pager.innerHTML = '';
    const prev = document.createElement('button');
    prev.textContent = 'Prev';
    prev.disabled = currentPage <= 1;
    prev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage -= 1;
        load();
      }
    });
    const next = document.createElement('button');
    next.textContent = 'Next';
    next.disabled = currentPage >= totalPages;
    next.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage += 1;
        load();
      }
    });
    const info = document.createElement('span');
    info.textContent = ` Page ${currentPage} of ${totalPages} `;
    pager.appendChild(prev);
    pager.appendChild(info);
    pager.appendChild(next);
  }

  async function load() {
    const params = new URLSearchParams({
      limit: String(limit),
      page: String(currentPage),
      sort: `${sortKey} ${sortDir}`,
      offset: String((currentPage - 1) * limit),
    });
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') params.append(k, v);
    });
    const resp = await fetch(`/videos?${params.toString()}`);
    if (!resp.ok) return;
    const data = await resp.json();
    const videos = data.videos || [];
    const total = data.count || data.total || 0;
    tbody.innerHTML = '';
    videos.forEach(v => {
      const tr = document.createElement('tr');
      const titleTd = document.createElement('td');
      const link = document.createElement('a');
      const name = v.name || v.title;
      link.href = `/video/${encodeURIComponent(name)}`;
      link.textContent = name;
      link.addEventListener('click', e => {
        e.preventDefault();
        if (window.router instanceof Router) {
          window.router.navigate(link.getAttribute('href'));
        } else {
          window.location.pathname = link.getAttribute('href');
        }
      });
      titleTd.appendChild(link);
      tr.appendChild(titleTd);

      const durTd = document.createElement('td');
      durTd.textContent = formatDuration(v.duration);
      tr.appendChild(durTd);

      const resTd = document.createElement('td');
      resTd.textContent = getResolution(v);
      tr.appendChild(resTd);

      const dateTd = document.createElement('td');
      dateTd.textContent = formatDate(v.date_added || v.created);
      tr.appendChild(dateTd);

      const playsTd = document.createElement('td');
      playsTd.textContent = v.plays != null ? v.plays : '';
      tr.appendChild(playsTd);

      const codecTd = document.createElement('td');
      codecTd.textContent = getCodec(v);
      tr.appendChild(codecTd);

      const sizeTd = document.createElement('td');
      sizeTd.textContent = formatSize(v.size);
      tr.appendChild(sizeTd);

      tbody.appendChild(tr);
    });
    renderPager(total);
  }

  await load();
}

window.renderList = renderList;
// ---------------------------------------------------------------------------
// Simple player renderer
// ---------------------------------------------------------------------------
async function renderPlayer(name, options = {}) {
  const { containerId = 'view', autoplay = false } = options;
  const container = document.getElementById(containerId);
  if (!container) return;

  detachPlayerHotkeys();

  container.innerHTML = '';
  let currentName = name;
  const settings = window.Settings || {};

  // Fetch detail/metadata for subtitles and scenes
  let detail = {};
  try {
    const r = await fetch(`/videos/${encodeURIComponent(currentName)}`);
    if (r.ok) detail = await r.json();
  } catch (_) {
    // ignore
  }
  const titleInput = document.createElement('input');
  titleInput.type = 'text';
  titleInput.value = currentName;
  container.appendChild(titleInput);

  const video = document.createElement('video');
  video.controls = true;
  video.src = `/videos/${encodeURIComponent(currentName)}`;
  video.setAttribute('playsinline', '');
  video.autoplay = autoplay;
  container.appendChild(video);

  // Subtitles
  if (detail.artifacts && detail.artifacts.subtitles && detail.artifacts.subtitles.exists) {
    const track = document.createElement('track');
    track.kind = 'subtitles';
    track.src = detail.artifacts.subtitles.url;
    track.default = true;
    video.appendChild(track);
  }

  // Scene tick marks container
  const tickBar = document.createElement('div');
  tickBar.style.position = 'relative';
  tickBar.style.height = '4px';
  tickBar.style.background = '#444';
  tickBar.style.marginTop = '4px';
  container.appendChild(tickBar);

  let sceneMarkers = [];
  try {
    if (detail.artifacts && detail.artifacts.scenes && detail.artifacts.scenes.exists) {
      const sc = await fetch(detail.artifacts.scenes.url);
      if (sc.ok) {
        const scj = await sc.json();
        sceneMarkers = scj.markers || [];
      }
    }
  } catch (_) {
    // ignore
  }

  function renderTicks() {
    if (!sceneMarkers.length || !video.duration) return;
    tickBar.innerHTML = '';
    sceneMarkers.forEach(m => {
      const tk = document.createElement('div');
      tk.style.position = 'absolute';
      tk.style.left = `${(m.time / video.duration) * 100}%`;
      tk.style.width = '2px';
      tk.style.height = '100%';
      tk.style.background = '#fff';
      tickBar.appendChild(tk);
    });
  }

  if (sceneMarkers.length) {
    if (video.readyState >= 1) {
      renderTicks();
    } else {
      video.addEventListener('loadedmetadata', renderTicks);
    }
  }

  if (autoplay) {
    video.addEventListener('canplay', () => {
      video.play().catch(() => {});
    }, { once: true });
  }

  const keyHandler = e => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    switch (e.key) {
      case ' ':
      case 'k':
        e.preventDefault();
        if (video.paused) video.play(); else video.pause();
        break;
      case 'j':
        video.currentTime = Math.max(0, video.currentTime - 10);
        break;
      case 'l':
        video.currentTime = Math.min(video.duration || Infinity, video.currentTime + 10);
        break;
      case 'ArrowLeft':
        video.currentTime = Math.max(0, video.currentTime - 5);
        break;
      case 'ArrowRight':
        video.currentTime = Math.min(video.duration || Infinity, video.currentTime + 5);
        break;
      case 'ArrowUp':
        video.volume = Math.min(1, video.volume + 0.1);
        break;
      case 'ArrowDown':
        video.volume = Math.max(0, video.volume - 0.1);
        break;
    }
  };
  window.__playerKeyHandler = keyHandler;
  window.addEventListener('keydown', keyHandler);

  // Auto-advance: fetch only the next video name when needed
  if (settings.autoAdvance !== false) {
    video.addEventListener('ended', async () => {
      try {
        const resp = await fetch(`/videos/next?current=${encodeURIComponent(currentName)}`);
        if (resp.ok) {
          const data = await resp.json();
          const {next} = data;
          if (next) {
            if (window.router instanceof Router) {
              window.router.navigate(`/video/${encodeURIComponent(next)}`);
            } else {
              renderPlayer(next, { autoplay: true });
            }
          }
        }
      } catch (_) {
        // ignore
      }
    });
  }

  let tagData = { tags: [], performers: [], description: '' };
  try {
    const resp = await fetch(`/videos/${encodeURIComponent(currentName)}/tags`);
    if (resp.ok) {
      const d = await resp.json();
      tagData = {
        tags: d.tags || [],
        performers: d.performers || [],
        description: d.description || '',
      };
    }
  } catch (e) {
    // ignore
  }

  const descInput = document.createElement('textarea');
  descInput.value = tagData.description || '';
  container.appendChild(descInput);

  const tagsInput = document.createElement('input');
  tagsInput.type = 'text';
  tagsInput.value = (tagData.tags || []).join(', ');
  container.appendChild(tagsInput);

  const perfInput = document.createElement('input');
  perfInput.type = 'text';
  perfInput.value = (tagData.performers || []).join(', ');
  container.appendChild(perfInput);

  function splitList(str) {
    return str.split(',').map(s => s.trim()).filter(Boolean);
  }

  async function patchTags(payload, field) {
    try {
      const resp = await fetch(`/videos/${encodeURIComponent(currentName)}/tags`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error('bad');
      field.dataset.retry = '';
      return true;
    } catch (err) {
      field.dataset.retry = '1';
      showToast('Update failed');
      return false;
    }
  }

  descInput.addEventListener('blur', async () => {
    const val = descInput.value;
    if (descInput.dataset.retry !== '1' && val === tagData.description) return;
    const ok = await patchTags({ description: val }, descInput);
    if (ok) tagData.description = val;
  });

  tagsInput.addEventListener('blur', async () => {
    const newTags = splitList(tagsInput.value);
    if (tagsInput.dataset.retry !== '1' && newTags.join(',') === (tagData.tags || []).join(',')) return;
    const ok = await patchTags({ replace: true, add: newTags }, tagsInput);
    if (ok) tagData.tags = newTags;
  });

  perfInput.addEventListener('blur', async () => {
    const newPerfs = splitList(perfInput.value);
    if (perfInput.dataset.retry !== '1' && newPerfs.join(',') === (tagData.performers || []).join(',')) return;
    const payload = { performers_remove: tagData.performers, performers_add: newPerfs };
    const ok = await patchTags(payload, perfInput);
    if (ok) tagData.performers = newPerfs;
  });

  async function renameVideo(oldName, newName, field) {
    try {
      const resp = await fetch(`/videos/${encodeURIComponent(oldName)}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: newName }),
      });
      if (!resp.ok) throw new Error('Failed to rename video');
      currentName = newName;
      video.src = `/videos/${encodeURIComponent(newName)}`;
      field.value = newName;
      field.dataset.retry = '';
      showToast(`Renamed to ${newName}`, {
        duration: 10000,
        actionText: 'Undo',
        onAction: () => renameVideo(newName, oldName, field),
      });
      return true;
    } catch (err) {
      field.dataset.retry = '1';
      showToast('Rename failed');
      return false;
    }
  }

  titleInput.addEventListener('blur', () => {
    const newName = titleInput.value.trim();
    if (!newName || (titleInput.dataset.retry !== '1' && newName === currentName)) return;
    if (newName !== currentName) {
      const confirmed = window.confirm('Are you sure you want to rename the video?');
      if (confirmed) {
        renameVideo(currentName, newName, titleInput);
      }
    }
  });
}

window.renderPlayer = renderPlayer;

// ---------------------------------------------------------------------------
// Artifact coverage dashboard & job management
// ---------------------------------------------------------------------------

const MAX_CONCURRENT_JOBS = 4;
const _activeJobs = new Map();

function _artifactTask(key) {
  return key === 'faces' ? 'embed' : key;
}

async function renderReport(opts = {}) {
  detachPlayerHotkeys();
  const { containerId = 'view', directory = '.', recursive = false } = opts;
  const container = document.getElementById(containerId);
  if (!container) return;
  container.textContent = 'Loading...';
  try {
    const resp = await fetch(`/report?directory=${encodeURIComponent(directory)}&recursive=${recursive ? 1 : 0}`);
    if (!resp.ok) throw new Error('failed');
    const data = await resp.json();
    container.textContent = '';
    const grid = document.createElement('div');
    grid.style.display = 'flex';
    grid.style.flexWrap = 'wrap';
    grid.style.gap = '10px';
    const total = data.total || 0;
    const counts = data.counts || {};
    const coverage = data.coverage || {};
    Object.keys(counts).forEach(key => {
      const card = document.createElement('div');
      card.style.border = '1px solid #ccc';
      card.style.padding = '8px';
      card.style.width = '180px';
      const h = document.createElement('h3');
      h.textContent = key;
      h.style.margin = '0 0 4px 0';
      const pct = ((coverage[key] || 0) * 100).toFixed(1);
      const p = document.createElement('p');
      p.textContent = `${counts[key]}/${total} (${pct}%)`;
      p.style.margin = '0 0 6px 0';
      card.appendChild(h);
      card.appendChild(p);
      const btn = document.createElement('button');
      btn.textContent = 'Generate Missing';
      if (coverage[key] >= 1) btn.disabled = true;
      btn.addEventListener('click', () => _startJobForArtifact(key, card, btn));
      card.appendChild(btn);
      grid.appendChild(card);
    });
    container.appendChild(grid);
  } catch (err) {
    container.textContent = 'Failed to load';
  }
}

async function _startJobForArtifact(key, card, button) {
  if (_activeJobs.size >= MAX_CONCURRENT_JOBS) {
    showToast('Maximum 4 concurrent jobs');
    return;
  }
  const task = _artifactTask(key);
  button.disabled = true;
  try {
    const resp = await fetch('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, directory: '.', recursive: false, force: false })
    });
    if (!resp.ok) throw new Error('submit failed');
    const job = await resp.json();
    _trackJob(job, card, button);
  } catch (err) {
    button.disabled = false;
    showToast('Job submission failed');
  }
}

function _trackJob(job, card, button) {
  const progress = document.createElement('progress');
  progress.max = job.progress_total || 1;
  progress.value = job.progress_current || 0;
  progress.style.display = 'block';
  progress.style.width = '100%';
  card.appendChild(progress);

  const status = document.createElement('span');
  status.textContent = job.status;
  status.className = 'job-status';
  card.appendChild(status);

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.marginLeft = '8px';
  cancelBtn.addEventListener('click', () => _cancelJob(job.id, cancelBtn));
  card.appendChild(cancelBtn);

  let pollTimer = null;
  _activeJobs.set(job.id, { progress, status, cancelBtn, button });

  const es = new EventSource(`/jobs/${job.id}/events`);

  function update(d) {
    progress.max = d.progress_total || 1;
    progress.value = d.progress_current || 0;
    status.textContent = d.status;
    if (['done', 'error', 'canceled'].includes(d.status)) {
      cleanup();
      renderReport();
    }
  }

  function cleanup() {
    if (pollTimer) clearInterval(pollTimer);
    es.onerror = null;
    try { es.close(); } catch (e) {}
    cancelBtn.remove();
    progress.remove();
    status.remove();
    button.disabled = false;
    _activeJobs.delete(job.id);
  }

  es.addEventListener('progress', ev => {
    const d = JSON.parse(ev.data);
    update(d);
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    es.close();
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch(`/jobs/${job.id}`);
        if (!resp.ok) throw new Error('gone');
        const d = await resp.json();
        update(d);
        if (['done', 'error', 'canceled'].includes(d.status)) {
          clearInterval(pollTimer);
        }
      } catch (err) {
        clearInterval(pollTimer);
      }
    }, 1000);
  };
}

async function _cancelJob(id, btn) {
  btn.disabled = true;
  try {
    await fetch(`/jobs/${id}`, { method: 'DELETE' });
  } catch (e) {
    // ignore
  }
}

window.renderReport = renderReport;
