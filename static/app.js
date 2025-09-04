class Router {
  constructor(rootId = 'view') {
    this.root = document.getElementById(rootId);
    this.routes = [];
    this.handle = this.handle.bind(this);
    window.addEventListener('hashchange', this.handle);
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

  // Find the first matching route for the current hash and dispatch.
  handle() {
    const fragment = window.location.hash.slice(1) || '/';
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

// ---------------------------------------------------------------------------
// Video listing with pagination, sorting, and optional infinite scrolling
// ---------------------------------------------------------------------------

const state = {
  page: 1,
  perPage: 16,
  sort: { column: 'title', dir: 'asc' },
  total: 0,
  infinite: false,
  loading: false,
};

function formatDuration(sec) {
  if (!sec && sec !== 0) return '';
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  const m = Math.floor((sec / 60) % 60).toString().padStart(2, '0');
  const h = Math.floor(sec / 3600);
  return h ? `${h}:${m}:${s}` : `${m}:${s}`;
}

function formatSize(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let num = bytes;
  while (num >= 1024 && i < units.length - 1) {
    num /= 1024;
    i += 1;
  }
  return `${num.toFixed(1)} ${units[i]}`;
}

const columns = [
  { key: 'title', label: 'Title', get: v => v.name || v.title || '' },
  { key: 'duration', label: 'Duration', get: v => v.duration || 0, fmt: v => formatDuration(v.duration) },
  {
    key: 'resolution',
    label: 'Resolution',
    get: v => v.resolution || (v.width && v.height ? `${v.width}x${v.height}` : ''),
  },
  {
    key: 'added',
    label: 'Date added',
    get: v => v.added || v.video_mtime || v.mtime || 0,
    fmt: v => (v.added || v.video_mtime || v.mtime ? new Date((v.added || v.video_mtime || v.mtime) * 1000).toLocaleString() : ''),
  },
  { key: 'plays', label: 'Plays', get: v => v.plays || 0 },
  { key: 'codec', label: 'Codec', get: v => v.vcodec || v.codec || '' },
  { key: 'size', label: 'File size', get: v => v.size || 0, fmt: v => formatSize(v.size) },
];

async function listVideos(params = {}) {
  const url = new URL('/videos', window.location);
  const offset = params.offset || 0;
  const limit = params.limit || state.perPage;
  url.searchParams.set('offset', offset);
  url.searchParams.set('limit', limit);
  url.searchParams.set('detail', '1');

  const controller = new AbortController();
  const timeoutMs = 10000; // 10 seconds timeout
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(url.toString(), { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!resp.ok) throw new Error('Failed to load videos');
    return resp.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out while loading videos');
    } else if (err instanceof TypeError) {
      throw new Error('Network error while loading videos');
    } else {
      throw err;
    }
  }
}

function sortVideos(videos) {
  const { column, dir } = state.sort;
  const col = columns.find(c => c.key === column);
  if (!col) return videos;
  const factor = dir === 'asc' ? 1 : -1;
  return videos.sort((a, b) => {
    const va = col.get(a);
    const vb = col.get(b);
    if (typeof va === 'number' && typeof vb === 'number') {
      return (va - vb) * factor;
    }
    return String(va).localeCompare(String(vb)) * factor;
  });
}

function buildHeader(table) {
  if (table.tHead) return;
  const thead = table.createTHead();
  const row = thead.insertRow();
  columns.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.dataset.key = col.key;
    th.addEventListener('click', () => {
      if (state.sort.column === col.key) {
        state.sort.dir = state.sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        state.sort.column = col.key;
        state.sort.dir = 'asc';
      }
      renderList(true);
    });
    row.appendChild(th);
  });
}

function clearBody(table) {
  if (table.tBodies.length) {
    table.removeChild(table.tBodies[0]);
  }
}

function appendRows(table, videos) {
  const tbody = table.tBodies[0] || table.createTBody();
  videos.forEach(v => {
    const tr = tbody.insertRow();
    columns.forEach(col => {
      const td = tr.insertCell();
      const raw = col.get(v);
      td.textContent = col.fmt ? col.fmt(v) : raw;
    });
  });
}

async function renderList(reset = false) {
  if (state.loading) return;
  state.loading = true;
  if (reset) {
    state.page = 1;
    const table = document.getElementById('video-table');
    clearBody(table);
  }
  const offset = (state.page - 1) * state.perPage;
  const data = await listVideos({ offset, limit: state.perPage });
  state.total = data.count || 0;
  let vids = data.videos || [];
  vids = sortVideos(vids);
  const table = document.getElementById('video-table');
  buildHeader(table);
  appendRows(table, vids);
  updatePagination();
  state.loading = false;
}

function updatePagination() {
  const pag = document.getElementById('pagination');
  if (!pag) return;
  if (state.infinite) {
    pag.style.display = 'none';
    return;
  }
  pag.style.display = 'block';
  const pages = Math.max(1, Math.ceil(state.total / state.perPage));
  const info = document.getElementById('page-info');
  info.textContent = `Page ${state.page} of ${pages}`;
  const prev = document.getElementById('prev-page');
  const next = document.getElementById('next-page');
  prev.disabled = state.page <= 1;
  next.disabled = state.page >= pages;
}

function handleScroll() {
  if (!state.infinite || state.loading) return;
  if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50) {
    const maxPages = Math.ceil(state.total / state.perPage);
    if (state.page < maxPages) {
      state.page += 1;
      renderList(false);
    }
  }
}

window.addEventListener('load', () => {
  const prev = document.getElementById('prev-page');
  const next = document.getElementById('next-page');
  if (prev && next) {
    prev.addEventListener('click', () => {
      if (state.page > 1) {
        state.page -= 1;
        renderList(true);
      }
    });
    next.addEventListener('click', () => {
      const pages = Math.ceil(state.total / state.perPage);
      if (state.page < pages) {
        state.page += 1;
        renderList(true);
      }
    });
  }
  const toggle = document.getElementById('infiniteToggle');
  if (toggle) {
    toggle.addEventListener('change', () => {
      state.infinite = toggle.checked;
      window.removeEventListener('scroll', handleScroll);
      if (state.infinite) {
        window.addEventListener('scroll', handleScroll);
      }
      renderList(true);
    });
  }
  renderList(true);
});

window.renderList = renderList;
