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

  // Optional programmatic navigation helper
  navigate(path) {
    if (!path.startsWith('#')) {
      window.location.hash = path;
    } else {
      window.location.hash = path.slice(1);
    }
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

// ---------------------------------------------------------------------------
// Grid rendering helper
// ---------------------------------------------------------------------------
// Options: { containerId: 'content', limit: 50, sort: 'date_added desc', ...filters }
// Currently supports lazy loading batches from `/videos` endpoint. The API
// does not yet implement filtering or sort, but query parameters are always
// sent so future backend features can hook in without client changes.
async function renderGrid(options = {}) {
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
          const target = `/player/${encodeURIComponent(v.name)}`;
          if (window.router instanceof Router) {
            window.router.navigate(target);
          } else {
            window.location.hash = target;
          }
        };

        tile.addEventListener('click', navigate);
        tile.addEventListener('dblclick', navigate);
        tile.addEventListener('keydown', e => {
          if (e.key === 'Enter') navigate();
        });

        container.appendChild(tile);
      });
    } finally {
      loading = false;
    }
  }

  async function onScroll() {
    if (done || loading) return;
    // Trigger when within 200px of container bottom
    const { bottom } = container.getBoundingClientRect();
    if (bottom - window.innerHeight < 200) {
      await appendBatch();
    }
  }

  window.addEventListener('scroll', onScroll);
  await appendBatch();
  // In case initial content doesn't fill viewport, attempt another batch.
  await onScroll();
}

window.renderGrid = renderGrid;

// ---------------------------------------------------------------------------
// List rendering helper
// ---------------------------------------------------------------------------
// Options: { containerId: 'content', limit: 16, sort: 'title asc', ...filters }
// Fetches batches from `/videos` and renders a table. Supports pagination or
// optional infinite scrolling (controlled by a global `Settings` object).
async function renderList(options = {}) {
  const {
    containerId = 'content',
    limit = 16,
    sort = 'title asc',
    ...filters
  } = options;

  const container = document.getElementById(containerId);
  if (!container) return;

  // State
  let offset = 0;
  let total = 0;
  let loading = false;
  let [sortKey, sortDir] = sort.split(/\s+/);
  sortDir = sortDir || 'asc';

  // Respect global Settings toggle if present
  const settings = window.Settings || {};
  let useInfinite = settings.infiniteScroll === undefined ? false : !!settings.infiniteScroll;

  container.innerHTML = '';

  // Toggle for infinite scroll
  const toggleWrap = document.createElement('div');
  const toggle = document.createElement('input');
  toggle.type = 'checkbox';
  toggle.id = 'list-inf-toggle';
  toggle.checked = useInfinite;
  const toggleLabel = document.createElement('label');
  toggleLabel.htmlFor = 'list-inf-toggle';
  toggleLabel.textContent = 'Infinite scroll';
  toggleWrap.appendChild(toggle);
  toggleWrap.appendChild(toggleLabel);
  container.appendChild(toggleWrap);

  // Table skeleton
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const columns = [
    { key: 'title', label: 'Title' },
    { key: 'duration', label: 'Duration' },
    { key: 'size', label: 'Size' },
    { key: 'vcodec', label: 'Video' },
    { key: 'acodec', label: 'Audio' },
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
      offset = 0;
      tbody.innerHTML = '';
      fetchPage(true);
    });
    headRow.appendChild(th);
  });

  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  container.appendChild(table);

  // Pagination controls
  const pager = document.createElement('div');
  const prevBtn = document.createElement('button');
  prevBtn.textContent = 'Prev';
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next';
  pager.appendChild(prevBtn);
  pager.appendChild(nextBtn);
  container.appendChild(pager);

  prevBtn.addEventListener('click', () => {
    if (offset >= limit) {
      offset -= limit;
      fetchPage(true);
    }
  });
  nextBtn.addEventListener('click', () => {
    if (offset + limit < total) {
      offset += limit;
      fetchPage(true);
    }
  });

  function updatePager() {
    pager.style.display = useInfinite ? 'none' : 'block';
    prevBtn.disabled = offset <= 0;
    nextBtn.disabled = offset + limit >= total;
  }

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

  function getVal(v, key) {
    switch (key) {
      case 'title':
        return v.name?.toLowerCase();
      default:
        return v[key];
    }
  }

  function sortLocal(arr) {
    arr.sort((a, b) => {
      const va = getVal(a, sortKey);
      const vb = getVal(b, sortKey);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (va < vb) return sortDir === 'asc' ? -1 : 1;
      if (va > vb) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
  }

  async function fetchPage(reset = false) {
    if (loading) return;
    loading = true;
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
      sort: `${sortKey} ${sortDir}`,
      detail: 'true',
    });
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') params.append(k, v);
    });
    try {
      const resp = await fetch(`/videos?${params.toString()}`);
      if (!resp.ok) return;
      const data = await resp.json();
      total = data.count || data.total || 0;
      let vids = data.videos || [];
      sortLocal(vids);
      if (reset) tbody.innerHTML = '';
      vids.forEach(v => {
        const tr = document.createElement('tr');
        const tdTitle = document.createElement('td');
        const a = document.createElement('a');
        a.href = `#/player/${encodeURIComponent(v.name)}`;
        a.textContent = v.name;
        tdTitle.appendChild(a);
        tr.appendChild(tdTitle);
        const tdDur = document.createElement('td');
        tdDur.textContent = formatDuration(v.duration);
        tr.appendChild(tdDur);
        const tdSize = document.createElement('td');
        tdSize.textContent = formatSize(v.size);
        tr.appendChild(tdSize);
        const tdV = document.createElement('td');
        tdV.textContent = v.vcodec || '';
        tr.appendChild(tdV);
        const tdA = document.createElement('td');
        tdA.textContent = v.acodec || '';
        tr.appendChild(tdA);
        tbody.appendChild(tr);
      });
      if (useInfinite) {
        offset += vids.length;
      }
    } finally {
      loading = false;
      updatePager();
    }
  }

  function onScroll() {
    if (!useInfinite || loading || offset >= total) return;
    const { bottom } = table.getBoundingClientRect();
    if (bottom - window.innerHeight < 200) {
      fetchPage(false);
    }
  }

  function applyScroll() {
    window.removeEventListener('scroll', onScroll);
    if (useInfinite) window.addEventListener('scroll', onScroll);
  }

  toggle.addEventListener('change', () => {
    useInfinite = toggle.checked;
    settings.infiniteScroll = useInfinite;
    offset = 0;
    tbody.innerHTML = '';
    fetchPage(true);
    applyScroll();
    updatePager();
  });

  applyScroll();
  await fetchPage(true);
  if (useInfinite) onScroll();
}

window.renderList = renderList;

// ---------------------------------------------------------------------------
// Simple player renderer
// ---------------------------------------------------------------------------
async function renderPlayer(name, options = {}) {
  const { containerId = 'view' } = options;
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';
  let currentName = name;

  const titleInput = document.createElement('input');
  titleInput.type = 'text';
  titleInput.value = currentName;
  container.appendChild(titleInput);

  const video = document.createElement('video');
  video.controls = true;
  video.src = `/videos/${encodeURIComponent(currentName)}`;
  container.appendChild(video);

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
