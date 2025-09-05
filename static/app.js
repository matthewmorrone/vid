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
// Simple player renderer
// ---------------------------------------------------------------------------
function renderPlayer(name, options = {}) {
  const { containerId = 'view' } = options;
  const container = document.getElementById(containerId);
  if (!container) return;

  // Reset view and create basic player element
  container.innerHTML = '';
  const title = document.createElement('h2');
  title.textContent = name;
  container.appendChild(title);

  const video = document.createElement('video');
  video.controls = true;
  // Assume the server can serve raw video at this path
  video.src = `/videos/${encodeURIComponent(name)}`;
  container.appendChild(video);
}

window.renderPlayer = renderPlayer;
