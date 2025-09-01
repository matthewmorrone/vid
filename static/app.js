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
