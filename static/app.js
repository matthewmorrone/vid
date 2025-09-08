import { listVideos, getVideo, startJob } from '/static/api.js';

const settings = {
  theme: localStorage.getItem('theme') || 'dark'
};

function applyTheme(){
  document.body.classList.remove('theme-dark','theme-light');
  document.body.classList.add(settings.theme === 'light' ? 'theme-light' : 'theme-dark');
}

applyTheme();

class Router {
  constructor(routes){
    this.routes = routes;
    window.addEventListener('popstate', () => this.load(location.pathname));
    document.body.addEventListener('click', e => {
      const a = e.target.closest('a[data-link]');
      if(a){
        e.preventDefault();
        this.go(a.getAttribute('href'));
      }
    });
  }
  go(path){
    history.pushState(null, '', path);
    this.load(path);
  }
  load(path){
    const view = this.routes[path] || this.routes['/grid'];
    view();
  }
}

function renderGrid(){
  const main = document.getElementById('content');
  main.innerHTML = '<div class="grid" id="grid"></div>';
  listVideos().then(data => {
    const grid = document.getElementById('grid');
    data.items.forEach(v => {
      const card = document.createElement('div');
      card.className='card';
      card.textContent = v.title || v.name;
      card.addEventListener('click', () => openSidebar(v.name));
      grid.appendChild(card);
    });
  }).catch(() => {
    main.textContent = 'Failed to load videos';
  });
}

function openSidebar(name){
  const aside = document.getElementById('sidebar');
  aside.textContent = 'Loading...';
  getVideo(name).then(v => {
    aside.innerHTML = `\n      <h2>${v.title || v.name}</h2>\n      <p>Duration: ${v.duration || '?'}s</p>\n    `;
  }).catch(()=>{
    aside.textContent = 'Failed to load';
  });
}

function renderTasks(){
  const main = document.getElementById('content');
  main.innerHTML = '<h2>Tasks</h2><div id="task-buttons"></div>';
  const tasks = [
    {label:'Generate Thumbnails',type:'thumbnails'},
    {label:'Generate Previews',type:'previews'}
  ];
  const div = document.getElementById('task-buttons');
  tasks.forEach(t => {
    const btn = document.createElement('button');
    btn.textContent = t.label;
    btn.addEventListener('click', () => {
      btn.disabled = true;
      startJob(t.type).then(()=>{
        btn.disabled = false;
      }).catch(()=>{
        btn.disabled = false;
        alert('Failed to start job');
      });
    });
    div.appendChild(btn);
  });
}

function renderSettings(){
  const main = document.getElementById('content');
  main.innerHTML = `\n    <h2>Settings</h2>\n    <label>Theme: <select id="theme-select">\n      <option value="dark">Dark</option>\n      <option value="light">Light</option>\n    </select></label>\n  `;
  const sel = document.getElementById('theme-select');
  sel.value = settings.theme;
  sel.addEventListener('change', () => {
    settings.theme = sel.value;
    localStorage.setItem('theme', settings.theme);
    applyTheme();
  });
}

const router = new Router({
  '/': renderGrid,
  '/grid': renderGrid,
  '/list': () => document.getElementById('content').textContent = 'List view (todo)',
  '/player': () => document.getElementById('content').textContent = 'Player view (todo)',
  '/tasks': renderTasks,
  '/settings': renderSettings,
  '/stats': () => document.getElementById('content').textContent = 'Stats view (todo)',
  '/random': () => document.getElementById('content').textContent = 'Random view (todo)'
});

router.load(location.pathname);
