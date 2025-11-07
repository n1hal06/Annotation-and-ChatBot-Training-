const API_BASE = 'http://127.0.0.1:5000';
const TOKEN = localStorage.getItem('nlu_token');
if (!TOKEN) window.location.href = 'auth.html';

async function load() {
  try {
    const resp = await fetch(API_BASE + '/api/workspaces', { headers: { 'Authorization': `Bearer ${TOKEN}` } });
    const data = await resp.json();
    const list = document.getElementById('list');
    list.innerHTML = '';
    (data.workspaces || []).forEach(ws => {
      const div = document.createElement('div');
      div.className = 'ws-item';
      div.innerHTML = `<div>${ws}</div><div><button data-ws="${ws}">Use</button></div>`;
      list.appendChild(div);
    });
    document.querySelectorAll('button[data-ws]').forEach(btn => {
      btn.addEventListener('click', (e) => {
          const id = e.target.getAttribute('data-ws');
          localStorage.setItem('nlu_workspace', id);
          // include skip_landing to bypass the landing redirect in index.html
          window.location.href = 'index.html?skip_landing=1';
      });
    });
  } catch (err) { alert('Failed to list workspaces: ' + err.message); }
}

document.getElementById('create-ws').addEventListener('click', async () => {
  const name = document.getElementById('ws-name').value || '';
  if (!name) return alert('Enter a workspace name');
  try {
    const resp = await fetch(API_BASE + '/api/workspaces', { method: 'POST', headers: {'Content-Type':'application/json', 'Authorization': `Bearer ${TOKEN}`}, body: JSON.stringify({name}) });
    const data = await resp.json();
    if (resp.ok) {
      load();
    } else {
      alert('Create failed: ' + JSON.stringify(data));
    }
  } catch (err) { alert('Network: ' + err.message); }
});

load();
