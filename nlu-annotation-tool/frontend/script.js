(function () {
  const apiBase = 'http://127.0.0.1:5000';

  const textInput = document.getElementById('text-input');
  const intentInput = document.getElementById('intent-input');
  const entStart = document.getElementById('ent-start');
  const entEnd = document.getElementById('ent-end');
  const entLabel = document.getElementById('ent-label');
  const entitiesList = document.getElementById('entities-list');
  const annotationsPreview = document.getElementById('annotations-preview');
  const modelsMeta = document.getElementById('models-meta');

  let entities = [];

  function renderEntities() {
    entitiesList.textContent = JSON.stringify(entities, null, 2);
  }

  function renderAnnotationsPreview() {
    const preview = {
      text: textInput.value || '',
      intent: intentInput.value || '',
      entities: entities
    };
    annotationsPreview.textContent = JSON.stringify(preview, null, 2);
  }

  document.getElementById('add-entity').addEventListener('click', () => {
    const s = parseInt(entStart.value);
    const e = parseInt(entEnd.value);
    const lab = entLabel.value && entLabel.value.trim();
    if (Number.isNaN(s) || Number.isNaN(e) || !lab) {
      alert('Please provide valid start, end and label for entity.');
      return;
    }
    entities.push({ start: s, end: e, label: lab });
    renderEntities();
    renderAnnotationsPreview();
  });

  document.getElementById('save-annotation').addEventListener('click', async () => {
    const payload = {
      text: textInput.value || '',
      intent: intentInput.value || '',
      entities: entities
    };
    try {
      const resp = await fetch(apiBase + '/save_annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (resp.ok) {
        alert('Annotation saved');
        // clear
        entities = [];
        renderEntities();
        renderAnnotationsPreview();
      } else {
        alert('Error saving annotation: ' + (data.error || data.details || JSON.stringify(data)));
      }
    } catch (err) {
      alert('Network error: ' + err.message);
    }
  });

  document.getElementById('train-spacy').addEventListener('click', async () => {
    try {
      const resp = await fetch(apiBase + '/train_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: 'spacy' })
      });
      const data = await resp.json();
      if (resp.ok) {
        alert('spaCy training finished: ' + JSON.stringify(data));
      } else {
        alert('spaCy training failed: ' + (data.error || data.details || JSON.stringify(data)));
      }
    } catch (err) {
      alert('Network error: ' + err.message);
    }
  });

  document.getElementById('train-rasa').addEventListener('click', async () => {
    try {
      const resp = await fetch(apiBase + '/train_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: 'rasa' })
      });
      const data = await resp.json();
      if (resp.ok) {
        alert('Rasa training placeholder finished: ' + JSON.stringify(data));
      } else {
        alert('Rasa training failed: ' + (data.error || data.details || JSON.stringify(data)));
      }
    } catch (err) {
      alert('Network error: ' + err.message);
    }
  });

  document.getElementById('fetch-models').addEventListener('click', fetchModelMetadata);

  async function fetchModelMetadata() {
    try {
      const resp = await fetch(apiBase + '/model_metadata');
      const data = await resp.json();
      modelsMeta.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      modelsMeta.textContent = 'Error: ' + err.message;
    }
  }

  document.getElementById('do-tokenize').addEventListener('click', async () => {
    const text = document.getElementById('tokenize-text').value || '';
    try {
      const resp = await fetch(apiBase + '/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await resp.json();
      document.getElementById('tokenize-result').textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      document.getElementById('tokenize-result').textContent = 'Error: ' + err.message;
    }
  });

  // initial rendering
  renderEntities();
  renderAnnotationsPreview();
  fetchModelMetadata();
})();
