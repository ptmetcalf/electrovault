import os

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from photo_brain.core.env import load_dotenv_if_present
from photo_brain.events import group_events, summarize_events
from photo_brain.index import PgVectorBackend, init_db, session_factory
from photo_brain.search import execute_search, plan_search

app = FastAPI(title="Photo Brain API")

load_dotenv_if_present()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./photo_brain.db")
engine = init_db(DATABASE_URL)
SessionLocal = session_factory(engine)
vector_backend = PgVectorBackend()


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


def get_session() -> Session:
    with SessionLocal() as session:
        yield session


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(
    request: SearchRequest, session: Session = Depends(get_session)
) -> dict:
    search_query = plan_search(request.query, limit=request.limit)
    results = execute_search(session, vector_backend, search_query)
    return {"results": [result.model_dump() for result in results]}


@app.get("/events")
def list_events(session: Session = Depends(get_session)) -> dict:
    group_events(session)
    summaries = summarize_events(session)
    return {"events": [event.model_dump() for event in summaries]}


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Photo Brain</title>
      <style>
        body { margin: 0; font-family: 'Segoe UI', sans-serif; background: linear-gradient(120deg, #0f172a, #111827); color: #e5e7eb; }
        header { padding: 24px; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.07); }
        h1 { margin: 0; letter-spacing: 0.5px; }
        .container { display: grid; grid-template-columns: 300px 1fr; gap: 16px; padding: 20px; }
        .panel { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
        input[type="text"] { width: 100%; padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.08); color: #e5e7eb; }
        button { padding: 10px 14px; border: none; border-radius: 10px; background: #10b981; color: #0b1324; font-weight: 700; cursor: pointer; margin-top: 8px; }
        button:hover { background: #34d399; }
        ul { list-style: none; padding: 0; margin: 0; }
        li { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); cursor: pointer; }
        .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
        .card { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
        .meta { color: #a5b4fc; font-size: 12px; }
      </style>
    </head>
    <body>
      <header>
        <h1>Photo Brain</h1>
        <p>Search photos, browse events, stay local.</p>
      </header>
      <div class="container">
        <section class="panel">
          <h3>Search</h3>
          <form id="search-form">
            <input id="query" type="text" placeholder="e.g. beach person:alice after:2024-01-01" />
            <button type="submit">Search</button>
          </form>
          <div id="status" class="meta"></div>
          <h4>Events</h4>
          <ul id="events"></ul>
        </section>
        <section class="panel">
          <h3>Results</h3>
          <div id="results" class="results"></div>
        </section>
      </div>
      <script>
        async function fetchEvents() {
          const res = await fetch('/events');
          const data = await res.json();
          const list = document.getElementById('events');
          list.innerHTML = '';
          data.events.forEach(evt => {
            const li = document.createElement('li');
            li.textContent = `${evt.title} (${evt.photo_ids.length} photos)`;
            li.onclick = () => {
              const q = document.getElementById('query');
              q.value = `event:${evt.id} ` + q.value;
            };
            list.appendChild(li);
          });
        }

        async function search(query) {
          const res = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit: 24 })
          });
          const data = await res.json();
          const container = document.getElementById('results');
          container.innerHTML = '';
          data.results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'card';
            const path = item.record.file.path;
            const score = item.score.toFixed(3);
            const caption = item.record.vision ? item.record.vision.description : 'No caption';
            const people = item.record.faces.map(f => f.person_id || f.label).filter(Boolean).join(', ');
            card.innerHTML = `
              <div><strong>${path.split('/').pop()}</strong></div>
              <div class="meta">Score ${score}</div>
              <div>${caption}</div>
              <div class="meta">${people ? 'People: ' + people : ''}</div>
            `;
            container.appendChild(card);
          });
          document.getElementById('status').textContent = `${data.results.length} results`;
        }

        document.getElementById('search-form').addEventListener('submit', (e) => {
          e.preventDefault();
          search(document.getElementById('query').value);
        });

        fetchEvents();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
