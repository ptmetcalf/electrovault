import { useEffect, useMemo, useState } from "react";
import "./App.css";

type Classification = {
  label: string;
  score: number;
  source: string;
};

type Face = {
  person_id?: string;
  label?: string;
};

type Vision = {
  description: string;
};

type PhotoRecord = {
  file: { id: string; path: string };
  vision?: Vision;
  classifications: Classification[];
  faces: Face[];
};

type SearchResult = {
  record: PhotoRecord;
  score: number;
};

type EventSummary = {
  id: string;
  title: string;
  photo_ids: string[];
};

const API_BASE = "";
const formatScore = (score: number) => score.toFixed(3);

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());

  const labelBadges = useMemo(() => {
    return results.flatMap((r) => r.record.classifications.map((c) => c.label));
  }, [results]);

  async function fetchEvents() {
    try {
      const res = await fetch(`${API_BASE}/events`);
      const data = await res.json();
      setEvents(data.events || []);
    } catch (err) {
      console.error(err);
    }
  }

  async function search(q: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, limit: 24 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError("Search failed");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchEvents();
  }, []);

  return (
    <div className="page">
      <header className="header">
        <div className="title-row">
          <h1>Photo Brain</h1>
          <span className="pill">Local-first</span>
        </div>
        <div className="pill muted">API on port 8000</div>
      </header>
      <main className="grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Search</h3>
            <div className="stat">
              <div className="label">Results</div>
              <div className="value">{results.length}</div>
            </div>
          </div>
          <form
            className="stack"
            onSubmit={(e) => {
              e.preventDefault();
              search(query);
            }}
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. beach event:holiday person:alice after:2024-01-01"
            />
            <div className="row">
              <button type="submit" disabled={loading}>
                {loading ? "Searching..." : "Search"}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setQuery("");
                  setResults([]);
                  setSelectedLabels(new Set());
                  setError(null);
                }}
              >
                Clear
              </button>
            </div>
            {error && <div className="error">{error}</div>}
          </form>
          <div className="panel-head">
            <h4>Events</h4>
            <button type="button" className="secondary" onClick={fetchEvents}>
              Refresh
            </button>
          </div>
          <ul className="events">
            {events.map((evt) => (
              <li
                key={evt.id}
                onClick={() => {
                  setQuery(`event:${evt.id} ${query}`);
                  search(`event:${evt.id} ${query}`);
                }}
              >
                <div className="row space">
                  <span>{evt.title}</span>
                  <span className="pill muted">{evt.photo_ids.length}</span>
                </div>
              </li>
            ))}
            {!events.length && <div className="muted">No events yet.</div>}
          </ul>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h3>Results</h3>
            <div className="row wrap">
              {Array.from(new Set(labelBadges))
                .slice(0, 12)
                .map((l) => {
                  const active = selectedLabels.has(l);
                  return (
                    <span
                      key={l}
                      className={`badge ${active ? "active" : ""}`}
                      onClick={() => {
                        const next = new Set(selectedLabels);
                        active ? next.delete(l) : next.add(l);
                        setSelectedLabels(next);
                      }}
                    >
                      {l}
                    </span>
                  );
                })}
            </div>
          </div>
          <div className="cards">
            {loading && <div className="empty">Loading…</div>}
            {!loading && results.length === 0 && (
              <div className="empty">No results yet. Try “family” or “birthday”.</div>
            )}
            {!loading &&
              results
                .filter((item) => {
                  if (!selectedLabels.size) return true;
                  const labels = item.record.classifications.map((c) => c.label);
                  return Array.from(selectedLabels).every((l) => labels.includes(l));
                })
                .map((item) => {
                  const path = item.record.file.path.split("/").pop() || item.record.file.path;
                  const people = item.record.faces
                    .map((f) => f.person_id || f.label)
                    .filter(Boolean)
                    .join(", ");
                  const labels = Array.from(
                    new Set(item.record.classifications.map((c) => c.label))
                  ).slice(0, 6);
                  return (
                    <div key={item.record.file.id} className="card">
                      <div className="flex-between">
                        <strong className="truncate">{path}</strong>
                        <span className="pill small">Score {formatScore(item.score)}</span>
                      </div>
                      <div className="meta">
                        {item.record.vision ? item.record.vision.description : "No caption"}
                      </div>
                      {people && <div className="meta">People: {people}</div>}
                      <div className="row wrap">
                        {labels.map((l) => (
                          <span key={l} className="badge">
                            {l}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
          </div>
        </section>
      </main>
    </div>
  );
}
