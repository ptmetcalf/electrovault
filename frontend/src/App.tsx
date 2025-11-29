import { useEffect, useMemo, useState } from "react";
import "./App.css";

type Classification = {
  label: string;
  score: number;
  source: string;
};

type Face = {
  detection_id?: number;
  person_id?: string;
  label?: string;
  confidence?: number;
};

type Vision = {
  description: string;
  user_context?: string;
};

type Detection = {
  id?: number;
  bbox: [number, number, number, number];
  confidence: number;
};

type PhotoRecord = {
  file: { id: string; path: string };
  vision?: Vision;
  classifications: Classification[];
  detections: Detection[];
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
  const [selectedPhoto, setSelectedPhoto] = useState<PhotoRecord | null>(null);
  const [selectedScore, setSelectedScore] = useState<number | null>(null);
  const [photoLoading, setPhotoLoading] = useState(false);
  const [selectedError, setSelectedError] = useState<string | null>(null);
  const [faceEdits, setFaceEdits] = useState<Record<number, string>>({});
  const [contextDraft, setContextDraft] = useState("");
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  const labelBadges = useMemo(() => {
    return results.flatMap((r) => r.record.classifications.map((c) => c.label));
  }, [results]);

  const facesByDetection = useMemo(() => {
    const map = new Map<number, Face>();
    if (!selectedPhoto) return map;
    selectedPhoto.faces.forEach((face) => {
      if (face.detection_id != null) {
        map.set(face.detection_id, face);
      }
    });
    return map;
  }, [selectedPhoto]);

  function updateResultRecord(next: PhotoRecord) {
    setResults((prev) =>
      prev.map((res) =>
        res.record.file.id === next.file.id ? { ...res, record: next } : res
      )
    );
  }

  function hydrateSelection(record: PhotoRecord, score?: number) {
    setSelectedPhoto(record);
    setSelectedScore(score ?? null);
    setContextDraft(record.vision?.user_context || "");
    setImageSize(null);
    const edits: Record<number, string> = {};
    record.faces.forEach((face) => {
      if (face.detection_id != null) {
        edits[face.detection_id] = face.person_id || face.label || "";
      }
    });
    setFaceEdits(edits);
  }

  async function openPhoto(record: PhotoRecord, score?: number) {
    setPhotoLoading(true);
    setSelectedError(null);
    try {
      const res = await fetch(`${API_BASE}/photos/${record.file.id}`);
      if (!res.ok) {
        hydrateSelection(record, score);
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.photo) {
        hydrateSelection(data.photo, score);
        updateResultRecord(data.photo);
      } else {
        hydrateSelection(record, score);
      }
    } catch (err) {
      console.error(err);
      setSelectedError("Unable to load photo details");
    } finally {
      setPhotoLoading(false);
    }
  }

  async function saveContext() {
    if (!selectedPhoto) return;
    setPhotoLoading(true);
    setSelectedError(null);
    try {
      const res = await fetch(`${API_BASE}/photos/${selectedPhoto.file.id}/context`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: contextDraft, reindex: true }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.photo) {
        hydrateSelection(data.photo, selectedScore ?? undefined);
        updateResultRecord(data.photo);
      }
    } catch (err) {
      console.error(err);
      setSelectedError("Saving context failed");
    } finally {
      setPhotoLoading(false);
    }
  }

  async function saveFaceLabel(detectionId: number) {
    if (!selectedPhoto) return;
    const value = faceEdits[detectionId]?.trim();
    if (!value) {
      setSelectedError("Name cannot be empty");
      return;
    }
    setPhotoLoading(true);
    setSelectedError(null);
    try {
      const res = await fetch(`${API_BASE}/photos/${selectedPhoto.file.id}/faces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          detection_id: detectionId,
          person_label: value,
          reindex: true,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.photo) {
        hydrateSelection(data.photo, selectedScore ?? undefined);
        updateResultRecord(data.photo);
      }
    } catch (err) {
      console.error(err);
      setSelectedError("Updating face label failed");
    } finally {
      setPhotoLoading(false);
    }
  }

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
    setSelectedPhoto(null);
    setSelectedScore(null);
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
                  setSelectedPhoto(null);
                  setSelectedScore(null);
                  setFaceEdits({});
                  setContextDraft("");
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
          {selectedPhoto && (
            <div className="detail">
              <div className="detail-head">
                <div>
                  <div className="muted">Selected photo</div>
                  <div className="row wrap">
                    <strong className="truncate">
                      {selectedPhoto.file.path.split("/").pop() || selectedPhoto.file.path}
                    </strong>
                    {selectedScore != null && (
                      <span className="pill small">Score {formatScore(selectedScore)}</span>
                    )}
                  </div>
                </div>
                <div className="row">
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => setSelectedPhoto(null)}
                  >
                    Close
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => openPhoto(selectedPhoto, selectedScore ?? undefined)}
                  >
                    Refresh
                  </button>
                </div>
              </div>
              <div className="detail-body">
                <div className="image-frame">
                  <img
                    src={`${API_BASE}/thumb/${selectedPhoto.file.id}`}
                    alt="Selected"
                    onLoad={(evt) => {
                      setImageSize({
                        width: evt.currentTarget.naturalWidth,
                        height: evt.currentTarget.naturalHeight,
                      });
                    }}
                  />
                  {imageSize &&
                    (selectedPhoto.detections || [])
                      .filter((det) => det.id != null)
                      .map((det) => {
                        if (!det.id || !imageSize.width || !imageSize.height) return null;
                        const [x1, y1, x2, y2] = det.bbox;
                        const left = (x1 / imageSize.width) * 100;
                        const top = (y1 / imageSize.height) * 100;
                        const width = ((x2 - x1) / imageSize.width) * 100;
                        const height = ((y2 - y1) / imageSize.height) * 100;
                        const label =
                          faceEdits[det.id] ||
                          facesByDetection.get(det.id)?.person_id ||
                          facesByDetection.get(det.id)?.label ||
                          `Face ${det.id}`;
                        return (
                          <div
                            key={det.id}
                            className="face-box"
                            style={{
                              left: `${left}%`,
                              top: `${top}%`,
                              width: `${width}%`,
                              height: `${height}%`,
                            }}
                          >
                            <span className="face-label">{label}</span>
                          </div>
                        );
                      })}
                </div>
                <div className="detail-meta">
                  <div className="stack">
                    <div>
                      <div className="muted small">Caption</div>
                      <div className="meta">
                        {selectedPhoto.vision?.description || "No caption yet"}
                      </div>
                    </div>
                    <div className="stack">
                      <div className="muted small">User context</div>
                      <textarea
                        value={contextDraft}
                        rows={3}
                        onChange={(e) => setContextDraft(e.target.value)}
                        placeholder="Add reminders, places, or who is in the scene"
                      />
                      <div className="row">
                        <button type="button" onClick={saveContext} disabled={photoLoading}>
                          {photoLoading ? "Saving..." : "Save context & reindex"}
                        </button>
                        <button
                          type="button"
                          className="secondary"
                          onClick={() => setContextDraft(selectedPhoto.vision?.user_context || "")}
                        >
                          Reset
                        </button>
                      </div>
                    </div>
                    <div className="stack">
                      <div className="muted small">Faces</div>
                      {(selectedPhoto.detections || []).length === 0 && (
                        <div className="muted">No face detections yet.</div>
                      )}
                      {(selectedPhoto.detections || []).map((det) => {
                        const face = det.id ? facesByDetection.get(det.id) : undefined;
                        const value = det.id ? faceEdits[det.id] ?? face?.person_id ?? face?.label ?? "" : "";
                        return (
                          <div key={det.id || `${det.bbox[0]}-${det.bbox[1]}`} className="face-row">
                            <div className="muted">ID {det.id ?? "?"}</div>
                            <input
                              type="text"
                              value={value}
                              onChange={(e) => {
                                if (!det.id) return;
                                setFaceEdits((prev) => ({ ...prev, [det.id!]: e.target.value }));
                              }}
                              placeholder="Name this face"
                            />
                            {det.id && (
                              <button
                                type="button"
                                className="secondary"
                                onClick={() => saveFaceLabel(det.id!)}
                                disabled={photoLoading}
                              >
                                Save name
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
              {selectedError && <div className="error">{selectedError}</div>}
            </div>
          )}
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
                  const thumb = `${API_BASE}/thumb/${item.record.file.id}`;
                  return (
                    <div key={item.record.file.id} className="card">
                      <div className="flex-between">
                        <strong className="truncate">{path}</strong>
                        <span className="pill small">Score {formatScore(item.score)}</span>
                      </div>
                      <div className="thumb">
                        <img src={thumb} alt={path} />
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
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => openPhoto(item.record, item.score)}
                      >
                        Faces & context
                      </button>
                    </div>
                  );
                })}
          </div>
        </section>
      </main>
    </div>
  );
}
