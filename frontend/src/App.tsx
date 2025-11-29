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
  score: number | null;
};

type EventSummary = {
  id: string;
  title: string;
  photo_ids: string[];
};

type FacePreview = {
  detection: Detection;
  identity?: Face | null;
  photo: { id: string; path: string };
};

type Person = {
  id: string;
  display_name: string;
  face_count: number;
  sample_photo_id?: string | null;
};

type Mode = "search" | "browse" | "faces";

const API_BASE = "";
const formatScore = (score: number) => score.toFixed(3);
const PAGE_SIZE = 48;
const FACE_PAGE_SIZE = 40;

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [mode, setMode] = useState<Mode>("search");
  const [browseRecords, setBrowseRecords] = useState<PhotoRecord[]>([]);
  const [browseTotal, setBrowseTotal] = useState(0);
  const [browsePage, setBrowsePage] = useState(0);
  const [browseLoading, setBrowseLoading] = useState(false);
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
  const [hoverDetectionId, setHoverDetectionId] = useState<number | null>(null);
  const [maintenanceStatus, setMaintenanceStatus] = useState<string | null>(null);
  const [maintenanceError, setMaintenanceError] = useState<string | null>(null);
  const [preserveFaces, setPreserveFaces] = useState(true);
  const [faceItems, setFaceItems] = useState<FacePreview[]>([]);
  const [faceTotal, setFaceTotal] = useState(0);
  const [facePage, setFacePage] = useState(0);
  const [faceLoading, setFaceLoading] = useState(false);
  const [faceFilterPerson, setFaceFilterPerson] = useState("");
  const [faceUnassignedOnly, setFaceUnassignedOnly] = useState(true);
  const [faceLabelDrafts, setFaceLabelDrafts] = useState<Record<number, string>>({});
  const [savingFaceId, setSavingFaceId] = useState<number | null>(null);
  const [faceError, setFaceError] = useState<string | null>(null);
  const [persons, setPersons] = useState<Person[]>([]);
  const [mergeSourcePerson, setMergeSourcePerson] = useState("");
  const [mergeTargetPerson, setMergeTargetPerson] = useState("");
  const [mergeStatus, setMergeStatus] = useState<string | null>(null);
  const [mergeError, setMergeError] = useState<string | null>(null);
  const [merging, setMerging] = useState(false);

  const cards = useMemo(() => {
    if (mode === "browse") {
      return browseRecords.map((rec) => ({ record: rec, score: null as number | null }));
    }
    if (mode === "faces") {
      return [];
    }
    return results;
  }, [mode, browseRecords, results]);

  const labelBadges = useMemo(() => {
    const source = mode === "browse" ? browseRecords : results.map((r) => r.record);
    return source.flatMap((r) => r.classifications.map((c) => c.label));
  }, [mode, browseRecords, results]);

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

  const selectedFullImage = selectedPhoto
    ? `${API_BASE}/photos/${selectedPhoto.file.id}/image`
    : null;
  const selectedThumbImage = selectedPhoto ? `${API_BASE}/thumb/${selectedPhoto.file.id}` : null;

  const personOptions = useMemo(
    () =>
      persons.map((p) => ({
        id: p.id,
        label: p.display_name || p.id,
      })),
    [persons]
  );

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

  async function assignFaceLabel(
    photoId: string,
    detectionId: number,
    value: string
  ): Promise<PhotoRecord | null> {
    const res = await fetch(`${API_BASE}/photos/${photoId}/faces`, {
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
    return data.photo || null;
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
      const updated = await assignFaceLabel(selectedPhoto.file.id, detectionId, value);
      if (updated) {
        hydrateSelection(updated, selectedScore ?? undefined);
        updateResultRecord(updated);
      }
    } catch (err) {
      console.error(err);
      setSelectedError("Updating face label failed");
    } finally {
      setPhotoLoading(false);
    }
  }

  async function saveFaceLabelFromGrid(face: FacePreview) {
    const detectionId = face.detection.id;
    if (detectionId == null) return;
    const value = faceLabelDrafts[detectionId]?.trim();
    if (!value) {
      setFaceError("Name cannot be empty");
      return;
    }
    setSavingFaceId(detectionId);
    setFaceError(null);
    try {
      const updated = await assignFaceLabel(face.photo.id, detectionId, value);
      if (updated && selectedPhoto?.file.id === updated.file.id) {
        hydrateSelection(updated, selectedScore ?? undefined);
        updateResultRecord(updated);
      }
      setFaceItems((prev) =>
        prev.map((item) =>
          item.detection.id === detectionId
            ? {
                ...item,
                identity: {
                  detection_id: detectionId,
                  person_id: value,
                  label: value,
                },
              }
            : item
        )
      );
      if (faceUnassignedOnly) {
        await loadFaces(facePage);
      }
    } catch (err) {
      console.error(err);
      setFaceError("Saving face name failed");
    } finally {
      setSavingFaceId(null);
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
    setMode("search");
    setLoading(true);
    setError(null);
    setSelectedPhoto(null);
    setSelectedScore(null);
    setBrowseRecords([]);
    setBrowseTotal(0);
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

  async function browse(page: number) {
    setMode("browse");
    setBrowseLoading(true);
    setError(null);
    setSelectedPhoto(null);
    setSelectedScore(null);
    setResults([]);
    try {
      const res = await fetch(`${API_BASE}/photos?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setBrowseRecords(data.photos || []);
      setBrowseTotal(data.total || 0);
      setBrowsePage(page);
    } catch (err) {
      setError("Browse failed");
      console.error(err);
    } finally {
      setBrowseLoading(false);
    }
  }

  async function loadPersons() {
    try {
      const res = await fetch(`${API_BASE}/persons?limit=200`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setPersons(data.persons || []);
    } catch (err) {
      console.error(err);
    }
  }

  async function loadFaces(page: number, overrides?: { unassignedOnly?: boolean; person?: string }) {
    const useUnassigned = overrides?.unassignedOnly ?? faceUnassignedOnly;
    const personFilter = overrides?.person ?? faceFilterPerson;
    setMode("faces");
    setFaceLoading(true);
    setFaceError(null);
    setError(null);
    setSelectedPhoto(null);
    setSelectedScore(null);
    setResults([]);
    setBrowseRecords([]);
    setBrowseTotal(0);
    try {
      const params = new URLSearchParams();
      params.set("limit", FACE_PAGE_SIZE.toString());
      params.set("offset", (page * FACE_PAGE_SIZE).toString());
      if (useUnassigned) params.set("unassigned", "true");
      if (personFilter.trim()) params.set("person", personFilter.trim());
      const res = await fetch(`${API_BASE}/faces?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const faces: FacePreview[] = data.faces || [];
      setFaceItems(faces);
      setFaceTotal(data.total || faces.length);
      setFacePage(page);
      const drafts: Record<number, string> = {};
      faces.forEach((f) => {
        const id = f.detection.id;
        if (id != null) {
          drafts[id] = f.identity?.label || f.identity?.person_id || "";
        }
      });
      setFaceLabelDrafts(drafts);
      loadPersons();
    } catch (err) {
      console.error(err);
      setFaceError("Loading faces failed");
    } finally {
      setFaceLoading(false);
    }
  }

  async function mergePersonsRequest(sourceId: string, targetId: string) {
    const res = await fetch(`${API_BASE}/persons/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function handleMergePersons() {
    if (!mergeSourcePerson || !mergeTargetPerson) {
      setMergeError("Select both source and target people");
      return;
    }
    if (mergeSourcePerson === mergeTargetPerson) {
      setMergeError("Source and target must be different");
      return;
    }
    setMerging(true);
    setMergeError(null);
    setMergeStatus(null);
    try {
      const data = await mergePersonsRequest(mergeSourcePerson, mergeTargetPerson);
      setMergeStatus(`Merged into ${data.person?.display_name || data.person?.id || "target"}`);
      setMergeSourcePerson("");
      setMergeTargetPerson("");
      await loadPersons();
      await loadFaces(facePage);
      if (selectedPhoto) {
        await openPhoto(selectedPhoto, selectedScore ?? undefined);
      }
    } catch (err) {
      console.error(err);
      setMergeError("Merging people failed");
    } finally {
      setMerging(false);
    }
  }

  async function runReindex(kind: "pending" | "full") {
    setMaintenanceStatus(`Running ${kind} reindex...`);
    setMaintenanceError(null);
    try {
      const res = await fetch(`${API_BASE}/reindex/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMaintenanceStatus(`Reindex ${kind} processed ${data.processed ?? "?"} photos`);
    } catch (err) {
      console.error(err);
      setMaintenanceStatus(null);
      setMaintenanceError(`Reindex ${kind} failed`);
    }
  }

  async function reindexSelected() {
    if (!selectedPhoto) return;
    setPhotoLoading(true);
    setSelectedError(null);
    try {
      const res = await fetch(`${API_BASE}/reindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          photo_id: selectedPhoto.file.id,
          preserve_faces: preserveFaces,
          context: contextDraft || undefined,
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
      setSelectedError("Reindex failed");
    } finally {
      setPhotoLoading(false);
    }
  }

  useEffect(() => {
    fetchEvents();
  }, []);

  return (
    <div className="page">
      <header className="header">
        <div className="title-column">
          <div className="title-row">
            <h1>Photo Brain</h1>
            <span className="pill">Local-first</span>
          </div>
          <p className="subtitle">Agentic console for search, faces, and reindexing</p>
        </div>
        <div className="pill muted">API on port 8000</div>
      </header>
      <main className="grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Search</h3>
            <div className="stat">
              <div className="label">Results</div>
              <div className="value">{mode === "browse" ? browseTotal : results.length}</div>
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
                onClick={() => browse(0)}
                disabled={browseLoading}
              >
                {browseLoading ? "Browsing..." : "Browse all"}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => loadFaces(0)}
                disabled={faceLoading}
              >
                {faceLoading ? "Loading faces..." : "Faces"}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setQuery("");
                  setResults([]);
                  setBrowseRecords([]);
                  setBrowseTotal(0);
                  setFaceItems([]);
                  setFaceTotal(0);
                  setFacePage(0);
                  setFaceLabelDrafts({});
                  setFaceFilterPerson("");
                  setFaceUnassignedOnly(true);
                  setFaceError(null);
                  setPersons([]);
                  setMode("search");
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
            <h4>Maintenance</h4>
          </div>
          <div className="stack">
            <div className="row">
              <button type="button" className="secondary" onClick={() => runReindex("pending")}>
                Reindex pending
              </button>
              <button type="button" className="secondary" onClick={() => runReindex("full")}>
                Reindex all
              </button>
            </div>
            {maintenanceStatus && <div className="muted small">{maintenanceStatus}</div>}
            {maintenanceError && <div className="error">{maintenanceError}</div>}
          </div>
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
                        if (active) {
                          next.delete(l);
                        } else {
                          next.add(l);
                        }
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
                    src={selectedFullImage || selectedThumbImage || ""}
                    alt="Selected"
                    onLoad={(evt) => {
                      setImageSize({
                        width: evt.currentTarget.naturalWidth,
                        height: evt.currentTarget.naturalHeight,
                      });
                    }}
                    onError={(evt) => {
                      if (selectedThumbImage && evt.currentTarget.src !== selectedThumbImage) {
                        evt.currentTarget.src = selectedThumbImage;
                      }
                    }}
                  />
                  {imageSize &&
                    (selectedPhoto.detections || [])
                      .filter((det) => det.id != null)
                      .map((det) => {
                        if (!det.id || !imageSize.width || !imageSize.height) return null;
                        const [x1, y1, x2, y2] = det.bbox;
                        if (x2 <= x1 || y2 <= y1) return null;
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
                            className={`face-box ${hoverDetectionId === det.id ? "active" : ""}`}
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
                          <div
                            key={det.id || `${det.bbox[0]}-${det.bbox[1]}`}
                            className="face-row"
                            onMouseEnter={() => det.id && setHoverDetectionId(det.id)}
                            onMouseLeave={() => setHoverDetectionId(null)}
                          >
                            <div className="muted">ID {det.id ?? "?"}</div>
                            <input
                              type="text"
                              value={value}
                              onChange={(e) => {
                                if (!det.id) return;
                                setFaceEdits((prev) => ({ ...prev, [det.id!]: e.target.value }));
                              }}
                              placeholder="Name this face"
                              onFocus={() => det.id && setHoverDetectionId(det.id)}
                              onBlur={() => setHoverDetectionId(null)}
                            />
                            {det.id && (
                              <button
                                type="button"
                                className="secondary"
                                onClick={() => saveFaceLabel(det.id!)}
                                disabled={photoLoading}
                                onMouseEnter={() => setHoverDetectionId(det.id!)}
                                onMouseLeave={() => setHoverDetectionId(null)}
                              >
                                Save name
                              </button>
                            )}
                          </div>
                        );
                      })}
                      <div className="row wrap">
                        <label className="muted small">
                          <input
                            type="checkbox"
                            checked={preserveFaces}
                            onChange={(e) => setPreserveFaces(e.target.checked)}
                          />{" "}
                          Preserve face identities on reindex
                        </label>
                        <button
                          type="button"
                          className="secondary"
                          onClick={reindexSelected}
                          disabled={photoLoading}
                        >
                          Reindex photo
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              {selectedError && <div className="error">{selectedError}</div>}
            </div>
          )}
          {mode === "faces" ? (
            <>
              <div className="panel-head">
                <h4>Face assignment</h4>
                <div className="row wrap">
                  <label className="muted small">
                    <input
                      type="checkbox"
                      checked={faceUnassignedOnly}
                      onChange={(e) => {
                        setFaceUnassignedOnly(e.target.checked);
                        loadFaces(0, { unassignedOnly: e.target.checked });
                      }}
                    />{" "}
                    Unassigned only
                  </label>
                  <input
                    type="text"
                    value={faceFilterPerson}
                    onChange={(e) => setFaceFilterPerson(e.target.value)}
                    placeholder="Filter by person name"
                  />
                  <button type="button" className="secondary" onClick={() => loadFaces(0)}>
                    Refresh faces
                  </button>
                </div>
              </div>
              <datalist id="person-options">
                {persons.map((p) => (
                  <option key={p.id} value={p.display_name} />
                ))}
              </datalist>
              <div className="panel-head">
                <h4>Merge people</h4>
              </div>
              <div className="stack">
                <div className="row wrap">
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div className="muted small">Source (to merge from)</div>
                    <select
                      value={mergeSourcePerson}
                      onChange={(e) => setMergeSourcePerson(e.target.value)}
                    >
                      <option value="">Select person</option>
                      {personOptions.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div className="muted small">Target (keep)</div>
                    <select
                      value={mergeTargetPerson}
                      onChange={(e) => setMergeTargetPerson(e.target.value)}
                    >
                      <option value="">Select person</option>
                      {personOptions.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <button type="button" onClick={handleMergePersons} disabled={merging}>
                    {merging ? "Merging..." : "Merge people"}
                  </button>
                </div>
                {mergeStatus && <div className="muted small">{mergeStatus}</div>}
                {mergeError && <div className="error">{mergeError}</div>}
              </div>
              {faceError && <div className="error">{faceError}</div>}
              <div className="faces-grid">
                {(faceLoading || loading) && <div className="empty">Loading faces…</div>}
                {!faceLoading && faceItems.length === 0 && (
                  <div className="empty">No faces found for this filter.</div>
                )}
                {!faceLoading &&
                  faceItems.map((face) => {
                    const detectionId = face.detection.id ?? 0;
                    const path = face.photo.path.split("/").pop() || face.photo.path;
                    const draft =
                      faceLabelDrafts[detectionId] ??
                      face.identity?.person_id ??
                      face.identity?.label ??
                      "";
                    const cropUrl = face.detection.id
                      ? `${API_BASE}/faces/${face.detection.id}/crop?size=420`
                      : `${API_BASE}/thumb/${face.photo.id}`;
                    return (
                      <div key={`${face.photo.id}-${detectionId}`} className="face-card">
                        <div className="flex-between">
                          <span className="muted small">#{detectionId}</span>
                          <button
                            type="button"
                            className="secondary"
                            onClick={() =>
                              openPhoto(
                                {
                                  file: face.photo,
                                  vision: undefined,
                                  classifications: [],
                                  detections: [face.detection],
                                  faces: face.identity ? [face.identity] : [],
                                }
                              )
                            }
                          >
                            Open photo
                          </button>
                        </div>
                        <div className="face-crop">
                          <img src={cropUrl} alt={path} />
                        </div>
                        <div className="stack">
                          <input
                            type="text"
                            value={draft}
                            list="person-options"
                            onChange={(e) =>
                              setFaceLabelDrafts((prev) => ({
                                ...prev,
                                [detectionId]: e.target.value,
                              }))
                            }
                            placeholder="Who is this?"
                          />
                          <div className="row wrap">
                            <button
                              type="button"
                              onClick={() => saveFaceLabelFromGrid(face)}
                              disabled={savingFaceId === detectionId}
                            >
                              {savingFaceId === detectionId ? "Saving..." : "Save name"}
                            </button>
                            {face.identity?.person_id && (
                              <span className="pill small">Current: {face.identity.person_id}</span>
                            )}
                            <span className="pill small">
                              Confidence {formatScore(face.detection.confidence)}
                            </span>
                            <span className="pill small truncate">{path}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
              {faceTotal > FACE_PAGE_SIZE && (
                <div className="row space pager">
                  <button
                    type="button"
                    className="secondary"
                    disabled={facePage === 0 || faceLoading}
                    onClick={() => loadFaces(Math.max(0, facePage - 1))}
                  >
                    Previous
                  </button>
                  <div className="muted small">
                    Page {facePage + 1} / {Math.ceil(faceTotal / FACE_PAGE_SIZE)} ({faceTotal} faces)
                  </div>
                  <button
                    type="button"
                    className="secondary"
                    disabled={(facePage + 1) * FACE_PAGE_SIZE >= faceTotal || faceLoading}
                    onClick={() => loadFaces(facePage + 1)}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="cards">
                {(loading || browseLoading) && <div className="empty">Loading…</div>}
                {!loading && !browseLoading && cards.length === 0 && (
                  <div className="empty">No results yet. Try “family” or “birthday”.</div>
                )}
                {!loading &&
                  !browseLoading &&
                  cards
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
                            {item.score != null && (
                              <span className="pill small">Score {formatScore(item.score)}</span>
                            )}
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
                            onClick={() => openPhoto(item.record, item.score ?? undefined)}
                          >
                            Faces & context
                          </button>
                        </div>
                      );
                    })}
              </div>
              {mode === "browse" && browseTotal > PAGE_SIZE && (
                <div className="row space pager">
                  <button
                    type="button"
                    className="secondary"
                    disabled={browsePage === 0 || browseLoading}
                    onClick={() => browse(Math.max(0, browsePage - 1))}
                  >
                    Previous
                  </button>
                  <div className="muted small">
                    Page {browsePage + 1} / {Math.ceil(browseTotal / PAGE_SIZE)} ({browseTotal} photos)
                  </div>
                  <button
                    type="button"
                    className="secondary"
                    disabled={(browsePage + 1) * PAGE_SIZE >= browseTotal || browseLoading}
                    onClick={() => browse(browsePage + 1)}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
