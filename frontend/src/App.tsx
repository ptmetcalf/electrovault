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
  auto_assigned?: boolean;
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

type FaceGroupProposal = {
  id: string;
  status: string;
  suggested_label?: string | null;
  score_min?: number | null;
  score_max?: number | null;
  score_mean?: number | null;
  size: number;
  members: FacePreview[];
  created_at?: string | null;
};

type Mode = "search" | "gallery" | "faces" | "events" | "admin";

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
  const [faceGroups, setFaceGroups] = useState<FaceGroupProposal[]>([]);
  const [faceGroupLabelDrafts, setFaceGroupLabelDrafts] = useState<Record<string, string>>({});
  const [faceGroupLoading, setFaceGroupLoading] = useState(false);
  const [faceGroupError, setFaceGroupError] = useState<string | null>(null);
  const [savingGroupId, setSavingGroupId] = useState<string | null>(null);

  const navItems: { id: Mode; label: string; description: string }[] = [
    { id: "search", label: "Search", description: "Semantic search & filters" },
    { id: "gallery", label: "Gallery", description: "Browse the full library" },
    { id: "faces", label: "Faces", description: "Assign, review, and merge people" },
    { id: "events", label: "Events", description: "Timelines and event queries" },
    { id: "admin", label: "Admin", description: "Reindexing and maintenance" },
  ];

  const cards = useMemo(() => {
    if (mode === "gallery") {
      return browseRecords.map((rec) => ({ record: rec, score: null as number | null }));
    }
    if (mode === "faces") {
      return [];
    }
    return results;
  }, [mode, browseRecords, results]);

  const labelBadges = useMemo(() => {
    const source = mode === "gallery" ? browseRecords : results.map((r) => r.record);
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

  function handleNav(next: Mode) {
    setMode(next);
    if (next === "gallery") {
      browse(0);
    } else if (next === "faces") {
      loadFaces(0);
      loadPersons();
      loadFaceGroups();
    } else if (next === "events") {
      fetchEvents();
      setSelectedPhoto(null);
    } else if (next === "search") {
      setSelectedPhoto(null);
    } else if (next === "admin") {
      setSelectedPhoto(null);
    }
  }

  function resetUI() {
    setQuery("");
    setResults([]);
    setBrowseRecords([]);
    setBrowseTotal(0);
    setBrowsePage(0);
    setBrowseLoading(false);
    setEvents([]);
    setLoading(false);
    setError(null);
    setSelectedLabels(new Set());
    setSelectedPhoto(null);
    setSelectedScore(null);
    setPhotoLoading(false);
    setSelectedError(null);
    setFaceEdits({});
    setContextDraft("");
    setImageSize(null);
    setHoverDetectionId(null);
    setMaintenanceStatus(null);
    setMaintenanceError(null);
    setPreserveFaces(true);
    setFaceItems([]);
    setFaceTotal(0);
    setFacePage(0);
    setFaceLoading(false);
    setFaceFilterPerson("");
    setFaceUnassignedOnly(true);
    setFaceLabelDrafts({});
    setSavingFaceId(null);
    setFaceError(null);
    setPersons([]);
    setMergeSourcePerson("");
    setMergeTargetPerson("");
    setMergeStatus(null);
    setMergeError(null);
    setMerging(false);
    setFaceGroups([]);
    setFaceGroupLabelDrafts({});
    setFaceGroupError(null);
    setFaceGroupLoading(false);
    setSavingGroupId(null);
    setMode("search");
  }

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
    value: string,
    opts: { reindex?: boolean } = {}
  ): Promise<PhotoRecord | null> {
    const res = await fetch(`${API_BASE}/photos/${photoId}/faces`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        detection_id: detectionId,
        person_label: value,
        reindex: opts.reindex ?? false,
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
      }
      if (mode === "faces") {
        await loadFaces(facePage);
      }
      await loadPersons();
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
      await loadPersons();
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
    setMode("gallery");
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

  async function loadFaceGroups() {
    setFaceGroupLoading(true);
    setFaceGroupError(null);
    try {
      const res = await fetch(`${API_BASE}/face_groups?status=pending&limit=50`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const proposals: FaceGroupProposal[] = data.proposals || [];
      setFaceGroups(proposals);
      const drafts: Record<string, string> = {};
      proposals.forEach((p) => {
        drafts[p.id] = p.suggested_label || "";
      });
      setFaceGroupLabelDrafts(drafts);
    } catch (err) {
      console.error(err);
      setFaceGroupError("Loading suggestions failed");
    } finally {
      setFaceGroupLoading(false);
    }
  }

  async function rebuildFaceGroups() {
    setFaceGroupLoading(true);
    setFaceGroupError(null);
    setMaintenanceStatus("Rebuilding face suggestions...");
    try {
      const res = await fetch(`${API_BASE}/face_groups/rebuild`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ unassigned_only: true }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadFaceGroups();
      setMaintenanceStatus("Face suggestions rebuilt");
    } catch (err) {
      console.error(err);
      setFaceGroupError("Rebuilding suggestions failed");
      setFaceGroupLoading(false);
      setMaintenanceStatus(null);
    }
  }

  async function acceptFaceGroup(proposal: FaceGroupProposal) {
    const label = faceGroupLabelDrafts[proposal.id]?.trim();
    setSavingGroupId(proposal.id);
    setFaceGroupError(null);
    try {
      const res = await fetch(`${API_BASE}/face_groups/${proposal.id}/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_label: label || proposal.suggested_label || undefined }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadPersons();
      await loadFaceGroups();
      await loadFaces(facePage);
    } catch (err) {
      console.error(err);
      setFaceGroupError("Accepting group failed");
    } finally {
      setSavingGroupId(null);
    }
  }

  async function rejectFaceGroup(proposal: FaceGroupProposal) {
    setSavingGroupId(proposal.id);
    setFaceGroupError(null);
    try {
      const res = await fetch(`${API_BASE}/face_groups/${proposal.id}/reject`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadFaceGroups();
    } catch (err) {
      console.error(err);
      setFaceGroupError("Rejecting group failed");
    } finally {
      setSavingGroupId(null);
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

  const primaryCount =
    mode === "gallery"
      ? browseTotal
      : mode === "faces"
      ? faceTotal
      : mode === "events"
      ? events.length
      : results.length;
  const primaryLabel =
    mode === "faces" ? "Faces" : mode === "gallery" ? "Photos" : mode === "events" ? "Events" : "Results";
  const pageHeadline: Record<Mode, string> = {
    search: "Search across your photos",
    gallery: "Browse the full gallery",
    faces: "Review and name faces",
    events: "Explore events and timelines",
    admin: "Operate and maintain the index",
  };
  const pageSubline: Record<Mode, string> = {
    search: "Natural language, filters, and quick opens",
    gallery: "Paginate through everything with quick labels",
    faces: "Assign identities, merge duplicates, and open source photos",
    events: "Kick off event-focused queries",
    admin: "Run reindex jobs and check health",
  };

  const detailPanel = selectedPhoto ? (
    <div className="detail">
      <div className="detail-head">
        <div>
          <div className="muted">Selected photo</div>
          <div className="row wrap">
            <strong className="truncate">
              {selectedPhoto.file.path.split("/").pop() || selectedPhoto.file.path}
            </strong>
            {selectedScore != null && <span className="pill small">Score {formatScore(selectedScore)}</span>}
          </div>
        </div>
        <div className="row">
          <button type="button" className="secondary" onClick={() => setSelectedPhoto(null)}>
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
              <div className="meta">{selectedPhoto.vision?.description || "No caption yet"}</div>
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
                      list="person-options"
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
                <button type="button" className="secondary" onClick={reindexSelected} disabled={photoLoading}>
                  Reindex photo
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {selectedError && <div className="error">{selectedError}</div>}
    </div>
  ) : null;

  const filteredCards = cards.filter((item) => {
    if (!selectedLabels.size) return true;
    const labels = item.record.classifications.map((c) => c.label);
    return Array.from(selectedLabels).every((l) => labels.includes(l));
  });

  const cardsContent = (
    <>
      <div className="panel-head">
        <h3>{mode === "gallery" ? "Gallery" : "Results"}</h3>
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
      <div className="cards">
        {(loading || browseLoading) && <div className="empty">Loading...</div>}
        {!loading && !browseLoading && filteredCards.length === 0 && (
          <div className="empty">
            {mode === "gallery"
              ? "No photos on this page yet."
              : 'No results yet. Try "family" or "birthday".'}
          </div>
        )}
        {!loading &&
          !browseLoading &&
          filteredCards.map((item) => {
            const path = item.record.file.path.split("/").pop() || item.record.file.path;
            const people = item.record.faces
              .map((f) => f.person_id || f.label)
              .filter(Boolean)
              .join(", ");
            const labels = Array.from(new Set(item.record.classifications.map((c) => c.label))).slice(0, 6);
            const thumb = `${API_BASE}/thumb/${item.record.file.id}`;
            return (
              <div key={item.record.file.id} className="card">
                <div className="flex-between">
                  <strong className="truncate">{path}</strong>
                  {item.score != null && <span className="pill small">Score {formatScore(item.score)}</span>}
                </div>
                <div className="thumb">
                  <img src={thumb} alt={path} />
                </div>
                <div className="meta">{item.record.vision ? item.record.vision.description : "No caption"}</div>
                {people && <div className="meta">People: {people}</div>}
                <div className="row wrap">
                  {labels.map((l) => (
                    <span key={l} className="badge">
                      {l}
                    </span>
                  ))}
                </div>
                <button type="button" className="secondary" onClick={() => openPhoto(item.record, item.score ?? undefined)}>
                  Faces & context
                </button>
              </div>
            );
          })}
      </div>
      {mode === "gallery" && browseTotal > PAGE_SIZE && (
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
  );

  return (
    <div className="app-shell">
      <aside className="sidebar panel">
        <div className="title-column sidebar-brand">
          <div className="title-row">
            <h1>Photo Brain</h1>
            <span className="pill">Local-first</span>
          </div>
          <p className="subtitle">Operator console for search, faces, and events</p>
          <div className="muted small">API on port 8000</div>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${mode === item.id ? "active" : ""}`}
              onClick={() => handleNav(item.id)}
            >
              <div className="nav-label">{item.label}</div>
              <div className="nav-desc">{item.description}</div>
            </button>
          ))}
        </nav>
        <div className="sidebar-section">
          <div className="panel-head">
            <h4>Events</h4>
            <button type="button" className="secondary compact" onClick={fetchEvents}>
              Refresh
            </button>
          </div>
          <ul className="events slim">
            {events.slice(0, 8).map((evt) => (
              <li
                key={evt.id}
                onClick={() => {
                  setQuery(`event:${evt.id}`);
                  search(`event:${evt.id}`);
                }}
              >
                <div className="row space">
                  <span className="truncate">{evt.title}</span>
                  <span className="pill small muted">{evt.photo_ids.length}</span>
                </div>
              </li>
            ))}
            {!events.length && <div className="muted small">No events yet.</div>}
            {events.length > 8 && <div className="muted small">More in Events view</div>}
          </ul>
        </div>
      </aside>
      <div className="content">
        <header className="page-header">
          <div>
            <div className="muted small">{pageSubline[mode]}</div>
            <h2 className="page-title">{pageHeadline[mode]}</h2>
          </div>
          <div className="row wrap">
            <div className="stat">
              <div className="label">{primaryLabel}</div>
              <div className="value">{primaryCount}</div>
            </div>
            {mode === "events" && (
              <div className="stat">
                <div className="label">Events</div>
                <div className="value">{events.length}</div>
              </div>
            )}
            <button type="button" className="secondary" onClick={resetUI}>
              Reset view
            </button>
          </div>
        </header>
        <div className="content-body">
          {mode === "search" && (
            <>
              <section className="panel content-panel">
                <div className="panel-head">
                  <h3>Semantic search</h3>
                  <div className="row wrap">
                    <button type="button" className="secondary" onClick={() => handleNav("gallery")}>
                      Gallery
                    </button>
                    <button type="button" className="secondary" onClick={() => handleNav("faces")}>
                      Faces
                    </button>
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
                  <div className="row wrap">
                    <button type="submit" disabled={loading}>
                      {loading ? "Searching..." : "Run search"}
                    </button>
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => browse(0)}
                      disabled={browseLoading}
                    >
                      {browseLoading ? "Loading gallery..." : "Open gallery"}
                    </button>
                    <button type="button" className="secondary" onClick={resetUI}>
                      Clear
                    </button>
                  </div>
                  {error && <div className="error">{error}</div>}
                </form>
              </section>
              <section className="panel content-panel">
                {detailPanel}
                {cardsContent}
              </section>
            </>
          )}
          {mode === "gallery" && (
            <>
              <section className="panel content-panel">
                <div className="panel-head">
                  <h3>Gallery</h3>
                  <div className="row wrap">
                    <button type="button" className="secondary" onClick={() => browse(0)} disabled={browseLoading}>
                      {browseLoading ? "Loading..." : "Refresh gallery"}
                    </button>
                    <button type="button" className="secondary" onClick={() => handleNav("search")}>
                      Back to search
                    </button>
                  </div>
                </div>
                <div className="row space muted small">
                  <span>
                    Page {browsePage + 1} / {Math.max(1, Math.ceil(Math.max(browseTotal, 1) / PAGE_SIZE))}
                  </span>
                  <div className="row wrap">
                    <button
                      type="button"
                      className="secondary compact"
                      disabled={browsePage === 0 || browseLoading}
                      onClick={() => browse(Math.max(0, browsePage - 1))}
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      className="secondary compact"
                      disabled={(browsePage + 1) * PAGE_SIZE >= browseTotal || browseLoading}
                      onClick={() => browse(browsePage + 1)}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </section>
              <section className="panel content-panel">
                {detailPanel}
                {cardsContent}
              </section>
            </>
          )}
          {mode === "faces" && (
            <section className="panel content-panel">
              {detailPanel}
              <div className="panel-head">
                <h4>Suggested groups</h4>
                <div className="row wrap">
                  <button type="button" className="secondary" onClick={loadFaceGroups} disabled={faceGroupLoading}>
                    {faceGroupLoading ? "Loading..." : "Refresh"}
                  </button>
                  <button type="button" className="secondary" onClick={rebuildFaceGroups} disabled={faceGroupLoading}>
                    Rebuild suggestions
                  </button>
                </div>
              </div>
              {faceGroupError && <div className="error">{faceGroupError}</div>}
              <div className="cards">
                {faceGroupLoading && <div className="empty">Loading suggestions...</div>}
                {!faceGroupLoading && faceGroups.length === 0 && (
                  <div className="empty">No pending suggestions.</div>
                )}
                {!faceGroupLoading &&
                  faceGroups.map((group) => {
                    const exemplar = group.members[0];
                    const cropId = exemplar?.detection.id;
                    const thumb =
                      cropId != null
                        ? `${API_BASE}/faces/${cropId}/crop?size=420`
                        : exemplar
                        ? `${API_BASE}/thumb/${exemplar.photo.id}`
                        : "";
                    return (
                      <div key={group.id} className="card">
                        <div className="flex-between">
                          <strong>Group of {group.size}</strong>
                          <span className="pill small">
                            {group.score_min != null && group.score_max != null
                              ? `Sim ${group.score_min}-${group.score_max}`
                              : "Similarity"}
                          </span>
                        </div>
                        {thumb && (
                          <div className="thumb">
                            <img src={thumb} alt="Group sample" />
                          </div>
                        )}
                        <div className="stack">
                          <input
                            type="text"
                            value={faceGroupLabelDrafts[group.id] ?? ""}
                            placeholder={group.suggested_label || "Name this person"}
                            list="person-options"
                            onChange={(e) =>
                              setFaceGroupLabelDrafts((prev) => ({ ...prev, [group.id]: e.target.value }))
                            }
                          />
                        <div className="row wrap">
                          <button
                            type="button"
                            onClick={() => acceptFaceGroup(group)}
                            disabled={savingGroupId === group.id}
                          >
                            {savingGroupId === group.id ? "Accepting..." : "Accept & merge"}
                          </button>
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => rejectFaceGroup(group)}
                            disabled={savingGroupId === group.id}
                          >
                            Reject
                          </button>
                          <div className="muted small">
                            {group.suggested_label ? `Suggested: ${group.suggested_label}` : "No suggestion"}
                          </div>
                        </div>
                      </div>
                        <div className="row wrap">
                          {group.members.map((m) => {
                            const detId = m.detection.id ?? 0;
                            const cropUrl =
                              detId != null
                                ? `${API_BASE}/faces/${detId}/crop?size=160`
                                : `${API_BASE}/thumb/${m.photo.id}`;
                            const path = m.photo.path.split("/").pop() || m.photo.path;
                            return (
                              <div key={`${group.id}-${detId}`} className="face-chip">
                                <img src={cropUrl} alt={path} />
                                <div className="muted small truncate">{path}</div>
                                <button
                                  type="button"
                                  className="secondary compact"
                                  onClick={() =>
                                    openPhoto({
                                      file: m.photo,
                                      vision: undefined,
                                      classifications: [],
                                      detections: [m.detection],
                                      faces: m.identity ? [m.identity] : [],
                                    })
                                  }
                                >
                                  Open
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
              </div>
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
                    <select value={mergeSourcePerson} onChange={(e) => setMergeSourcePerson(e.target.value)}>
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
                    <select value={mergeTargetPerson} onChange={(e) => setMergeTargetPerson(e.target.value)}>
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
                {(faceLoading || loading) && <div className="empty">Loading faces...</div>}
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
                          {face.identity?.auto_assigned && <span className="pill small muted">Auto</span>}
                            <span className="pill small">Confidence {formatScore(face.detection.confidence)}</span>
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
            </section>
          )}
          {mode === "events" && (
            <section className="panel content-panel">
              <div className="panel-head">
                <h3>Events</h3>
                <button type="button" className="secondary" onClick={fetchEvents}>
                  Refresh
                </button>
              </div>
              <div className="stack">
                {events.map((evt) => (
                  <div key={evt.id} className="card">
                    <div className="flex-between">
                      <strong className="truncate">{evt.title}</strong>
                      <span className="pill small">{evt.photo_ids.length} photos</span>
                    </div>
                    <div className="row wrap">
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          const queryText = `event:${evt.id}`;
                          setQuery(queryText);
                          search(queryText);
                        }}
                      >
                        Search this event
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          setMode("gallery");
                          browse(0);
                        }}
                      >
                        Open gallery
                      </button>
                    </div>
                  </div>
                ))}
                {!events.length && <div className="empty">No events yet.</div>}
              </div>
            </section>
          )}
          {mode === "admin" && (
            <section className="panel content-panel">
              <div className="panel-head">
                <h3>Admin & maintenance</h3>
                <button type="button" className="secondary" onClick={() => handleNav("search")}>
                  Back to search
                </button>
              </div>
              <div className="stack">
                <div className="row wrap">
                  <button type="button" className="secondary" onClick={() => runReindex("pending")}>
                    Reindex pending
                  </button>
                  <button type="button" className="secondary" onClick={() => runReindex("full")}>
                    Reindex all
                  </button>
                </div>
                {maintenanceStatus && <div className="muted small">{maintenanceStatus}</div>}
                {maintenanceError && <div className="error">{maintenanceError}</div>}
                <div className="muted small">
                  Use this pane for operator workflows - reindex respects context and preserves faces when enabled on
                  the photo detail.
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
