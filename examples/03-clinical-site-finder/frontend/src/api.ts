// Thin client for the FastAPI backend (see backend/app.py).

export type Site = {
  key: string;
  name: string;
  city: string;
  country: string;
  kind: string;
  area: number;
  area_name: string;
  secondary_area_name: string | null;
  capacity: number;
  startup_days: number;
  enrollment_rate: number;
  score?: number;
  trials?: number;
};

export type SiteProfile = Site & {
  trial_count: number;
  indications: { name: string; area: string; trials: number }[];
  sponsors: { name: string; trials: number }[];
  recent_trials: { key: string; title: string; phase: string; year: number; sponsor: string }[];
};

export type Indication = { key: string; name: string; area: string };

export type Recommendation = {
  indication: Indication;
  seed_sites: number;
  experienced_total: number;
  experienced: Site[];
  lookalikes: Site[];
};

export type Job = {
  state: "idle" | "running" | "done" | "error";
  step: string;
  epoch: number;
  epochs: number;
  metrics: { test_accuracy?: number; seconds?: number };
  error: string | null;
};

export type Status = {
  mode: "arangodb" | "offline";
  error: string | null;
  has_graph: boolean;
  has_embeddings: boolean;
  counts: { sites?: number; trials?: number; indications?: number };
  job: Job;
};

export type MapPoint = {
  key: string;
  x: number;
  y: number;
  area: number;
  name: string;
  city: string;
  country: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  status: () => request<Status>("/api/status"),
  train: (epochs: number) =>
    request<Job>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, embedding_size: 64 }),
    }),
  facets: () => request<{ indications: Indication[]; countries: string[] }>("/api/facets"),
  recommend: (indication: string, country: string) => {
    const params = new URLSearchParams({ indication, k: "10" });
    if (country) params.set("country", country);
    return request<Recommendation>(`/api/recommend?${params}`);
  },
  sites: (q: string) => request<Site[]>(`/api/sites?limit=6&q=${encodeURIComponent(q)}`),
  site: (key: string) => request<SiteProfile>(`/api/sites/${key}`),
  similar: (key: string) => request<Site[]>(`/api/sites/${key}/similar?k=6`),
  map: () => request<MapPoint[]>("/api/map"),
};

// One colour per therapeutic area, in the order of the area ids.
export const AREAS = [
  { name: "Oncology", color: "#7B4BB7" },
  { name: "Cardiology", color: "#C2413B" },
  { name: "Neurology", color: "#2F6FB3" },
  { name: "Immunology", color: "#C98A1B" },
  { name: "Infectious Disease", color: "#2E8B6E" },
  { name: "Endocrinology", color: "#5C6B7A" },
];
export const areaColor = (name: string) =>
  AREAS.find((a) => a.name === name)?.color ?? "#8a94a3";
