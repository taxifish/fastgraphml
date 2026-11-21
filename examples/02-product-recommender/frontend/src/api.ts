// Thin client for the FastAPI backend (see backend/app.py).

export type Product = {
  key: string;
  name: string;
  category: string;
  label: number | null;
  degree?: number;
  score?: number | null;
  bought_together?: boolean;
};

export type Category = { label: number; name: string; count: number };

export type Job = {
  state: "idle" | "running" | "done" | "error";
  step: string;
  epoch: number;
  epochs: number;
  metrics: {
    model?: string;
    embedding_size?: number;
    val_accuracy?: number;
    test_accuracy?: number;
    seconds?: number;
  };
  error: string | null;
};

export type Status = {
  mode: "arangodb" | "offline";
  error: string | null;
  has_graph: boolean;
  has_embeddings: boolean;
  counts: { products?: number; co_purchases?: number };
  job: Job;
};

export type MapPoints = { keys: string[]; x: number[]; y: number[]; labels: number[] };

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
  train: (model: string, epochs: number, embedding_size: number) =>
    request<Job>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, epochs, embedding_size }),
    }),
  categories: () => request<Category[]>("/api/categories"),
  products: (category: number | null, q: string, offset: number) => {
    const params = new URLSearchParams({ limit: "50", offset: String(offset) });
    if (category !== null) params.set("category", String(category));
    if (q) params.set("q", q);
    return request<Product[]>(`/api/products?${params}`);
  },
  product: (key: string) => request<Product>(`/api/products/${key}`),
  similar: (key: string) => request<Product[]>(`/api/products/${key}/similar?k=12`),
  boughtTogether: (key: string) =>
    request<Product[]>(`/api/products/${key}/bought-together?limit=12`),
  map: () => request<MapPoints>("/api/map"),
};

// One colour per product category, used by the legend, the lists and the map.
export const CATEGORY_COLORS = [
  "#1E6B52", "#B4692B", "#3D5A99", "#8C3F6D", "#6E7F1F",
  "#C2442E", "#1F8A9C", "#7A5BB5", "#A8842A", "#4A4F57",
];
export const colorOf = (label: number | null | undefined) =>
  label === null || label === undefined ? "#8A948F" : CATEGORY_COLORS[label % 10];
