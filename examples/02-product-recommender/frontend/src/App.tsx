import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  colorOf,
  type Category,
  type MapPoints,
  type Product,
  type Status,
} from "./api";
import EmbeddingMap from "./EmbeddingMap";

const EMPTY_MAP: MapPoints = { keys: [], x: [], y: [], labels: [] };
const number = (n: number | undefined) => (n ?? 0).toLocaleString("en-US");
const percent = (n: number | undefined) => `${Math.round((n ?? 0) * 100)}%`;

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [category, setCategory] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [products, setProducts] = useState<Product[]>([]);
  const [selected, setSelected] = useState<Product | null>(null);
  const [similar, setSimilar] = useState<Product[]>([]);
  const [bought, setBought] = useState<Product[]>([]);
  const [points, setPoints] = useState<MapPoints>(EMPTY_MAP);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("SAGE");
  const [epochs, setEpochs] = useState(10);
  const wasRunning = useRef(false);

  const refreshStatus = useCallback(() => {
    api.status().then(setStatus).catch((e) => setError(String(e.message)));
  }, []);

  // poll the status: quickly while a training runs
  const running = status?.job.state === "running";
  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, running ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [refreshStatus, running]);

  const hasGraph = status?.has_graph ?? false;
  const hasEmbeddings = status?.has_embeddings ?? false;

  // (re)load catalogue and map when the graph or new embeddings appear
  useEffect(() => {
    if (!hasGraph) return;
    api.categories().then(setCategories).catch(() => {});
  }, [hasGraph]);

  useEffect(() => {
    if (hasEmbeddings && !running) api.map().then(setPoints).catch(() => {});
  }, [hasEmbeddings, running]);

  useEffect(() => {
    if (!hasGraph) return;
    const timer = setTimeout(() => {
      api.products(category, query.trim(), 0).then(setProducts).catch(() => {});
    }, 200);
    return () => clearTimeout(timer);
  }, [hasGraph, category, query]);

  const select = useCallback(
    (key: string) => {
      setError(null);
      window.history.replaceState(null, "", `?product=${encodeURIComponent(key)}`);
      api.product(key).then(setSelected).catch((e) => setError(String(e.message)));
      api.boughtTogether(key).then(setBought).catch(() => setBought([]));
      if (hasEmbeddings) api.similar(key).then(setSimilar).catch(() => setSimilar([]));
      else setSimilar([]);
    },
    [hasEmbeddings],
  );

  // deep link: /?product=<key> opens that product
  const opened = useRef(false);
  useEffect(() => {
    const key = new URLSearchParams(window.location.search).get("product");
    if (key && status && hasGraph && !opened.current) {
      opened.current = true;
      select(key);
    }
  }, [status, hasGraph, select]);

  // after a training finishes: refresh the lists of the selected product
  useEffect(() => {
    if (wasRunning.current && !running && selected) select(selected.key);
    wasRunning.current = running;
  }, [running, selected, select]);

  function train() {
    setError(null);
    api
      .train(model, epochs, 64)
      .then(refreshStatus)
      .catch((e) => setError(String(e.message)));
  }

  function loadMore() {
    api
      .products(category, query.trim(), products.length)
      .then((more) => setProducts([...products, ...more]))
      .catch(() => {});
  }

  const job = status?.job;
  const metrics = job?.metrics ?? {};
  const newFinds = similar.filter((p) => !p.bought_together).length;

  return (
    <div className="app">
      <header className="bar">
        <div className="brand">
          <h1>Product recommender</h1>
          <p>
            {status
              ? `${number(status.counts.products)} products, ${number(
                  status.counts.co_purchases,
                )} co-purchases, ${status.mode === "offline" ? "offline data" : "ArangoDB"}`
              : "Connecting"}
          </p>
        </div>
        <div className="trainer">
          {running ? (
            <div className="progress" role="status">
              <span>
                {job!.step}
                {job!.step === "Training" && `, epoch ${job!.epoch} of ${job!.epochs}`}
              </span>
              <progress
                max={job!.epochs + 2}
                value={job!.step === "Training" ? job!.epoch : job!.epochs + 1}
              />
            </div>
          ) : (
            <>
              {metrics.test_accuracy !== undefined && (
                <span className="metric">
                  {metrics.model} embeddings classify {percent(metrics.test_accuracy)} of
                  held-out products correctly
                </span>
              )}
              <label>
                Model
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  <option value="SAGE">GraphSAGE</option>
                  <option value="GAT">GAT</option>
                </select>
              </label>
              <label>
                Epochs
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value) || 1)}
                />
              </label>
              <button className="primary" onClick={train} disabled={!status || !!status.error}>
                {hasEmbeddings ? "Retrain model" : "Train model"}
              </button>
            </>
          )}
        </div>
      </header>

      {(error || status?.error || job?.error) && (
        <div className="notice" role="alert">
          {error || status?.error || `Training failed. ${job?.error}`}
        </div>
      )}

      <main className="columns">
        <aside className="catalogue">
          <input
            className="search"
            type="search"
            placeholder="Find a product by its number"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Find a product by its number"
          />
          <ul className="categories">
            <li>
              <button
                className={category === null ? "on" : ""}
                onClick={() => setCategory(null)}
              >
                <span className="swatch all" />
                All categories
              </button>
            </li>
            {categories.map((c) => (
              <li key={c.label}>
                <button
                  className={category === c.label ? "on" : ""}
                  onClick={() => setCategory(category === c.label ? null : c.label)}
                >
                  <span className="swatch" style={{ background: colorOf(c.label) }} />
                  {c.name}
                  <span className="count">{number(c.count)}</span>
                </button>
              </li>
            ))}
          </ul>
          <ul className="products">
            {products.map((p) => (
              <li key={p.key}>
                <button
                  className={selected?.key === p.key ? "on" : ""}
                  onClick={() => select(p.key)}
                >
                  <span className="swatch" style={{ background: colorOf(p.label) }} />
                  {p.name}
                </button>
              </li>
            ))}
            {products.length > 0 && products.length % 50 === 0 && (
              <li>
                <button className="more" onClick={loadMore}>
                  Show 50 more
                </button>
              </li>
            )}
            {hasGraph && products.length === 0 && (
              <li className="none">No product matches this search.</li>
            )}
          </ul>
        </aside>

        <section className="stage">
          {points.keys.length > 0 ? (
            <EmbeddingMap
              points={points}
              selected={selected?.key ?? null}
              similar={similar}
              boughtTogether={bought}
              dimmedExcept={category}
              onSelect={select}
            />
          ) : (
            <div className="empty">
              <h2>No embeddings yet</h2>
              <p>
                Train a model to turn every product into a vector. Products that are
                bought in similar company end up close together, and the map and the
                recommendations appear here.
              </p>
              {!running && (
                <button className="primary" onClick={train} disabled={!status || !!status.error}>
                  Train model
                </button>
              )}
            </div>
          )}
        </section>

        <aside className="detail">
          {selected ? (
            <>
              <div className="selected">
                <span className="swatch big" style={{ background: colorOf(selected.label) }} />
                <div>
                  <h2>{selected.name}</h2>
                  <p>
                    {selected.category}, bought together with {number(selected.degree)}{" "}
                    other products
                  </p>
                </div>
              </div>

              <h3>Recommended</h3>
              {hasEmbeddings ? (
                <>
                  <p className="hint">
                    Nearest products in embedding space. {newFinds} of {similar.length} were
                    never bought together with this one: the graph alone would not suggest
                    them.
                  </p>
                  <ProductList items={similar} onSelect={select} showNew />
                </>
              ) : (
                <p className="hint">Train a model to get recommendations.</p>
              )}

              <h3>Bought together</h3>
              <p className="hint">
                Direct neighbours in the graph
                {hasEmbeddings && ", ordered by embedding similarity"}.
              </p>
              <ProductList items={bought} onSelect={select} />
            </>
          ) : (
            <div className="empty small">
              <h2>Pick a product</h2>
              <p>Choose one from the list{points.keys.length > 0 && " or click a dot on the map"}.</p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

function ProductList(props: {
  items: Product[];
  onSelect: (key: string) => void;
  showNew?: boolean;
}) {
  if (props.items.length === 0) return <p className="none">Nothing to show.</p>;
  return (
    <ol className="results">
      {props.items.map((p) => (
        <li key={p.key}>
          <button onClick={() => props.onSelect(p.key)}>
            <span className="swatch" style={{ background: colorOf(p.label) }} />
            <span className="key">{p.key}</span>
            <span className="name">{p.category}</span>
            {props.showNew && !p.bought_together && <span className="tag">new find</span>}
            {p.score !== null && p.score !== undefined && (
              <span className="score" title="Cosine similarity of the embeddings">
                <span className="bar-track">
                  <span className="bar-fill" style={{ width: `${Math.max(0, p.score) * 100}%` }} />
                </span>
                {p.score.toFixed(2)}
              </span>
            )}
          </button>
        </li>
      ))}
    </ol>
  );
}
