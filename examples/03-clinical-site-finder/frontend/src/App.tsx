import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  areaColor,
  AREAS,
  type Indication,
  type MapPoint,
  type Recommendation,
  type Site,
  type SiteProfile,
  type Status,
} from "./api";
import SiteMap from "./SiteMap";

const number = (n: number | undefined) => (n ?? 0).toLocaleString("en-US");
const params = () => new URLSearchParams(window.location.search);

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [indications, setIndications] = useState<Indication[]>([]);
  const [countries, setCountries] = useState<string[]>([]);
  const [indication, setIndication] = useState(params().get("indication") ?? "");
  const [country, setCountry] = useState(params().get("country") ?? "");
  const [siteKey, setSiteKey] = useState(params().get("site") ?? "");
  const [result, setResult] = useState<Recommendation | null>(null);
  const [profile, setProfile] = useState<SiteProfile | null>(null);
  const [similar, setSimilar] = useState<Site[]>([]);
  const [points, setPoints] = useState<MapPoint[]>([]);
  const [lookup, setLookup] = useState("");
  const [matches, setMatches] = useState<Site[]>([]);
  const [epochs, setEpochs] = useState(15);
  const [error, setError] = useState<string | null>(null);
  const failed = (e: Error) => setError(e.message);

  const refreshStatus = useCallback(() => {
    api.status().then(setStatus).catch(failed);
  }, []);

  const running = status?.job.state === "running";
  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, running ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [refreshStatus, running]);

  const hasGraph = status?.has_graph ?? false;
  const hasEmbeddings = (status?.has_embeddings ?? false) && !running;

  useEffect(() => {
    if (!hasGraph) return;
    api
      .facets()
      .then((f) => {
        setIndications(f.indications);
        setCountries(f.countries);
        setIndication((current) => current || f.indications[0]?.key || "");
      })
      .catch(failed);
  }, [hasGraph]);

  useEffect(() => {
    if (hasEmbeddings) api.map().then(setPoints).catch(failed);
  }, [hasEmbeddings]);

  // the query: runs whenever the sentence at the top changes
  useEffect(() => {
    if (!hasEmbeddings || !indication) return;
    setError(null);
    api.recommend(indication, country).then(setResult).catch(failed);
  }, [hasEmbeddings, indication, country]);

  useEffect(() => {
    if (!siteKey || !hasGraph) return;
    api.site(siteKey).then(setProfile).catch(failed);
    if (hasEmbeddings) api.similar(siteKey).then(setSimilar).catch(() => setSimilar([]));
  }, [siteKey, hasGraph, hasEmbeddings]);

  // keep the address bar in step, so a result can be shared as a link
  useEffect(() => {
    const next = new URLSearchParams();
    if (indication) next.set("indication", indication);
    if (country) next.set("country", country);
    if (siteKey) next.set("site", siteKey);
    window.history.replaceState(null, "", `?${next}`);
  }, [indication, country, siteKey]);

  const lookupTimer = useRef<number | undefined>(undefined);
  function onLookup(text: string) {
    setLookup(text);
    window.clearTimeout(lookupTimer.current);
    if (text.trim().length < 2) return setMatches([]);
    lookupTimer.current = window.setTimeout(() => {
      api.sites(text.trim()).then(setMatches).catch(() => setMatches([]));
    }, 200);
  }

  function train() {
    setError(null);
    api.train(epochs).then(refreshStatus).catch(failed);
  }

  const byArea = useMemo(() => {
    const groups = new Map<string, Indication[]>();
    for (const i of indications) groups.set(i.area, [...(groups.get(i.area) ?? []), i]);
    return [...groups.entries()];
  }, [indications]);

  const job = status?.job;
  const experiencedKeys = useMemo(
    () => new Set(result?.experienced.map((s) => s.key)),
    [result],
  );
  const lookalikeKeys = useMemo(() => new Set(result?.lookalikes.map((s) => s.key)), [result]);

  return (
    <div className="app">
      <header className="bar">
        <div>
          <h1>Clinical site finder</h1>
          <p>
            {status
              ? `${number(status.counts.sites)} sites, ${number(status.counts.trials)} trials, ${
                  status.mode === "offline" ? "synthetic data in memory" : "synthetic data in ArangoDB"
                }`
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
              {job?.metrics.test_accuracy !== undefined && (
                <span className="metric">
                  Embeddings predict a site's primary area with{" "}
                  {Math.round(job.metrics.test_accuracy * 100)}% accuracy
                </span>
              )}
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
              <button className="primary" onClick={train} disabled={!hasGraph}>
                {status?.has_embeddings ? "Retrain model" : "Train model"}
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

      <section className="question">
        <p>
          Find sites for a new{" "}
          <select
            value={indication}
            onChange={(e) => setIndication(e.target.value)}
            aria-label="Indication"
          >
            {byArea.map(([area, items]) => (
              <optgroup key={area} label={area}>
                {items.map((i) => (
                  <option key={i.key} value={i.key}>
                    {i.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>{" "}
          trial in{" "}
          <select value={country} onChange={(e) => setCountry(e.target.value)} aria-label="Country">
            <option value="">any country</option>
            {countries.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </p>
      </section>

      <main className="columns">
        <section className="answers">
          {!status?.has_embeddings && !running ? (
            <div className="empty">
              <h2>The model is not trained yet</h2>
              <p>
                Training walks the graph from site to trial to indication and back, and
                gives every site a vector that summarises its trial history. It takes
                about half a minute.
              </p>
              <button className="primary" onClick={train} disabled={!hasGraph}>
                Train model
              </button>
            </div>
          ) : (
            result && (
              <>
                <SiteList
                  title="Experienced sites"
                  note={`${result.experienced_total} sites${
                    country ? ` in ${country}` : ""
                  } have run ${result.indication.name} trials. Ordered by number of trials.`}
                  sites={result.experienced}
                  kind="experienced"
                  selected={siteKey}
                  onSelect={setSiteKey}
                />
                <SiteList
                  title="Lookalike sites"
                  note={`No ${result.indication.name} trial yet, but a trial history closest to the sites that have. Ordered by similarity.`}
                  sites={result.lookalikes}
                  kind="lookalike"
                  selected={siteKey}
                  onSelect={setSiteKey}
                />
              </>
            )
          )}
        </section>

        <aside className="context">
          {points.length > 0 && (
            <>
              <SiteMap
                points={points}
                selected={siteKey}
                experienced={experiencedKeys}
                lookalikes={lookalikeKeys}
                onSelect={setSiteKey}
              />
              <ul className="legend">
                {AREAS.map((a) => (
                  <li key={a.name}>
                    <span className="dot" style={{ background: a.color }} />
                    {a.name}
                  </li>
                ))}
              </ul>
            </>
          )}

          <div className="lookup">
            <input
              type="search"
              placeholder="Look up a site by name or city"
              aria-label="Look up a site by name or city"
              value={lookup}
              onChange={(e) => onLookup(e.target.value)}
            />
            {matches.length > 0 && (
              <ul>
                {matches.map((s) => (
                  <li key={s.key}>
                    <button
                      onClick={() => {
                        setSiteKey(s.key);
                        setMatches([]);
                        setLookup("");
                      }}
                    >
                      {s.name}, {s.city}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {profile ? (
            <SiteCard profile={profile} similar={similar} onSelect={setSiteKey} />
          ) : (
            <p className="muted">Select a site to see its trial history.</p>
          )}
        </aside>
      </main>
    </div>
  );
}

function SiteList(props: {
  title: string;
  note: string;
  sites: Site[];
  kind: "experienced" | "lookalike";
  selected: string;
  onSelect: (key: string) => void;
}) {
  return (
    <div className={`list ${props.kind}`}>
      <h2>{props.title}</h2>
      <p className="muted">{props.note}</p>
      {props.sites.length === 0 ? (
        <p className="muted none">No site matches.</p>
      ) : (
        <ol>
          {props.sites.map((s) => (
            <li key={s.key}>
              <button
                className={props.selected === s.key ? "on" : ""}
                onClick={() => props.onSelect(s.key)}
              >
                <span className="who">
                  <strong>{s.name}</strong>
                  <span>
                    {s.city}, {s.country}
                  </span>
                </span>
                <span className="area" style={{ color: areaColor(s.area_name) }}>
                  {s.area_name}
                </span>
                <span className="measure">
                  {props.kind === "experienced" ? (
                    <>
                      <strong>{s.trials}</strong> trials
                    </>
                  ) : (
                    <>
                      <span className="track">
                        <span style={{ width: `${Math.max(0, s.score ?? 0) * 100}%` }} />
                      </span>
                      {(s.score ?? 0).toFixed(2)}
                    </>
                  )}
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function SiteCard(props: {
  profile: SiteProfile;
  similar: Site[];
  onSelect: (key: string) => void;
}) {
  const p = props.profile;
  const most = Math.max(1, ...p.indications.map((i) => i.trials));
  return (
    <article className="card">
      <h2>{p.name}</h2>
      <p className="muted">
        {p.kind} in {p.city}, {p.country}. Primary area {p.area_name}
        {p.secondary_area_name && `, also ${p.secondary_area_name}`}.
      </p>
      <dl className="facts">
        <div>
          <dt>Trials</dt>
          <dd>{p.trial_count}</dd>
        </div>
        <div>
          <dt>Start-up</dt>
          <dd>{p.startup_days} days</dd>
        </div>
        <div>
          <dt>Enrollment</dt>
          <dd>{p.enrollment_rate} / month</dd>
        </div>
      </dl>

      <h3>Trial history by indication</h3>
      <ul className="history">
        {p.indications.slice(0, 8).map((i) => (
          <li key={i.name}>
            <span>{i.name}</span>
            <span className="track">
              <span
                style={{ width: `${(i.trials / most) * 100}%`, background: areaColor(i.area) }}
              />
            </span>
            <span className="n">{i.trials}</span>
          </li>
        ))}
      </ul>

      {props.similar.length > 0 && (
        <>
          <h3>Sites with a similar history</h3>
          <ul className="similar">
            {props.similar.map((s) => (
              <li key={s.key}>
                <button onClick={() => props.onSelect(s.key)}>
                  <span>
                    {s.name}, {s.city}
                  </span>
                  <span className="n">{(s.score ?? 0).toFixed(2)}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      <h3>Most frequent sponsors</h3>
      <p className="muted">{p.sponsors.map((s) => `${s.name} (${s.trials})`).join(", ")}</p>
    </article>
  );
}
