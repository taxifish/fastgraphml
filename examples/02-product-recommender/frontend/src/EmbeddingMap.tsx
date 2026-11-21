import { useEffect, useMemo, useRef, useState } from "react";
import { colorOf, type MapPoints, type Product } from "./api";

type Props = {
  points: MapPoints;
  selected: string | null;
  similar: Product[];
  boughtTogether: Product[];
  dimmedExcept: number | null; // category filter of the catalogue
  onSelect: (key: string) => void;
};

const PAD = 18;

/** All products as dots, positioned by the 2D projection of their embeddings.
 *  Drawn on a canvas because there are more than 13,000 of them. */
export default function EmbeddingMap(props: Props) {
  const { points, selected, similar, boughtTogether, dimmedExcept, onSelect } = props;
  const wrapper = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 600, h: 500 });
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);

  const indexOf = useMemo(() => {
    const map = new Map<string, number>();
    points.keys.forEach((key, i) => map.set(key, i));
    return map;
  }, [points]);

  useEffect(() => {
    if (!wrapper.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.max(200, width), h: Math.max(200, height) });
    });
    observer.observe(wrapper.current);
    return () => observer.disconnect();
  }, []);

  const px = (i: number) => PAD + points.x[i] * (size.w - 2 * PAD);
  const py = (i: number) => PAD + (1 - points.y[i]) * (size.h - 2 * PAD);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ratio = window.devicePixelRatio || 1;
    el.width = size.w * ratio;
    el.height = size.h * ratio;
    const ctx = el.getContext("2d")!;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const focus = selected !== null && indexOf.has(selected);
    for (let i = 0; i < points.keys.length; i++) {
      const dimmed = dimmedExcept !== null && points.labels[i] !== dimmedExcept;
      ctx.globalAlpha = dimmed ? 0.06 : focus ? 0.28 : 0.6;
      ctx.fillStyle = colorOf(points.labels[i]);
      ctx.beginPath();
      ctx.arc(px(i), py(i), 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (!focus) return;

    const s = indexOf.get(selected!)!;
    // lines from the selected product to its recommendations
    ctx.strokeStyle = "rgba(180, 105, 43, 0.55)";
    ctx.lineWidth = 1;
    for (const p of similar) {
      const i = indexOf.get(p.key);
      if (i === undefined) continue;
      ctx.beginPath();
      ctx.moveTo(px(s), py(s));
      ctx.lineTo(px(i), py(i));
      ctx.stroke();
    }
    // direct co-purchases: open rings
    ctx.strokeStyle = "#16211C";
    ctx.lineWidth = 1.2;
    for (const p of boughtTogether) {
      const i = indexOf.get(p.key);
      if (i === undefined) continue;
      ctx.beginPath();
      ctx.arc(px(i), py(i), 5, 0, Math.PI * 2);
      ctx.stroke();
    }
    // recommendations: filled copper dots
    for (const p of similar) {
      const i = indexOf.get(p.key);
      if (i === undefined) continue;
      ctx.fillStyle = "#B4692B";
      ctx.beginPath();
      ctx.arc(px(i), py(i), 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
    // the selected product
    ctx.fillStyle = "#16211C";
    ctx.beginPath();
    ctx.arc(px(s), py(s), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [points, size, selected, similar, boughtTogether, dimmedExcept, indexOf]);

  function nearest(event: React.MouseEvent) {
    const rect = canvas.current!.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = -1;
    let bestDistance = 14 * 14;
    for (let i = 0; i < points.keys.length; i++) {
      const dx = px(i) - x;
      const dy = py(i) - y;
      const d = dx * dx + dy * dy;
      if (d < bestDistance) {
        best = i;
        bestDistance = d;
      }
    }
    return best === -1 ? null : { i: best, x, y };
  }

  return (
    <div className="map" ref={wrapper}>
      <canvas
        ref={canvas}
        style={{ width: size.w, height: size.h }}
        role="img"
        aria-label="Map of all products, positioned by their graph embedding"
        onMouseMove={(e) => setHover(nearest(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const hit = nearest(e);
          if (hit) onSelect(points.keys[hit.i]);
        }}
      />
      {hover && (
        <div className="map-tip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          Product {points.keys[hover.i]}
        </div>
      )}
      <ul className="map-key">
        <li>
          <span className="key-dot key-selected" /> selected
        </li>
        <li>
          <span className="key-dot key-similar" /> recommended
        </li>
        <li>
          <span className="key-dot key-bought" /> bought together
        </li>
      </ul>
    </div>
  );
}
