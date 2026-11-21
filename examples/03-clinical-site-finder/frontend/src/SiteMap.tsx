import { useState } from "react";
import { AREAS, type MapPoint } from "./api";

type Props = {
  points: MapPoint[];
  selected: string;
  experienced: Set<string>;
  lookalikes: Set<string>;
  onSelect: (key: string) => void;
};

const W = 400;
const H = 300;
const PAD = 14;

/** Every site as a dot, positioned by the 2D projection of its embedding and
 *  coloured by primary therapeutic area. A few hundred points: plain SVG. */
export default function SiteMap({ points, selected, experienced, lookalikes, onSelect }: Props) {
  const [hover, setHover] = useState<MapPoint | null>(null);
  const x = (p: MapPoint) => PAD + p.x * (W - 2 * PAD);
  const y = (p: MapPoint) => PAD + (1 - p.y) * (H - 2 * PAD);
  const highlighted = points.filter(
    (p) => experienced.has(p.key) || lookalikes.has(p.key) || p.key === selected,
  );

  return (
    <figure className="sitemap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Map of all sites by embedding">
        {points.map((p) => (
          <circle
            key={p.key}
            cx={x(p)}
            cy={y(p)}
            r={2.6}
            fill={AREAS[p.area]?.color ?? "#8a94a3"}
            opacity={0.55}
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onSelect(p.key)}
          />
        ))}
        {highlighted.map((p) => (
          <circle
            key={`h-${p.key}`}
            cx={x(p)}
            cy={y(p)}
            r={p.key === selected ? 7 : 5}
            className={
              p.key === selected
                ? "ring-selected"
                : lookalikes.has(p.key)
                  ? "ring-lookalike"
                  : "ring-experienced"
            }
            pointerEvents="none"
          />
        ))}
      </svg>
      <figcaption>
        {hover ? (
          `${hover.name}, ${hover.city}`
        ) : (
          <>
            <span className="key experienced" /> experienced
            <span className="key lookalike" /> lookalike
            <span className="key selected" /> selected
          </>
        )}
      </figcaption>
    </figure>
  );
}
