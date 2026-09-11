"use client";

import { useMemo, useState } from "react";

import { predict, type Model } from "../lib/model.ts";

interface Biopsy {
  id: string;
  patient: string;
  tissue: string;
  label: string;
  values: Record<string, number>;
}

interface Props {
  models: Record<string, Model>;
  winner: string;
  genes: string[];
  biopsies: Biopsy[];
  stats: Record<string, { mean: number; std: number }>;
}

const HCC = "HCC";

/** Divergent expression colour: cool below the training mean, warm above. */
function colour(z: number): string {
  const clamped = Math.max(-2, Math.min(2, z));
  const t = (clamped + 2) / 4;
  const cold = [47, 127, 168];
  const mid = [190, 188, 178];
  const hot = [194, 69, 60];
  const [a, b, u] = t < 0.5 ? [cold, mid, t * 2] : [mid, hot, (t - 0.5) * 2];
  const mix = a.map((v, i) => Math.round(v + (b[i] - v) * u));
  return `rgb(${mix.join(",")})`;
}

export default function Readout({ models, winner, genes, biopsies, stats }: Props) {
  // Two labelled biopsies to start with, so the page says something before it
  // is touched: one carcinoma, one normal.
  const opening = useMemo(() => {
    const hcc = biopsies.findIndex((b) => b.label === HCC);
    return hcc >= 0 ? hcc : 0;
  }, [biopsies]);
  const [picked, setPicked] = useState(opening);
  const [drawn, setDrawn] = useState(false);

  const biopsy = biopsies[picked];
  const names = Object.keys(models);

  const calls = names.map((name) => {
    const p = predict(models[name], biopsy.values);
    return { name, p, says: p >= 0.5 ? HCC : "normal" };
  });
  const unanimous = new Set(calls.map((c) => c.says)).size === 1;

  // Counted here rather than written into the copy: the page's whole point is
  // that a number stated in prose drifts away from the thing it describes.
  const agreed = useMemo(() => {
    if (names.length < 2) return biopsies.length;
    return biopsies.filter((b) => {
      const verdicts = names.map((n) => predict(models[n], b.values) >= 0.5);
      return new Set(verdicts).size === 1;
    }).length;
  }, [models, biopsies, names]);

  const shortlist = useMemo(() => {
    const hcc = biopsies.filter((b) => b.label === HCC).slice(0, 3);
    const normal = biopsies.filter((b) => b.label !== HCC).slice(0, 3);
    return [...hcc, ...normal].map((b) => biopsies.indexOf(b));
  }, [biopsies]);

  return (
    <div className="readout">
      <div className="picker">
        {shortlist.map((i) => (
          <button
            key={biopsies[i].id}
            className="chip"
            aria-pressed={picked === i && !drawn}
            onClick={() => { setPicked(i); setDrawn(false); }}
          >
            <span className={`tag ${biopsies[i].label === HCC ? "hcc" : "normal"}`} />
            {biopsies[i].id}
          </button>
        ))}
        <button
          className="chip shuffle"
          aria-pressed={drawn}
          onClick={() => {
            setPicked(Math.floor(Math.random() * biopsies.length));
            setDrawn(true);
          }}
        >
          ⟳ draw a random biopsy
        </button>
      </div>

      <div className="panel">
        <div className="rowlabel">
          expression · biopsy {biopsy.id} · patient {biopsy.patient} ·{" "}
          {biopsy.tissue === "A" ? "tumour site" : "adjacent non-tumour site"}
        </div>
        <div className="heat">
          {genes.map((gene) => {
            const stat = stats[gene] ?? { mean: 0, std: 1 };
            const z = (biopsy.values[gene] - stat.mean) / (stat.std || 1);
            return (
              <div className="gene" key={gene}>
                <div
                  className="swatch"
                  style={{ background: colour(z) }}
                  title={
                    `${gene}: ${biopsy.values[gene].toFixed(2)} `
                    + `(${z >= 0 ? "+" : ""}${z.toFixed(2)} SD vs training mean)`
                  }
                >
                  {z >= 0 ? "+" : ""}{z.toFixed(1)}
                </div>
                <div className="genename" title={gene}>{gene.replace(/_at$/, "")}</div>
              </div>
            );
          })}
        </div>
        <div className="scale">
          <span>−2 SD</span>
          <span className="scalebar" />
          <span>+2 SD</span>
          <span style={{ marginLeft: "auto" }}>
            every value is relative to the training mean for that probe
          </span>
        </div>
      </div>

      <div className="verdicts">
        {calls.map(({ name, p, says }) => (
          <div className="verdict" key={name}>
            <div className="who">
              <span>{name}</span>
              {name === winner && <span className="crown">shipped model</span>}
            </div>
            <div className={`call ${says === HCC ? "hcc" : "normal"}`}>
              {says === HCC ? "Hepatocellular carcinoma" : "Normal liver tissue"}
            </div>
            <div className="against">
              {says === biopsy.label
                ? <span className="ok">agrees with the confirmed diagnosis</span>
                : <span className="no">confirmed diagnosis was {biopsy.label}</span>}
            </div>
            <div className="meter"><i style={{ width: `${p * 100}%` }} /></div>
            <div className="pval">P(HCC) = {(p * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <p className="agree">
        {unanimous
          ? `Both models agree here — as they do on ${agreed} of the ${biopsies.length}
             held-out biopsies. Their scores are identical, but that is not the
             same as agreeing.`
          : `This is one of the two biopsies the models disagree about. Their
             confusion matrices are identical; the mistakes behind them are not.`}
      </p>
    </div>
  );
}
