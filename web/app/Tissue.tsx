"use client";

import { useEffect, useRef } from "react";

/**
 * The page sits on a field of hepatocytes.
 *
 * Two things matter to make it read as tissue rather than as scattered dots.
 * The cell radius is larger than half the lattice pitch, so neighbours overlap
 * and the field is continuous edge to edge; and the fills are translucent, so
 * the overlaps darken and give the field its mottled grain. The ink is kept
 * deliberately faint — this is a ground for the page, not a picture of its own.
 */

const PITCH = 34;
const SEED = 20250911;
const SPEED = 0.006; // px per ms, drifting upward

interface Cell {
  x: number;
  y: number;
  rr: number;
  squash: number;
  rot: number;
  nx: number;
  ny: number;
  nr: number;
  drift: number;
  tone: number;
  bi: boolean;
}

/** Deterministic noise, so the field is identical on every load and resize. */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
}

function makeCells(w: number, h: number): Cell[] {
  const r = lcg(SEED);
  const cells: Cell[] = [];
  for (let gy = -2; gy * PITCH < h + PITCH * 4; gy++) {
    for (let gx = -2; gx * PITCH < w + PITCH * 3; gx++) {
      cells.push({
        x: gx * PITCH + (r() - 0.5) * PITCH * 0.55,
        y: gy * PITCH + (r() - 0.5) * PITCH * 0.55 + PITCH * 2,
        rr: 22 + r() * 13,
        squash: 0.78 + r() * 0.4,
        rot: r() * Math.PI,
        nx: (r() - 0.5) * 0.38,
        ny: (r() - 0.5) * 0.38,
        nr: 4.6 + r() * 3.4,
        drift: r() * Math.PI * 2,
        tone: r(),
        bi: r() < 0.1,
      });
    }
  }
  return cells;
}

export default function Tissue() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let cells: Cell[] = [];
    let w = 0;
    let h = 0;
    let span = 0;
    let raf = 0;

    function fit() {
      const c = canvas!;
      const g = ctx!;
      w = innerWidth;
      h = innerHeight;
      const dpr = devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = h * dpr;
      c.style.width = `${w}px`;
      c.style.height = `${h}px`;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      span = h + PITCH * 6;
      cells = makeCells(w, h);
    }

    function frame(t: number) {
      const g = ctx!;
      g.clearRect(0, 0, w, h);
      const dy = t * SPEED;
      for (const o of cells) {
        // Subtracting the drift walks each cell up the page; the modulo sends
        // it back in at the bottom once it has left the top.
        let cy = (o.y - dy) % span;
        if (cy < 0) cy += span;
        cy -= 68;
        if (cy < -80 || cy > h + 80) continue;

        g.save();
        g.translate(o.x + Math.sin(t / 3400 + o.drift) * 2.4, cy);
        g.rotate(o.rot + Math.sin(t / 6000 + o.drift) * 0.05);

        g.beginPath();
        g.ellipse(0, 0, o.rr, o.rr * o.squash, 0, 0, 7);
        g.fillStyle = `rgba(216,124,140,${0.03 + o.tone * 0.028})`;
        g.fill();
        g.strokeStyle = `rgba(172,80,100,${0.036 + o.tone * 0.03})`;
        g.lineWidth = 1;
        g.stroke();

        // Roughly a tenth of hepatocytes are binucleate.
        g.beginPath();
        g.ellipse(o.rr * o.nx, o.rr * o.ny, o.nr, o.nr * 0.9, 0, 0, 7);
        g.fillStyle = `rgba(78,44,122,${0.066 + o.tone * 0.05})`;
        g.fill();
        if (o.bi) {
          g.beginPath();
          g.ellipse(-o.rr * o.nx, -o.rr * o.ny, o.nr * 0.85, o.nr * 0.78, 0, 0, 7);
          g.fillStyle = `rgba(78,44,122,${0.056 + o.tone * 0.04})`;
          g.fill();
        }
        g.restore();
      }
      if (!still) raf = requestAnimationFrame(frame);
    }

    fit();
    // Paint one frame synchronously. requestAnimationFrame does not fire in a
    // background tab, so scheduling the first frame through it leaves the page
    // blank until the tab is focused — and blank is what a broken canvas looks
    // like. frame() schedules its own continuation.
    frame(0);

    const onResize = () => {
      fit();
      frame(performance.now());
    };
    addEventListener("resize", onResize);
    return () => {
      removeEventListener("resize", onResize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return <canvas id="tissue" ref={ref} aria-hidden="true" />;
}
