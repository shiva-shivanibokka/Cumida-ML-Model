/**
 * Re-predict every held-out biopsy with the module the page ships, and compare
 * against scikit-learn's own answers.
 *
 * This is the check that makes "the model runs in your browser" a fact rather
 * than a hope. It imports `lib/model.ts` -- the same file the page imports, not
 * a copy -- so anything that changes the arithmetic, the gene ordering, or the
 * exported parameters shows up here as a number that no longer matches Python.
 *
 * Run with:  npm run check:golden
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { predict, type Model } from "../lib/model.ts";

const DATA = join(import.meta.dirname, "..", "public", "data");
const read = (name: string) => JSON.parse(readFileSync(join(DATA, name), "utf8"));

// Floating-point arithmetic is not associative, so summing 180 trees in
// JavaScript may land a few ulps from NumPy's summation of the same values.
// The exported parameters are rounded to 12 digits, which bounds the honest
// disagreement well below this; anything larger is a real defect.
const TOLERANCE = 1e-9;

const models: Record<string, Model> = read("models.json").models;
const golden: Record<string, number[]> = read("golden.json").probabilities;
const { biopsies } = read("biopsies.json");

let worst = 0;
let worstWhere = "";
let checked = 0;

for (const [name, model] of Object.entries(models)) {
  const expected = golden[name];
  if (!expected) throw new Error(`golden.json has no probabilities for ${name}`);
  if (expected.length !== biopsies.length) {
    throw new Error(
      `${name}: ${expected.length} golden values for ${biopsies.length} biopsies`,
    );
  }

  biopsies.forEach((biopsy: { id: string; values: Record<string, number> }, i: number) => {
    const got = predict(model, biopsy.values);
    const delta = Math.abs(got - expected[i]);
    checked += 1;
    if (delta > worst) {
      worst = delta;
      worstWhere = `${name} / ${biopsy.id}: TypeScript ${got} vs scikit-learn ${expected[i]}`;
    }
  });
}

console.log(`checked ${checked} predictions across ${Object.keys(models).length} models`);
console.log(`largest disagreement with scikit-learn: ${worst.toExponential(3)}`);

if (worst > TOLERANCE) {
  console.error(`FAIL  exceeds tolerance ${TOLERANCE.toExponential(0)}`);
  console.error(`      ${worstWhere}`);
  process.exit(1);
}
console.log("the browser reproduces scikit-learn.");
