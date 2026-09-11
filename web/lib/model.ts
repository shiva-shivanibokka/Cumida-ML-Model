/**
 * The trained models, running in the browser.
 *
 * Nothing here is a re-implementation or an approximation. `scripts/export_web_artifacts.py`
 * writes out the fitted parameters -- the scaler's mean and scale, the logistic
 * coefficients, every regression tree in the ensemble -- and these two functions
 * evaluate them the way scikit-learn does. `npm run check:golden` re-predicts all
 * 72 held-out biopsies with this exact module and compares against scikit-learn's
 * own answers, so a drift here fails the build rather than shipping a confident
 * wrong verdict.
 *
 * The gene order comes from the exported `genes` array and never from iterating
 * an object's keys: a model's columns are ordered by how it was fitted, and JSON
 * key order is not that order.
 */

export interface LogisticModel {
  kind: "logistic";
  genes: string[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
}

export interface Tree {
  feature: number[];
  threshold: number[];
  left: number[];
  right: number[];
  value: number[];
}

export interface BoostingModel {
  kind: "boosting";
  genes: string[];
  mean: number[];
  scale: number[];
  init: number;
  learning_rate: number;
  trees: Tree[];
}

export type Model = LogisticModel | BoostingModel;

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/** Standardise one biopsy into the column order the model was fitted on. */
function standardise(model: Model, values: Record<string, number>): number[] {
  return model.genes.map((gene, i) => {
    const raw = values[gene];
    if (raw === undefined) throw new Error(`biopsy is missing gene ${gene}`);
    return (raw - model.mean[i]) / model.scale[i];
  });
}

/**
 * Walk one regression tree. A leaf is marked by a left child of -1, which is
 * how scikit-learn's `children_left` encodes it; the split is `<=`, matching
 * `sklearn.tree`'s own convention rather than the more usual `<`.
 */
function runTree(tree: Tree, x: number[]): number {
  let node = 0;
  while (tree.left[node] !== -1) {
    node = x[tree.feature[node]] <= tree.threshold[node]
      ? tree.left[node]
      : tree.right[node];
  }
  return tree.value[node];
}

/** P(HCC) for one biopsy. */
export function predict(model: Model, values: Record<string, number>): number {
  const x = standardise(model, values);

  if (model.kind === "logistic") {
    let z = model.intercept;
    for (let i = 0; i < model.coef.length; i += 1) z += model.coef[i] * x[i];
    return sigmoid(z);
  }

  let z = model.init;
  for (const tree of model.trees) z += model.learning_rate * runTree(tree, x);
  return sigmoid(z);
}

/** The genes a model can actually move its answer with. */
export function activeGenes(model: Model): Set<string> {
  if (model.kind === "logistic") {
    return new Set(model.genes.filter((_, i) => model.coef[i] !== 0));
  }
  const used = new Set<number>();
  for (const tree of model.trees) {
    tree.feature.forEach((f, node) => {
      if (tree.left[node] !== -1) used.add(f);
    });
  }
  return new Set(model.genes.filter((_, i) => used.has(i)));
}
