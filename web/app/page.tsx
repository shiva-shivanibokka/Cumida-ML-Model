import Hint from "./Hint.tsx";
import Readout from "./Readout.tsx";
import Tabs from "./Tabs.tsx";
import type { Model } from "../lib/model.ts";
import biopsyData from "../public/data/biopsies.json";
import modelData from "../public/data/models.json";
import summary from "../public/data/summary.json";

const { metrics, honesty, weights, dataset } = summary;
const interval = honesty.held_out_interval;
const split = honesty.split_comparison;
const ceiling = honesty.single_probe_ceiling;
const oldLeak = honesty.stratified_split_leak;
const agree = summary.agreement;

const f4 = (x: number) => x.toFixed(4);

/** Position on a 0.85–1.0 axis, which is where every score on this task lives. */
const AXIS_LOW = 0.85;
const place = (x: number) => `${((x - AXIS_LOW) / (1 - AXIS_LOW)) * 100}%`;

const TABS = ["Readout", "What it's worth", "The split", "Genes", "Method"];

export default function Page() {
  const models = modelData.models as unknown as Record<string, Model>;
  const rows = [
    { name: "Logistic Regression", m: metrics.logistic_regression },
    { name: "Gradient Boosting", m: metrics.gradient_boosting },
  ];
  const lrParams = metrics.logistic_regression.best_params;
  const gbParams = metrics.gradient_boosting.best_params;

  return (
    <main>
      <div className="shell">
        <section className="hero">
          <p className="eyebrow">CuMiDa · GSE14520 · {dataset.probes.toLocaleString()} probes</p>
          <h1>
            Identical scores, <em>different mistakes</em>
          </h1>
          <p className="lede">
            A liver biopsy is classified as hepatocellular carcinoma or normal tissue
            from its microarray gene expression. The two models below reach the{" "}
            <strong>same confusion matrix</strong> — same F1, same precision, same
            recall, to the last digit — while disagreeing about{" "}
            <strong>
              {agree.disagreements} of the {agree.n} biopsies
            </strong>
            . Each is wrong three times; they share only two of those mistakes. This
            page is as much about what a headline number hides as about the models.
          </p>
          <span className="runs-here">
            <span className="dot" />
            Both models run in this browser — there is no server to call
          </span>
        </section>

        <Tabs labels={TABS}>
          {/* ---------------------------------------------------- 1 · Readout */}
          <section>
            <h2>The readout</h2>
            <p className="sub">
              Pick one of the {dataset.samples - metrics.n_train} held-out biopsies, or draw
              one at random. Its genes are shown as deviations from the training mean, then
              both models are evaluated on it here, in the page.
            </p>
            <Readout
              models={models}
              winner={modelData.winner}
              genes={biopsyData.genes}
              biopsies={biopsyData.biopsies}
              stats={biopsyData.stats}
            />
          </section>

          {/* -------------------------------------------- 2 · What it's worth */}
          {/* A real element, not a fragment: Children.toArray flattens
              fragments, which would split this tab into two. */}
          <div>
            <section>
              <h2>What the headline number is worth</h2>
              <p className="sub">
                The test set is {interval.n_test} biopsies. Bootstrapping it{" "}
                {interval.draws.toLocaleString()} times puts a 95% interval of{" "}
                <strong>
                  [{f4(interval.ci_low)}, {f4(interval.ci_high)}]
                </strong>{" "}
                around that F1 — nearly {interval.ci_width.toFixed(2)} wide. One extra
                missed tumour moves the score further than the entire gap between the two
                models, which is why there is no winner to announce.
              </p>

              <div className="panel interval">
                <Hint>
                  The upright tick is the shipped model&rsquo;s F1. The band is where
                  that score would land if the {interval.n_test} test biopsies had been
                  a different {interval.n_test}, drawn the same way — resampled{" "}
                  {interval.draws.toLocaleString()} times. The two rings are the same
                  pipeline under each split strategy.
                </Hint>
                <div className="rowlabel">F1 on the held-out biopsies, 95% bootstrap interval</div>
                <div className="track">
                  <span className="rail" />
                  <span
                    className="band"
                    style={{
                      left: place(interval.ci_low),
                      right: `${100 - parseFloat(place(interval.ci_high))}%`,
                    }}
                  />
                  <span className="tick" style={{ left: place(interval.point) }} />
                  <span
                    className="other"
                    style={{ left: place(split.random.f1_mean) }}
                    title={`stratified split, mean of ${split.seeds}: ${f4(split.random.f1_mean)}`}
                  />
                  <span
                    className="other grouped"
                    style={{ left: place(split.grouped.f1_mean) }}
                    title={`patient-grouped split, mean of ${split.seeds}: ${f4(split.grouped.f1_mean)}`}
                  />
                </div>
                <div className="axis">
                  <span>{AXIS_LOW.toFixed(2)}</span>
                  <span>0.90</span>
                  <span>0.95</span>
                  <span>1.00</span>
                </div>
                <div className="legend">
                  <span>
                    <i className="point" /> shipped model, {f4(interval.point)}
                  </span>
                  <span>
                    <i /> stratified split, mean of {split.seeds}
                  </span>
                  <span>
                    <i className="grouped" /> patient-grouped split, mean of {split.seeds}
                  </span>
                </div>
                <p className="note">
                  Every model on this page, and the single-probe baseline on this tab,
                  falls inside that band. A comparison this test set cannot resolve is not
                  a result.
                </p>
              </div>

              <div className="panel" style={{ marginTop: "1.1rem" }}>
                <Hint>
                  Every metric each model scores on the held-out biopsies. CV F1 is
                  the one column that never saw them — it comes from the
                  cross-validation folds inside training.
                </Hint>
                <div className="rowlabel">the two models, side by side</div>
                <div className="scroller">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Test F1</th>
                      <th>ROC-AUC</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>CV F1</th>
                      <th>Genes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ name, m }) => (
                      <tr key={name} className={name === metrics.winner ? "win" : undefined}>
                        <td>
                          {name}
                          {name === metrics.winner ? " ·" : ""}
                        </td>
                        <td className="num">{f4(m.f1)}</td>
                        <td className="num">{f4(m.roc_auc)}</td>
                        <td className="num">{f4(m.precision)}</td>
                        <td className="num">{f4(m.recall)}</td>
                        <td className="num">{f4(m.cv_f1)}</td>
                        <td className="num">{m.n_genes}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                </div>
              </div>
              <p className="note">
                F1, precision and recall agree to the last digit — but not because the
                models agree. They are each wrong on three biopsies and share only two of
                them, so the counts coincide while the behaviour differs. They are
                separated by ROC-AUC, which reads the ranking rather than the verdicts, and{" "}
                <strong>cross-validation ranks them the other way round</strong>. The
                shipped model is Gradient Boosting on the AUC tie-break; that is a stated
                rule, not a finding.
              </p>

              <div className="panel" style={{ marginTop: "1.1rem" }}>
                <Hint>
                  The biopsies where the two models return different verdicts. Each
                  figure is that model&rsquo;s probability of carcinoma; over 50% is a
                  call of HCC.
                </Hint>
                <div className="rowlabel">
                  the {agree.disagreements} biopsies they disagree about
                </div>
                <div className="scroller">
                  <table>
                    <thead>
                      <tr>
                        <th>Biopsy</th>
                        <th>Confirmed</th>
                        {rows.map(({ name }) => (
                          <th key={name}>{name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {agree.disputed.map((d) => (
                        <tr key={d.id}>
                          <td className="num">{d.id}</td>
                          <td>{d.truth}</td>
                          {rows.map(({ name }) => {
                            const p = (d as unknown as Record<string, number>)[name];
                            return (
                              <td
                                className="num"
                                key={name}
                                style={{ color: p >= 0.5 ? "var(--hot)" : "var(--cold)" }}
                              >
                                {(p * 100).toFixed(1)}% {p >= 0.5 ? "HCC" : "normal"}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="note">
                  Both are normal tissue that one model calls cancer. Note the{" "}
                  {(agree.disputed[0]["Logistic Regression"] * 100).toFixed(1)}% and{" "}
                  {(agree.disputed[1]["Logistic Regression"] * 100).toFixed(1)}% — a
                  verdict that lands this close to the 50% threshold is a coin-flip dressed
                  as a decision.
                </p>
              </div>
            </section>

            <section>
              <h2>How hard is the task?</h2>
              <p className="sub">
                Before crediting any model, it is worth knowing where the floor is. Scoring
                each of the {ceiling.n_probes.toLocaleString()} probes on its own — one
                number, a threshold, no training at all — gets most of the way there.
              </p>
              <div className="stats">
                <div className="stat">
                  <Hint>
                    Take the single most informative probe, ignore the other{" "}
                    {(ceiling.n_probes - 1).toLocaleString()}, and threshold it. No
                    training, no combining. This is the floor a model has to beat.
                  </Hint>
                  <div className="n">{f4(ceiling.best_auc)}</div>
                  <div className="l">
                    AUC of the single best probe, <code>{ceiling.best_probe}</code>, used alone
                  </div>
                </div>
                <div className="stat">
                  <Hint>
                    How many probes reach that standard on their own. A count this
                    high means the tumour signal is spread across the whole array,
                    not hidden somewhere a model had to find it.
                  </Hint>
                  <div className="n">{ceiling.probes_over_95}</div>
                  <div className="l">probes that clear 0.95 AUC on their own</div>
                </div>
                <div className="stat">
                  <Hint>
                    The shipped model, tuned on {metrics.gradient_boosting.n_genes}{" "}
                    genes — here so the two figures beside it have something to be
                    compared against.
                  </Hint>
                  <div className="n">{f4(metrics.gradient_boosting.roc_auc)}</div>
                  <div className="l">AUC of the tuned 20-gene model</div>
                </div>
              </div>
              <p className="note">
                Tumour and adjacent normal liver differ enormously in expression, so a
                curated benchmark built from them is close to solved before the modelling
                starts. The work worth showing here is the <strong>methodology</strong> —
                the split, the leakage control, the error bars — not the fourth decimal
                place.
              </p>
            </section>
          </div>

          {/* -------------------------------------------------- 3 · The split */}
          <section>
            <h2>The split is grouped by patient</h2>
            <p className="sub">
              GSE14520 is a paired study: for {metrics.split.n_patients_paired} of its{" "}
              {metrics.split.n_patients_total} patients it holds both a tumour biopsy and a
              matched non-tumour biopsy from the same liver. Splitting on the label alone
              puts one in training and the other in test.
            </p>
            <div className="stats">
              <div className="stat">
                <Hint>
                  Under the split this project started with, this many test biopsies
                  came from a patient whose <strong>other</strong> biopsy was in
                  training. The model had effectively met that liver already.
                </Hint>
                <div className="n" style={{ color: "var(--hot)" }}>
                  {oldLeak.leaked} / {oldLeak.of}
                </div>
                <div className="l">
                  held-out biopsies that shared a patient with training, under the old
                  stratified split
                </div>
              </div>
              <div className="stat">
                <Hint>
                  Under the split actually used. Grouping keeps every biopsy from one
                  patient on the same side, so a held-out liver is one the model has
                  never seen in any form.
                </Hint>
                <div className="n" style={{ color: "var(--ok)" }}>
                  0
                </div>
                <div className="l">
                  patients on both sides of the split this model was trained on
                </div>
              </div>
              <div className="stat">
                <Hint>
                  What closing the leak cost, in F1, averaged over {split.seeds}{" "}
                  different splits. Negative means the honest split scores lower —
                  the expected direction, and the reason for measuring rather than
                  asserting it.
                </Hint>
                <div className="n">{split.cost_of_grouping_f1.toFixed(3)}</div>
                <div className="l">
                  F1 the grouping costs, averaged over {split.seeds} splits
                </div>
              </div>
            </div>
            <p className="note">
              Honest accounting: the leak was <strong>real but cheap</strong>. Removing it
              costs {Math.abs(split.cost_of_grouping_f1).toFixed(3)} F1, which is inside the
              seed-to-seed spread (±{split.grouped.f1_std.toFixed(3)}) — the tumour signal
              is large enough that knowing the patient adds little. It is fixed anyway,
              because &ldquo;biopsies the model has never seen&rdquo; has to be true rather
              than nearly true. Note that the spread <em>widens</em> once the leak is gone.
            </p>

            <div className="panel" style={{ marginTop: "1.1rem" }}>
              <Hint>
                The same pipeline run both ways — same features, same hyperparameter
                search, same {split.seeds} seeds. Only the splitter changes, so the
                gap between these rows is the leak and nothing else.
              </Hint>
              <div className="rowlabel">the same pipeline under both split strategies</div>
              <div className="scroller">
                <table>
                  <thead>
                    <tr>
                      <th>Split</th>
                      <th>F1, mean of {split.seeds}</th>
                      <th>F1 spread</th>
                      <th>AUC, mean</th>
                      <th>Patients on both sides</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Stratified on the label</td>
                      <td className="num">{f4(split.random.f1_mean)}</td>
                      <td className="num">±{split.random.f1_std.toFixed(3)}</td>
                      <td className="num">{f4(split.random.auc_mean)}</td>
                      <td className="num" style={{ color: "var(--hot)" }}>
                        {(split.random.patient_overlap * 100).toFixed(1)}%
                      </td>
                    </tr>
                    <tr className="win">
                      <td>Grouped by patient ·</td>
                      <td className="num">{f4(split.grouped.f1_mean)}</td>
                      <td className="num">±{split.grouped.f1_std.toFixed(3)}</td>
                      <td className="num">{f4(split.grouped.auc_mean)}</td>
                      <td className="num" style={{ color: "var(--ok)" }}>
                        {(split.grouped.patient_overlap * 100).toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="note">
                Same features, same hyperparameter search, same {split.seeds} seeds — only
                the splitter changes. Stating the cost is the point: a leak you have
                measured is a known quantity, and one you have only asserted is not.
              </p>
            </div>
          </section>

          {/* ------------------------------------------------------ 4 · Genes */}
          <section>
            <h2>What the models look at</h2>
            <p className="sub">
              Recursive feature elimination picked the genes; the fitted models then decide
              how much to use them. The L1 penalty sets some of the selected coefficients
              to exactly zero, so a gene can be chosen and still carry no weight.
            </p>
            {Object.entries(weights).map(([name, w]) => {
              const max = Math.max(...w.genes.map((g) => Math.abs(g.weight)), 1e-9);
              const ordered = [...w.genes].sort(
                (a, b) => Math.abs(b.weight) - Math.abs(a.weight),
              );
              return (
                <div className="panel" key={name} style={{ marginBottom: ".9rem" }}>
                  <Hint>
                    {w.signed
                      ? `Bar length is how far a gene moves the decision, direction is
                         which way. A dash means the L1 penalty zeroed it — the gene
                         was selected, then ignored.`
                      : `How much each gene reduced impurity across all the trees.
                         Always positive, so it says how much a gene matters but never
                         which way it points.`}
                  </Hint>
                  <div className="rowlabel">
                    {name} ·{" "}
                    {w.signed
                      ? "signed coefficients (positive = toward carcinoma)"
                      : "impurity importances"}
                  </div>
                  <div className="bars">
                    {ordered.map((g) => {
                      const width = (Math.abs(g.weight) / max) * (w.signed ? 50 : 100);
                      const positive = g.weight >= 0;
                      return (
                        <div className={`bar${g.weight === 0 ? " silent" : ""}`} key={g.gene}>
                          <span className="g">{g.gene}</span>
                          <span className="t">
                            <i
                              style={{
                                left: w.signed ? (positive ? "50%" : `${50 - width}%`) : "0%",
                                width: `${width}%`,
                                background: !w.signed
                                  ? "var(--accent)"
                                  : positive
                                    ? "var(--hot)"
                                    : "var(--cold)",
                              }}
                            />
                          </span>
                          <span className="v">
                            {g.weight === 0 ? "—" : g.weight.toFixed(3)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
            <p className="note">
              {(() => {
                const lr = weights["Logistic Regression"];
                const zero = lr.genes.filter((g) => g.weight === 0).length;
                return zero > 0
                  ? `Logistic Regression was handed ${lr.genes.length} genes and zeroed ${zero} of them
                     — it reaches its score on ${lr.genes.length - zero}.`
                  : `Every gene handed to the logistic model carries a non-zero weight.`;
              })()}{" "}
              The best single probe from the previous tab, <code>{ceiling.best_probe}</code>,
              is among the genes selected here.
            </p>
          </section>

          {/* ----------------------------------------------------- 5 · Method */}
          <section className="prose">
            <h2>How this was built</h2>
            <p className="sub">
              Every number on this page is produced by a script in the repository and
              committed as JSON; CI fails if the page and the pipeline disagree. The order
              below is the order that matters — the split is decided before a single label
              is looked at.
            </p>

            <ol className="steps">
              <li>
                <span>
                  <strong>Recover the patients.</strong> CuMiDa ships biopsies, not people.
                  The two GEO series matrices are fetched and their source names parsed —
                  a patient identifier with an <code>A</code> or <code>B</code> suffix for
                  tumour and non-tumour — which recovers{" "}
                  {metrics.split.n_patients_total} patients behind {dataset.samples}{" "}
                  biopsies, {metrics.split.n_patients_paired} of them contributing a
                  matched pair. The build refuses to continue if a single biopsy is left
                  unmapped.
                </span>
              </li>
              <li>
                <span>
                  <strong>Reduce the probes without looking at labels.</strong>{" "}
                  {dataset.probes.toLocaleString()} probes come down to{" "}
                  {metrics.n_probes_after_label_free_reduction.toLocaleString()} before any
                  outcome is consulted, so the reduction cannot leak.
                </span>
              </li>
              <li>
                <span>
                  <strong>Split on patients, not biopsies.</strong>{" "}
                  <code>{metrics.split.strategy}</code> — {metrics.n_train} biopsies from{" "}
                  {metrics.split.n_patients_train} patients to train,{" "}
                  {metrics.n_test} from {metrics.split.n_patients_test} to test, with{" "}
                  {metrics.split.patients_on_both_sides} patients on both sides. Training
                  aborts if that count is ever non-zero.
                </span>
              </li>
              <li>
                <span>
                  <strong>Select and tune inside the cross-validation.</strong> Recursive
                  feature elimination and the hyperparameter grid both sit inside the CV
                  folds, not outside them, so the test biopsies play no part in choosing
                  either.
                </span>
              </li>
              <li>
                <span>
                  <strong>Ship the parameters, not a server.</strong> The fitted
                  coefficients, the scaler, and every tree are exported to JSON and
                  re-implemented in TypeScript, so the classifier runs in the reader&rsquo;s
                  browser. A golden-fixture check re-predicts all {interval.n_test}{" "}
                  held-out biopsies with the exact module this page ships and compares
                  against scikit-learn; the build fails on any disagreement past the
                  twelfth decimal.
                </span>
              </li>
            </ol>

            <div className="panel" style={{ marginTop: "1.4rem" }}>
              <Hint>
                The values the grid search chose inside the cross-validation folds.
                The held-out biopsies played no part in picking them.
              </Hint>
              <div className="rowlabel">what the search settled on</div>
              <div className="scroller">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Genes kept</th>
                      <th>Hyperparameters</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Logistic Regression</td>
                      <td className="num">{lrParams.rfe__n_features_to_select}</td>
                      <td className="num">
                        {lrParams.clf__penalty} penalty · C = {lrParams.clf__C}
                      </td>
                    </tr>
                    <tr className="win">
                      <td>Gradient Boosting ·</td>
                      <td className="num">{gbParams.rfe__n_features_to_select}</td>
                      <td className="num">
                        {gbParams.clf__n_estimators} trees · depth {gbParams.clf__max_depth} ·
                        lr {gbParams.clf__learning_rate.toFixed(4)} · subsample{" "}
                        {gbParams.clf__subsample}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel" style={{ marginTop: ".9rem" }}>
              <Hint>
                Counts on the held-out biopsies: tumours called correctly, tumours
                missed, healthy tissue called cancer, healthy tissue called correctly.
                Every metric on this page is arithmetic on these four numbers.
              </Hint>
              <div className="rowlabel">
                the confusion matrix both models arrive at, on {metrics.n_test} held-out biopsies
              </div>
              <div className="scroller">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>True positive</th>
                      <th>False negative</th>
                      <th>False positive</th>
                      <th>True negative</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ name, m }) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="num">{m.confusion.tp}</td>
                        <td className="num">{m.confusion.fn}</td>
                        <td className="num">{m.confusion.fp}</td>
                        <td className="num">{m.confusion.tn}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note">
                Four identical counts, and the tab before this one shows they are not the
                same four biopsies. A confusion matrix says how often a model is wrong; it
                does not say when.
              </p>
            </div>
          </section>
        </Tabs>

        <div className="foot">
          <p>
            Data: <strong>{dataset.name}</strong> — {dataset.source}. {dataset.samples}{" "}
            biopsies from {dataset.patients} patients, {dataset.probes.toLocaleString()}{" "}
            Affymetrix probes. CuMiDa is the curated redistribution; the underlying study
            is GEO GSE14520.
          </p>
          <p>Educational project, not a clinical tool.</p>
          <p className="signoff">
            Built by Shivani Bokka ·{" "}
            <a href="https://github.com/shiva-shivanibokka/Cumida-ML-Model">
              source on GitHub
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}
