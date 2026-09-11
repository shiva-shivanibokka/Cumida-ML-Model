/**
 * The "?" beside a panel: what this box is, in one sentence.
 *
 * No JavaScript involved. The trigger is a real button, so the tooltip opens
 * on hover, on keyboard focus, and on tap — `:focus-within` covers the last
 * two, which a hover-only tooltip would leave out.
 */
export default function Hint({ children }: { children: React.ReactNode }) {
  return (
    <span className="hint">
      <button type="button" className="hint-mark" aria-label="What is this?">
        ?
      </button>
      <span className="hint-body" role="tooltip">
        {children}
      </span>
    </span>
  );
}
