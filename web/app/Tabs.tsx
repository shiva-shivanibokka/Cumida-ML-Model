"use client";

import { Children, useRef, useState } from "react";

interface Props {
  labels: string[];
  children: React.ReactNode;
}

/**
 * Five panels behind a tab bar, rather than one long scroll.
 *
 * Every panel is rendered and only the inactive ones are hidden, so the whole
 * page is still in the static export for anyone reading it without JavaScript
 * — and Ctrl-F still finds a number that happens to live on another tab.
 */
export default function Tabs({ labels, children }: Props) {
  const panels = Children.toArray(children);
  const [active, setActive] = useState(0);
  const bar = useRef<HTMLDivElement>(null);

  function move(to: number) {
    const next = (to + labels.length) % labels.length;
    setActive(next);
    const buttons = bar.current?.querySelectorAll("button");
    buttons?.[next]?.focus();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    const keys: Record<string, number> = {
      ArrowRight: active + 1,
      ArrowLeft: active - 1,
      Home: 0,
      End: labels.length - 1,
    };
    if (e.key in keys) {
      e.preventDefault();
      move(keys[e.key]);
    }
  }

  return (
    <>
      <div className="tabs" role="tablist" ref={bar} onKeyDown={onKeyDown}>
        {labels.map((label, i) => (
          <button
            key={label}
            type="button"
            role="tab"
            id={`tab-${i}`}
            aria-selected={i === active}
            aria-controls={`panel-${i}`}
            tabIndex={i === active ? 0 : -1}
            className="tab"
            onClick={() => setActive(i)}
          >
            {label}
          </button>
        ))}
      </div>
      {panels.map((panel, i) => (
        <div
          key={labels[i]}
          role="tabpanel"
          id={`panel-${i}`}
          aria-labelledby={`tab-${i}`}
          className="tabpanel"
          hidden={i !== active}
        >
          {panel}
        </div>
      ))}
    </>
  );
}
