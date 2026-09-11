/**
 * The mark: two cells, overlapping almost but not entirely.
 *
 * It is the page's argument in one shape — two models whose verdicts coincide
 * on nearly every biopsy and part company on a couple — drawn in the stains a
 * liver section is actually read under, haematoxylin violet and eosin pink.
 */
export default function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 46 32"
      height={size}
      width={(size * 46) / 32}
      role="img"
      aria-label="Two overlapping cells"
      focusable="false"
    >
      <defs>
        <clipPath id="lobe-a">
          <circle cx="17" cy="16" r="12" />
        </clipPath>
      </defs>
      <circle cx="17" cy="16" r="12" fill="#C2566B" fillOpacity=".14" />
      <circle cx="29" cy="16" r="12" fill="#4E2C7A" fillOpacity=".14" />
      {/* the shared verdicts: where the two cells actually overlap */}
      <g clipPath="url(#lobe-a)">
        <circle cx="29" cy="16" r="12" fill="#6B3F7E" fillOpacity=".26" />
      </g>
      <circle cx="17" cy="16" r="12" fill="none" stroke="#C2566B" strokeWidth="1.5" />
      <circle cx="29" cy="16" r="12" fill="none" stroke="#4E2C7A" strokeWidth="1.5" />
      <circle cx="11.5" cy="13" r="2.7" fill="#4E2C7A" />
      <circle cx="34.5" cy="19" r="2.7" fill="#4E2C7A" />
    </svg>
  );
}
