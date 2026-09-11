import type { Metadata } from "next";
import "./globals.css";
import Logo from "./Logo.tsx";
import Tissue from "./Tissue.tsx";

export const metadata: Metadata = {
  title: "Liver HCC Classifier — gene-expression readout",
  description:
    "Classifying liver biopsies as hepatocellular carcinoma or normal tissue from "
    + "microarray gene expression — with the error bars the headline number needs.",
};

/** The mark again, flattened into the tab icon. */
const FAVICON =
  "data:image/svg+xml,"
  + "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 46 32'>"
  + "<circle cx='17' cy='16' r='12' fill='%23C2566B' fill-opacity='.35'/>"
  + "<circle cx='29' cy='16' r='12' fill='%234E2C7A' fill-opacity='.35'/>"
  + "<circle cx='17' cy='16' r='12' fill='none' stroke='%23C2566B' stroke-width='2.4'/>"
  + "<circle cx='29' cy='16' r='12' fill='none' stroke='%234E2C7A' stroke-width='2.4'/>"
  + "<circle cx='11.5' cy='13' r='3' fill='%234E2C7A'/>"
  + "<circle cx='34.5' cy='19' r='3' fill='%234E2C7A'/></svg>";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href={
            "https://fonts.googleapis.com/css2"
            + "?family=Fraunces:ital,opsz,wght@0,9..144,400..600;1,9..144,400..600"
            + "&family=IBM+Plex+Mono:wght@400;500"
            + "&display=swap"
          }
        />
        <link rel="icon" href={FAVICON} />
      </head>
      <body>
        <Tissue />
        <header className="masthead">
          <div className="shell">
            <span className="wordmark">
              <Logo />
              <strong>Liver HCC Classifier</strong>
            </span>
            <a
              className="byline"
              href="https://github.com/shiva-shivanibokka/Cumida-ML-Model"
            >
              Source
            </a>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
