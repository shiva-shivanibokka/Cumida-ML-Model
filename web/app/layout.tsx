import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Liver HCC Classifier — gene-expression readout",
  description:
    "Classifying liver biopsies as hepatocellular carcinoma or normal tissue from "
    + "microarray gene expression — with the error bars the headline number needs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
        />
        <link
          rel="icon"
          href={
            "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            + "<text y='.9em' font-size='88'>%F0%9F%A7%AC</text></svg>"
          }
        />
      </head>
      <body>
        <header className="masthead">
          <div className="shell">
            <span className="wordmark">
              <strong>Liver HCC Classifier</strong>
            </span>
            <span className="byline">
              Built by Shivani Bokka ·{" "}
              <a href="https://github.com/shiva-shivanibokka/Cumida-ML-Model">Source</a>
            </span>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
