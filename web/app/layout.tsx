import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { API_BASE } from "@/lib/api";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "ComputeQuote",
  description:
    "Predicts memory feasibility, parallelism, step time and dollar cost for an AI training run.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${plexSans.variable} ${plexMono.variable} min-h-screen flex flex-col`}
      >
        <header className="border-b border-line bg-surface">
          <div className="mx-auto flex max-w-[1100px] flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-4">
            <span className="text-lg font-semibold tracking-tight">
              ComputeQuote
            </span>
            <span className="rounded border border-line px-1.5 py-0.5 font-mono text-[11px] uppercase tracking-wider text-ink-2">
              alpha
            </span>
            <p className="basis-full text-sm text-ink-2 sm:basis-auto">
              Memory feasibility, parallelism, step time and dollars for a
              training run.
            </p>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1100px] flex-1 px-4 py-6 sm:py-8">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-[1100px] flex-wrap items-center justify-between gap-2 px-4 py-3 text-xs text-ink-3">
            <span>
              Estimates are modelled, not measured. See{" "}
              <code className="font-mono">docs/MODELING.md</code>.
            </span>
            <span className="font-mono">
              API{" "}
              <a
                className="underline decoration-line underline-offset-2 hover:text-ink"
                href={`${API_BASE}/docs`}
                target="_blank"
                rel="noreferrer"
              >
                {API_BASE}
              </a>
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
