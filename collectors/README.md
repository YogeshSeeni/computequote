# collectors (placeholder)

Price collectors are not part of the alpha. Each collector will implement
`fetch() -> list[PriceObservation]` over a provider API or pricing page and run
nightly from GitHub Actions into Postgres. Until then the engine uses the
curated table in `engine/prices.py`. See the backlog for the tracking issue.
