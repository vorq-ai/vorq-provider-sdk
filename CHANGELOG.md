# Changelog

All notable changes to `vorqd` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Settle bills the backend's own usage. The `settle` op carries and signs
  `prompt_tok`, `cached_tok`, `completion_tok` and `rate_in_cached`
  (`Settle(bytes32 jobId,uint32 promptTok,uint32 cachedTok,uint32 completionTok,uint128 rateInCached,uint64 issuedAt)`).
  Presets read `usage.prompt_tokens` and `usage.prompt_tokens_details.cached_tokens`
  (Responses: `input_tokens`, `input_tokens_details.cached_tokens`); raw text
  results may map `prompt_tokens` and `cached_tokens`. Media settles its
  declared `units_in` as the prompt count.
- The pre-claim input check for text and embedding now refuses an order whose
  `units_in` (the request's canonical byte length) is below its payload size
  less 640 bytes of envelope slack (was 16 bytes a unit plus 4096).

### Added

- Per-window `rate_in_cached` (USD string, at most `rate_in`; default: no discount).
- Each pushed ask quote carries `rate_in_cached`, unsigned beside the signed
  rates: the configured cached rate capped at the quote's `rate_in`, or `rate_in`
  when none is configured. The coordinator lists it with the ask.

### Removed

- `provider.max_input_bytes_per_unit`.

## [0.2.0] — 2026-11-18

The first stable release. Every release from here bumps the patch version.

### Added

- A backend answering `402` (account out of funds) takes every model off the
  book at once, under the new reason `backend_unfunded`. Held jobs the backend
  has not accepted are failed inside the fail grace, jobs it already accepted
  keep running, and the breaker cooldown lists the models again. `402` may not
  appear in `retry_statuses`. A refused batch create now deletes the input file
  it uploaded.

### Changed

- A backend answering `404` or `410` now counts toward the circuit breaker. The
  job is still failed back at once, under the new reason `backend_gone`; after
  `trip_after` such jobs in a row the model's asks are withdrawn like any other
  backend fault. Before, a withdrawn endpoint failed every job with its ask
  still on the book.
- A claim the node refuses (other than `AtCapacity`) now shelves the job on the
  same doubling backoff as an unresolvable payload, instead of re-fetching and
  re-claiming it on every sweep until it expires.

### Fixed

- A backend error that a later attempt recovered from — a stream cut mid-way, a
  `504` — is no longer reported to Sentry as an unhandled task exception.

## [0.1.0rc1] — 2026-10-01

Initial public release.
