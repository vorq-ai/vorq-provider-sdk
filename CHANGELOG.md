# Changelog

All notable changes to `vorqd` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
