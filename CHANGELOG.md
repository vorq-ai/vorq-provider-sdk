# Changelog

All notable changes to `vorqd` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- A backend answering `404` or `410` now counts toward the circuit breaker. The
  job is still failed back at once, under the new reason `backend_gone`; after
  `trip_after` such jobs in a row the model's asks are withdrawn like any other
  backend fault. Before, a withdrawn endpoint failed every job with its ask
  still on the book.

## [0.1.0rc1] — 2026-10-01

Initial public release.
