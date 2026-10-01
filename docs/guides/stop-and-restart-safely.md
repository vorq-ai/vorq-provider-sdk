---
title: Stop and restart safely
description: Drain the daemon with SIGTERM, keep the state file, and know what a hard kill costs.
---

## Stop with SIGTERM

On `SIGTERM` or `SIGINT` the daemon drains:

1. It stops claiming new jobs, and `/healthz` turns `503`.
2. Jobs running at a resumable backend (a poll mapping with `handle`, or the
   `openai-responses` and `openai-batch` presets) are **suspended**: the work keeps running at
   the backend and the next start resumes polling it.
3. It withdraws every ask it published. The ask book has no expiry, so asks left standing
   would keep matching you with work.
4. It waits for every other in-flight job to finish and settle, then exits.

Give your supervisor a stop timeout at least as long as your longest synchronous job. A
supervisor that sends `SIGKILL` too early turns the drain into a hard kill.

## Keep the state file

Suspended jobs are identified by the handles in `provider.state_db` (default
`vorqd-state.sqlite` in the working directory). The file is small and only holds jobs in
flight, but it must survive a restart to be useful. In a container, keep it on a volume:

```yaml
provider:
  state_db: /var/lib/vorqd/state.sqlite
```

with `/var/lib/vorqd` a volume writable by the daemon's user.

If you scale to zero, suspended jobs wait until a daemon with the same state file starts
again. Keep one instance running, or accept that a long wait can run past the SLA.

## What happens on the next start

On its first poll sweep, the daemon lists the jobs claimed under its provider id and:

- resumes polling every job whose handle is on record (`resumed`);
- re-runs every other job still inside its SLA window for a model it serves (`recovered`);
- fails back the rest, so the client is refunded immediately;
- forgets handles for jobs that were settled, failed or reclaimed in the meantime.

## Hard kill

`SIGKILL` gives the daemon no chance to drain. The next start recovers what it can as above,
but downtime is not given back: a job whose window closed while the daemon was down is failed,
and that counts as a missed SLA against your reputation. Asks stay on the book until the new
process publishes its own. Use `SIGKILL` only on a daemon that is already stuck.

## Related

- [Deploy with Docker](./deploy-with-docker.md#stop-it)
- [Job lifecycle](../concepts/job-lifecycle.md)
