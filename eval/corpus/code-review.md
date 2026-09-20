# Code review and continuous integration

Every change to a production repository is reviewed by at least one other engineer before it is
merged. Code review is a shared responsibility, not a gate owned by senior engineers.

## Approvals

A normal change needs one approval. Changes to authentication, billing or infrastructure code need
two approvals, and the CODEOWNERS file makes sure one of the reviewers belongs to the owning team.
Authors may not approve their own pull requests, even when they have admin rights.

## Pull request size

Keep pull requests under 400 changed lines. A larger pull request must either be split into
smaller ones or link to a design document that explains the change. Generated files and lock files
do not count toward the limit.

## Review turnaround

Reviewers give a first review within one business day. Authors respond to review comments within
two business days, otherwise the pull request is labelled stale. Stale pull requests are closed
automatically after 14 days; they can be reopened at any time.

## Continuous integration

A pull request can only be merged when CI is green: linting, unit tests and integration tests must
all pass. Test coverage may not drop by more than half a percentage point in a single pull request.
A flaky test is quarantined within 24 hours of being reported and must be fixed within one week,
or deleted if it no longer provides value.

## Merging

We only use squash merges, so every pull request becomes a single commit on the main branch.
Commit messages follow the Conventional Commits format, for example "fix: handle empty cart".
