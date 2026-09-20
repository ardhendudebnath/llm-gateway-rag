# Security standards

These standards apply to every employee and contractor with access to Acme Cloud systems.

## Secrets management

Secrets live only in Keyring, our secrets vault. Secrets must never be committed to git, pasted
into tickets, or shared in chat. Long-lived secrets such as API keys for third-party services are
rotated every 90 days. Database credentials are not long-lived at all: the deploy pipeline issues
fresh credentials for every deploy, and they expire after 24 hours.

## Production access

Access to production requires multi-factor authentication with a hardware security key; phone
based one-time codes are not accepted. Production access is granted just in time, for a maximum
of eight hours per request, and every request must reference a ticket or an incident.

Managers review their team's access every quarter. When someone leaves the company, all of their
access is revoked within one hour of the HR notification.

## Vulnerability management

Dependencies are scanned for known vulnerabilities on every pull request. Remediation deadlines
depend on severity:

- critical vulnerabilities must be patched within 72 hours,
- high severity vulnerabilities within 14 days,
- medium severity vulnerabilities within 60 days,
- low severity findings are fixed during regular maintenance.

## Laptops

Company laptops must use full-disk encryption and lock the screen automatically after five
minutes of inactivity. Operating system updates must be installed within seven days of release.
A lost or stolen laptop must be reported to the IT desk immediately so it can be wiped remotely.

## Reporting security issues

Report any suspected security incident in the #security-alerts channel. Until the security team
has triaged it, a suspected security incident is handled as a SEV1.
