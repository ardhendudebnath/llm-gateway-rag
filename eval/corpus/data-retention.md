# Data retention and deletion

We keep data only as long as we need it. This policy defines retention periods by data type and
how customer data is deleted.

## Retention periods

| Data | Retention |
|---|---|
| Application logs | 30 days |
| HTTP access logs | 90 days |
| Distributed traces | 14 days |
| Metrics | 13 months, so year-over-year comparisons remain possible |
| Audit logs | 7 years, to meet financial regulations |
| Database backups | 35 days |

## Deleting customer data

When a customer closes their account, their data is deleted from production systems within 30 days
of account closure. Copies in backups are not edited; they age out as backups expire after 35
days. A closed account's data is therefore fully erased from all systems within 65 days.

## Privacy requests

Data subject access requests under the GDPR are answered within 30 days. Erasure requests are
processed within 14 days, which is faster than the legal maximum. A legal hold overrides every
retention period in this document: data under a legal hold is kept until the legal team releases
it.

## Backups

We take a full backup of every production database daily and an incremental backup every hour.
Backups are stored in a second region: the primary region is Frankfurt and backups are kept in
Dublin. A restore from backup is tested every month, and the result is recorded in the reliability
report.
