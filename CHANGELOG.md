# Changelog

Format: [Semantic Versioning](https://semver.org). Newest first.

## [1.0.0] - 2026-10-05
### Added
- CI workflow (GitHub Actions) with a smoke test of every app page.
- README, `.gitignore`, `CHANGELOG.md`, `docs/GIT_GUIDE.md`.
### Changed
- `Churn` and `EngagementScore` are optional columns.
- Streamlit `width="stretch"` replaces deprecated `use_container_width`.
- `requirements.txt` trimmed to app needs; notebook extras in `requirements-notebooks.txt`.
- `data.csv` converted to LF line endings.
### Fixed
- KPIs overview crash when a non-marketing CSV is uploaded.
- "nan%" ROI labels now show "n/a".
- Churn page no longer shows a misleading 0% without a `Churn` column.
