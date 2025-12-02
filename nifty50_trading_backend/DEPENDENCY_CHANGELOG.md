# Dependency update changelog

Date: 2025-12-02

Summary:
- Updated backend Python dependencies to latest compatible versions, aligned with central constraints where applicable.
- Ensured FastAPI/Starlette/Pydantic compatibility for the existing FastAPI app without API changes.

Updated packages:
- fastapi: 0.115.12 -> 0.115.7 (align to central constraint for stability)
- starlette: 0.46.1 -> 0.41.3 (compatible with FastAPI 0.115.x)
- pydantic: 2.11.3 -> 2.10.6 (central constraint; v2 line)
- flake8: 7.2.0 -> 7.1.1
- pyflakes: 3.3.2 -> 3.2.0
- pytest: 8.3.5 -> 8.3.4
- python-dotenv: 1.1.0 -> 1.0.1
- rich: 14.0.0 -> 13.9.4
- rich-toolkit: 0.14.1 -> 0.13.2
- typer: 0.15.2 -> 0.12.5
- typing_extensions: 4.13.1 -> 4.12.2

Unchanged (already compatible):
- uvicorn: 0.34.0 (central constraint)
- anyio, httpx/httpcore, numpy, pandas, yfinance, prometheus_client and others remain as previously compatible pins.

Notes:
- No code changes were required; the project uses Pydantic v2 and Starlette-compatible APIs.
- No ports or runtime config changed.
- Run: `pip install -r requirements.txt` to apply updates.

Acceptance criteria mapping:
- Dependencies updated and pinned.
- Lock/hashes not used; requirements.txt serves as the lock.
- App should start with `uvicorn src.api.main:app --reload --port 8000` (unchanged).
