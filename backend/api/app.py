from __future__ import annotations

import json
import time

from flask import Flask, Response, jsonify, request

from .service import VolatilityAnalysisService


def create_app() -> Flask:
    app = Flask(__name__)
    service = VolatilityAnalysisService()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/analyze")
    def analyze():
        body = request.get_json(silent=True) or {}
        ticker = body.get("ticker", "")
        sabr_beta = float(body.get("sabr_beta", 0.5))
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        try:
            job_id = service.start_job(ticker=ticker, sabr_beta=sabr_beta)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"job_id": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = service.get_job(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(
            {
                "job_id": job.id,
                "ticker": job.ticker,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "error": job.error,
            }
        )

    @app.get("/api/jobs/<job_id>/result")
    def job_result(job_id: str):
        job = service.get_job(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        if job.status == "failed":
            return jsonify({"status": job.status, "error": job.error}), 500
        if job.result is None:
            return jsonify({"status": job.status, "message": "result not ready"}), 202
        return jsonify({"status": job.status, "result": job.result})

    @app.get("/api/jobs/<job_id>/stream")
    def job_stream(job_id: str):
        job = service.get_job(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404

        def generate():
            # Initial handshake event.
            init = {
                "type": "connected",
                "job_id": job_id,
                "status": job.status,
            }
            yield service.event_to_sse(init)

            while True:
                try:
                    event = job.events.get(timeout=2.0)
                    yield service.event_to_sse(event)
                    if event.get("type") in {"completed", "error"}:
                        # Final snapshot payload for client convenience.
                        tail = {
                            "type": "final",
                            "job_id": job.id,
                            "status": job.status,
                            "result_ready": job.result is not None,
                            "error": job.error,
                        }
                        yield service.event_to_sse(tail)
                        break
                except Exception:
                    # Keep-alive for proxies/browsers.
                    yield ": keep-alive\n\n"
                    if job.status in {"completed", "failed"}:
                        break
                time.sleep(0.01)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
