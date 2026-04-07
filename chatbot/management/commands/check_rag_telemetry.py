from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


def _telemetry_path() -> str:
    base = os.path.join(getattr(settings, "BASE_DIR", "."), "logs")
    return os.path.join(base, "rag_telemetry.jsonl")


class Command(BaseCommand):
    help = "Analyze RAG telemetry and emit regression alerts + threshold recalibration hints."

    def add_arguments(self, parser):
        parser.add_argument("--window", type=int, default=200, help="Recent events window size")

    def handle(self, *args, **opts):
        path = _telemetry_path()
        if not os.path.exists(path):
            raise CommandError(f"Telemetry file not found: {path}")
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
        if not rows:
            raise CommandError("No telemetry rows found.")
        win = max(10, int(opts.get("window") or 200))
        recent = rows[-win:]
        min_samples = int(getattr(settings, "RAG_REGRESSION_ALERT_MIN_SAMPLES", 30))
        if len(recent) < min_samples:
            self.stdout.write(
                json.dumps(
                    {
                        "status": "insufficient_samples",
                        "samples": len(recent),
                        "min_required": min_samples,
                    },
                    indent=2,
                )
            )
            return

        fallback_rate = sum(1 for r in recent if bool(r.get("fallback_used"))) / len(recent)
        latencies = [int(r.get("latency_ms") or 0) for r in recent]
        p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0
        ans_scores = []
        for r in recent:
            diag = r.get("retrieval_diagnostics") or {}
            try:
                ans_scores.append(float(diag.get("score") or 0.0))
            except Exception:
                continue
        mean_ans = sum(ans_scores) / max(1, len(ans_scores))

        alerts = []
        recs = []
        if fallback_rate > 0.45:
            alerts.append("fallback_rate_high")
            recs.append("lower RAG_MIN_ANSWERABILITY_SCORE by 0.03-0.07")
        if p95 > 2200:
            alerts.append("latency_p95_high")
            recs.append("decrease RAG_CROSS_ENCODER_TOP_N or disable cross-encoder for long queries")
        if mean_ans < 0.45:
            alerts.append("answerability_mean_low")
            recs.append("tighten retrieval threshold or increase top_k by 1 for better recall")

        self.stdout.write(
            json.dumps(
                {
                    "samples": len(recent),
                    "fallback_rate": round(fallback_rate, 4),
                    "latency_p95_ms": int(p95),
                    "mean_answerability": round(mean_ans, 4),
                    "alerts": alerts,
                    "recalibration_hints": recs,
                },
                indent=2,
            )
        )

