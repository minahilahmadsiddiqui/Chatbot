from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from django.core.management.base import BaseCommand

from chatbot.services.cross_encoder_service import cross_encoder_healthcheck, rerank_with_cross_encoder


class Command(BaseCommand):
    help = "Cross-encoder readiness and simple latency/SLA benchmark."

    def add_arguments(self, parser):
        parser.add_argument("--runs", type=int, default=30, help="Number of benchmark runs")
        parser.add_argument("--top-n", type=int, default=20, help="Rerank top-N candidates")
        parser.add_argument("--sla-ms", type=int, default=900, help="Target average latency SLA")

    def handle(self, *args, **opts):
        health = cross_encoder_healthcheck()
        if not bool(health.get("ready")):
            self.stdout.write(json.dumps({"ready": False, "health": health}, indent=2))
            return

        runs = max(1, int(opts.get("runs") or 30))
        top_n = max(1, int(opts.get("top_n") or 20))
        sla_ms = max(1, int(opts.get("sla_ms") or 900))
        items: List[Dict[str, Any]] = []
        for i in range(max(top_n, 25)):
            items.append(
                {
                    "payload": {
                        "source_section": f"section {i}",
                        "text": f"Policy content {i} about leave reimbursement and approval workflow.",
                    },
                    "_rrf": 0.1 / (i + 1),
                }
            )
        query = "how many hajj leaves are allowed"
        times = []
        for _ in range(runs):
            t0 = time.time()
            rerank_with_cross_encoder(query=query, items=items, top_n=top_n)
            times.append(int((time.time() - t0) * 1000))
        avg_ms = int(sum(times) / max(1, len(times)))
        p95_ms = sorted(times)[int(0.95 * (len(times) - 1))] if times else 0
        self.stdout.write(
            json.dumps(
                {
                    "ready": True,
                    "runs": runs,
                    "top_n": top_n,
                    "avg_ms": avg_ms,
                    "p95_ms": int(p95_ms),
                    "sla_ms": sla_ms,
                    "sla_pass": bool(avg_ms <= sla_ms),
                    "health": health,
                },
                indent=2,
            )
        )

