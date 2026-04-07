from __future__ import annotations

import json
from typing import Any, Dict, List

from django.core.management.base import BaseCommand, CommandError

from chatbot.services.rag_service import run_rag_query


def _token_set(text: str) -> set[str]:
    import re

    return {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text or "") if len(t) >= 3}


class Command(BaseCommand):
    help = "Run offline RAG evaluation over a JSONL benchmark."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            required=True,
            help="Path to JSONL rows: {query, expected_terms?, gold_chunk_ids?, should_abstain?}",
        )
        parser.add_argument("--k", type=int, default=3, help="K for precision@k over retrieved chunks")

    def handle(self, *args, **opts):
        path = str(opts["input"])
        k = max(1, int(opts.get("k") or 3))
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    rows.append(json.loads(ln))
        except Exception as e:
            raise CommandError(f"Failed to read dataset: {e}")
        if not rows:
            raise CommandError("Dataset is empty.")

        total = 0
        abstained = 0
        correct_abstain = 0
        incorrect_abstain = 0
        faith_hits = 0
        hallu = 0
        p_at_k_sum = 0.0
        p_at_k_count = 0
        for r in rows:
            q = str(r.get("query") or "")
            exp = {str(t).lower() for t in (r.get("expected_terms") or [])}
            gold_chunk_ids = {str(x) for x in (r.get("gold_chunk_ids") or [])}
            should_abstain = bool(r.get("should_abstain"))
            if not q:
                continue
            res = run_rag_query(query=q)
            ans = str(res.get("answer") or "")
            total += 1
            is_abstain = bool(res.get("fallback_used"))
            if is_abstain:
                abstained += 1
            if should_abstain and is_abstain:
                correct_abstain += 1
            if (not should_abstain) and is_abstain:
                incorrect_abstain += 1
            ans_terms = _token_set(ans)
            if exp:
                if exp.intersection(ans_terms):
                    faith_hits += 1
                else:
                    hallu += 1
            if gold_chunk_ids:
                retrieved = res.get("retrieved") or []
                top = [str((x or {}).get("chunk_id") or "") for x in retrieved[:k]]
                if top:
                    hits = sum(1 for cid in top if cid in gold_chunk_ids)
                    p_at_k_sum += float(hits) / float(len(top))
                    p_at_k_count += 1

        precision = faith_hits / max(1, (total - abstained))
        abstain_rate = abstained / max(1, total)
        hallu_rate = hallu / max(1, total)
        p_at_k = p_at_k_sum / max(1, p_at_k_count)
        abstain_precision = correct_abstain / max(1, abstained)
        abstain_recall = correct_abstain / max(1, sum(1 for r in rows if bool(r.get("should_abstain"))))
        self.stdout.write(
            json.dumps(
                {
                    "total": total,
                    f"precision_at_{k}": round(p_at_k, 4),
                    "precision_non_abstain": round(precision, 4),
                    "faithfulness_proxy": round(precision, 4),
                    "abstain_rate": round(abstain_rate, 4),
                    "abstain_precision": round(abstain_precision, 4),
                    "abstain_recall": round(abstain_recall, 4),
                    "hallucination_proxy_rate": round(hallu_rate, 4),
                    "regression_alerts": [
                        x
                        for x in [
                            f"low_precision_at_{k}" if p_at_k < 0.7 and p_at_k_count > 0 else "",
                            "high_hallucination" if hallu_rate > 0.2 else "",
                            "abstain_recall_low" if abstain_recall < 0.7 else "",
                        ]
                        if x
                    ],
                },
                indent=2,
            )
        )

