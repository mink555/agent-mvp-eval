"""ToolCard JSON Store — 영속화 + 버전 이력 + Draft/Publish/Rollback.

운영자가 Admin UI에서 ToolCard 메타데이터를 수정하면:
  1. draft로 임시 저장 (챗봇 미반영)
  2. publish 시 메모리 REGISTRY + ChromaDB 즉시 반영 + JSON 영속화
  3. 이전 버전은 history에 누적되어 롤백 가능

JSON 파일(data/toolcard_overrides.json)이 없으면 코드 정의 카드만 사용.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.tool_search.tool_cards import ToolCard, REGISTRY as CODE_REGISTRY

logger = logging.getLogger("insurance.toolcard_store")

_DEFAULT_PATH = Path("data/toolcard_overrides.json")
_MAX_HISTORY = 30


def _card_to_dict(card: ToolCard) -> dict[str, Any]:
    return {
        "name": card.name,
        "purpose": card.purpose,
        "when_to_use": list(card.when_to_use),
        "when_not_to_use": list(card.when_not_to_use),
        "tags": list(card.tags),
    }


def _dict_to_card(d: dict[str, Any]) -> ToolCard:
    return ToolCard(
        name=d["name"],
        purpose=d.get("purpose", ""),
        when_to_use=tuple(d.get("when_to_use", ())),
        when_not_to_use=tuple(d.get("when_not_to_use", ())),
        tags=tuple(d.get("tags", ())),
    )


class ToolCardStore:
    """ToolCard 영속 저장소. 스레드 안전."""

    def __init__(
        self,
        path: Path = _DEFAULT_PATH,
        on_publish: Callable[[ToolCard], None] | None = None,
    ) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._on_publish = on_publish
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # ── 로드/저장 ─────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("ToolCard override file not found, starting fresh")
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._store = raw.get("cards", {})
            logger.info("Loaded %d ToolCard overrides from %s", len(self._store), self._path)
        except Exception:
            logger.exception("Failed to load ToolCard overrides")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cards": self._store,
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 읽기 ─────────────────────────────────────────────────────

    def get_published(self, name: str) -> ToolCard | None:
        """Publish된 override 카드. 없으면 None(코드 카드 사용)."""
        with self._lock:
            entry = self._store.get(name)
            if entry and entry.get("published"):
                return _dict_to_card(entry["published"])
        return None

    def get_draft(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(name)
            return copy.deepcopy(entry.get("draft")) if entry else None

    def get_history(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            entry = self._store.get(name)
            return copy.deepcopy(entry.get("history", [])) if entry else []

    def get_effective_card(self, name: str) -> ToolCard | None:
        """실제 적용 중인 카드: published override > 코드 정의."""
        pub = self.get_published(name)
        return pub if pub else CODE_REGISTRY.get(name)

    def list_overrides(self) -> list[str]:
        """Override가 있는 도구 이름 목록."""
        with self._lock:
            return list(self._store.keys())

    def get_status(self, name: str) -> dict[str, Any]:
        """도구의 현재 상태 요약."""
        with self._lock:
            entry = self._store.get(name, {})
            has_code = name in CODE_REGISTRY
            has_published = bool(entry.get("published"))
            has_draft = bool(entry.get("draft"))
            history = entry.get("history", [])
            return {
                "name": name,
                "source": "override" if has_published else "code" if has_code else "none",
                "has_draft": has_draft,
                "version": history[-1]["version"] if history else 0,
                "history_count": len(history),
            }

    # ── 쓰기 ─────────────────────────────────────────────────────

    def save_draft(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Draft 임시 저장. 챗봇에는 미반영."""
        data["name"] = name
        with self._lock:
            if name not in self._store:
                self._store[name] = {"published": None, "draft": None, "history": []}
            self._store[name]["draft"] = data
            self._save()
        logger.info("Draft saved for '%s'", name)
        return data

    def publish(self, name: str, note: str = "") -> ToolCard:
        """Draft → Published. 메모리 + ChromaDB 반영 + JSON 저장."""
        with self._lock:
            entry = self._store.get(name)
            if not entry or not entry.get("draft"):
                raise ValueError(f"No draft to publish for '{name}'")

            draft_data = entry["draft"]
            draft_data["name"] = name

            prev_published = copy.deepcopy(entry.get("published"))
            history = entry.get("history", [])
            next_version = (history[-1]["version"] + 1) if history else 1

            history.append({
                "version": next_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": copy.deepcopy(draft_data),
                "note": note or f"v{next_version} published",
                "previous": prev_published,
            })

            if len(history) > _MAX_HISTORY:
                history[:] = history[-_MAX_HISTORY:]

            entry["published"] = draft_data
            entry["draft"] = None
            entry["history"] = history
            self._save()

            card = _dict_to_card(draft_data)

        from app.tool_search.tool_cards import REGISTRY
        REGISTRY[name] = card
        logger.info("Published ToolCard '%s' (v%d)", name, next_version)

        if self._on_publish:
            self._on_publish(card)

        return card

    def publish_direct(self, name: str, data: dict[str, Any], note: str = "") -> ToolCard:
        """Draft 없이 바로 Publish. 간편 수정용."""
        self.save_draft(name, data)
        return self.publish(name, note)

    def rollback(self, name: str, target_version: int) -> ToolCard:
        """특정 버전으로 롤백."""
        with self._lock:
            entry = self._store.get(name)
            if not entry:
                raise ValueError(f"No override history for '{name}'")

            history = entry.get("history", [])
            target = next((h for h in history if h["version"] == target_version), None)
            if not target:
                raise ValueError(f"Version {target_version} not found for '{name}'")

            rollback_data = copy.deepcopy(target["data"])
            prev_published = copy.deepcopy(entry.get("published"))

            next_version = (history[-1]["version"] + 1) if history else 1
            history.append({
                "version": next_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": rollback_data,
                "note": f"rollback to v{target_version}",
                "previous": prev_published,
            })

            if len(history) > _MAX_HISTORY:
                history[:] = history[-_MAX_HISTORY:]

            entry["published"] = rollback_data
            entry["draft"] = None
            entry["history"] = history
            self._save()

            card = _dict_to_card(rollback_data)

        from app.tool_search.tool_cards import REGISTRY
        REGISTRY[name] = card
        logger.info("Rolled back '%s' to v%d (new v%d)", name, target_version, next_version)

        if self._on_publish:
            self._on_publish(card)

        return card

    def discard_draft(self, name: str) -> None:
        """Draft 폐기."""
        with self._lock:
            entry = self._store.get(name)
            if entry:
                entry["draft"] = None
                self._save()

    def reset_to_code(self, name: str) -> ToolCard | None:
        """Override를 제거하고 코드 정의 카드로 복원."""
        with self._lock:
            if name in self._store:
                del self._store[name]
                self._save()

        code_card = CODE_REGISTRY.get(name)
        if code_card:
            from app.tool_search.tool_cards import REGISTRY
            REGISTRY[name] = code_card
            if self._on_publish:
                self._on_publish(code_card)
        return code_card

    @staticmethod
    def diff(card_a: dict[str, Any], card_b: dict[str, Any]) -> dict[str, Any]:
        """두 카드 데이터의 필드별 차이를 반환."""
        changes: dict[str, Any] = {}
        all_keys = {"purpose", "when_to_use", "when_not_to_use", "tags"}
        for key in all_keys:
            old_val = card_a.get(key)
            new_val = card_b.get(key)
            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}
        return changes


_store_instance: ToolCardStore | None = None
_store_lock = threading.Lock()


def get_toolcard_store(**kwargs) -> ToolCardStore:
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = ToolCardStore(**kwargs)
    return _store_instance
