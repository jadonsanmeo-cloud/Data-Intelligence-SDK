from __future__ import annotations

from data_intelligence_sdk.runtime.skills import SkillRegistryClient
import pytest


class _Response:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Http:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/skills"):
            return _Response(
                [
                    {
                        "id": "report-validation",
                        "name": "Report validation",
                        "description": "Validate reports.",
                    }
                ]
            )
        return _Response(
            {
                "id": "report-validation",
                "name": "Report validation",
                "description": "Validate reports.",
                "body": "# Report validation\nCheck totals.",
            }
        )


class _NamedSkillHttp:
    def __init__(self, summaries) -> None:
        self.summaries = summaries
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/skills"):
            return _Response(self.summaries)
        skill_id = url.rsplit("/", 1)[-1]
        summary = next(item for item in self.summaries if item["id"] == skill_id)
        return _Response({**summary, "body": f"# {summary['name']} prompt"})


def test_skill_registry_client_loads_visible_skill_bodies_with_workspace_scope() -> (
    None
):
    http = _Http()
    client = SkillRegistryClient(
        base_url="http://skills",
        user_id="user-a",
        organization_id="tenant-a",
        workspace_id="workspace-a",
        http_client=http,
    )

    skills = client.load()

    assert skills[0].name == "Report validation"
    assert skills[0].body == "# Report validation\nCheck totals."
    assert http.calls[0][1]["headers"]["X-Workspace-ID"] == "workspace-a"
    assert http.calls[1][1]["params"] == {"workspace_id": "workspace-a"}


def test_skill_registry_client_loads_named_skill_on_demand() -> None:
    http = _NamedSkillHttp(
        [
            {
                "id": "report-validation",
                "name": "report-validation",
                "description": "Validate reports.",
                "scope": "workspace",
            }
        ]
    )
    client = SkillRegistryClient(
        base_url="http://skills",
        user_id="user-a",
        organization_id="tenant-a",
        workspace_id="workspace-a",
        http_client=http,
    )

    skill = client.load_skill("report-validation")

    assert skill.name == "report-validation"
    assert skill.body == "# report-validation prompt"
    assert http.calls[0][1]["params"] == {"workspace_id": "workspace-a"}


def test_skill_registry_client_requires_scope_for_duplicate_names() -> None:
    summaries = [
        {"id": "user-skill", "name": "review", "scope": "personal"},
        {"id": "shared-skill", "name": "review", "scope": "workspace"},
    ]
    client = SkillRegistryClient(
        base_url="http://skills",
        user_id="user-a",
        organization_id="tenant-a",
        workspace_id="workspace-a",
        http_client=_NamedSkillHttp(summaries),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        client.load_skill("review")

    assert client.load_skill("review", scope="workspace").skill_id == "shared-skill"
