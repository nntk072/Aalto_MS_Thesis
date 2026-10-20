from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path

    def runs(self) -> Path:
        return self.root / "runs"

    def run(self, run_id: str) -> Path:
        return self.runs() / run_id

    def campaigns(self) -> Path:
        return self.root / "campaigns"

    def campaign(self, campaign_id: str) -> Path:
        return self.campaigns() / campaign_id

    def evaluation(self, evaluation_id: str) -> Path:
        return self.root / "evaluation" / evaluation_id

    def research(self) -> Path:
        return self.root / "research"

    def reports(self) -> Path:
        return self.root / "reports"
