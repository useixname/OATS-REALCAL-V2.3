from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationVersionRegistry:
    highest_counter: dict[str, int] = field(default_factory=dict)
    payload_hashes: dict[tuple[str, int], str] = field(default_factory=dict)

    def register(self, role_version: str, counter: int, payload_hash: str) -> None:
        if counter < 1:
            raise ValueError("anti-rollback counter must be positive")
        previous = self.highest_counter.get(role_version, 0)
        key = (role_version, counter)
        if counter < previous:
            raise ValueError("calibration anti-rollback violation")
        if counter == previous:
            if key not in self.payload_hashes or self.payload_hashes[key] != payload_hash:
                raise ValueError("version/counter payload conflict")
            return
        self.payload_hashes[key] = payload_hash
        self.highest_counter[role_version] = counter
