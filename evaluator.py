from typing import Iterable, Optional


class Evaluator:
    """Scores API-testing performance before and after agent evolution."""

    # These are the test areas the agent is expected to cover.
    TEST_TYPES = ("auth", "edge_cases", "error_handling", "input_validation")

    def __init__(self, endpoint_target: int = 10, issue_target: int = 5) -> None:
        # Targets must be positive so the scoring math stays valid.
        if endpoint_target <= 0:
            raise ValueError("endpoint_target must be greater than 0")
        if issue_target <= 0:
            raise ValueError("issue_target must be greater than 0")

        self.endpoint_target = endpoint_target
        self.issue_target = issue_target
        self.before_score: Optional[int] = None
        self.after_score: Optional[int] = None

    def score(
        self,
        endpoints_discovered: int,
        test_types_run: Iterable[str],
        issues_found: int,
    ) -> int:
        # Endpoints and test coverage get most of the score.
        endpoint_points = self._ratio(endpoints_discovered, self.endpoint_target) * 40
        test_type_points = self._test_type_coverage(test_types_run) * 40
        issue_points = self._ratio(issues_found, self.issue_target) * 20

        return round(endpoint_points + test_type_points + issue_points)

    def record_before(
        self,
        endpoints_discovered: int,
        test_types_run: Iterable[str],
        issues_found: int,
    ) -> int:
        # Save the baseline before the agent has executed its improved plan.
        self.before_score = self.score(
            endpoints_discovered,
            test_types_run,
            issues_found,
        )
        return self.before_score

    def record_after(
        self,
        endpoints_discovered: int,
        test_types_run: Iterable[str],
        issues_found: int,
    ) -> int:
        # Save the score after the agent has run its tests.
        self.after_score = self.score(
            endpoints_discovered,
            test_types_run,
            issues_found,
        )
        return self.after_score

    def compare(self) -> dict:
        # Comparing early would hide a broken run, so fail clearly instead.
        if self.before_score is None or self.after_score is None:
            raise ValueError("Both before_score and after_score must be recorded first")

        return {
            "before_score": self.before_score,
            "after_score": self.after_score,
            "improvement": self.after_score - self.before_score,
        }

    def _test_type_coverage(self, test_types_run: Iterable[str]) -> float:
        # Normalize names so "edge cases" and "edge-cases" count the same.
        normalized_types = {
            test_type.strip().lower().replace(" ", "_").replace("-", "_")
            for test_type in test_types_run
        }
        covered_types = normalized_types.intersection(self.TEST_TYPES)

        return len(covered_types) / len(self.TEST_TYPES)

    @staticmethod
    def _ratio(value: int, target: int) -> float:
        # Clamp the ratio so extra findings never score above full credit.
        return min(max(value, 0) / target, 1.0)
