import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from evaluator import Evaluator
from logger import Logger


class StemAgent:
    """Agent that learns and improves API testing strategies."""

    def __init__(self, base_url: str = "https://jsonplaceholder.typicode.com") -> None:
        load_dotenv()

        self.base_url = base_url.rstrip("/")
        self.logger = Logger()
        self.evaluator = Evaluator()
        self.session = requests.Session()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        self.discovered_endpoints = []
        self.exploration_results = {}
        self.reflection = ""
        self.testing_strategy = {}
        self.execution_results = []

    def explore(self) -> dict:
        candidate_endpoints = (
            "/posts",
            "/comments",
            "/albums",
            "/photos",
            "/todos",
            "/users",
        )

        self.logger.info("Starting API exploration")
        self.discovered_endpoints = []
        self.exploration_results = {}

        for endpoint in candidate_endpoints:
            url = f"{self.base_url}{endpoint}"
            self.logger.decision(f"Probing {endpoint}")

            try:
                response = self.session.get(url, timeout=10)
            except requests.RequestException as error:
                self.exploration_results[endpoint] = {
                    "url": url,
                    "status_code": None,
                    "discovered": False,
                    "error": str(error),
                }
                self.logger.info(f"{endpoint} failed: {error}")
                continue

            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "discovered": response.ok,
            }

            if response.ok:
                self.discovered_endpoints.append(endpoint)

                try:
                    data = response.json()
                except ValueError:
                    data = None

                if isinstance(data, list):
                    sample = data[0] if data else {}
                    result["item_count"] = len(data)
                    result["sample_keys"] = list(sample.keys()) if isinstance(sample, dict) else []
                elif isinstance(data, dict):
                    result["item_count"] = 1
                    result["sample_keys"] = list(data.keys())
                else:
                    result["item_count"] = 0
                    result["sample_keys"] = []

                self.logger.result(f"Discovered {endpoint} with status {response.status_code}")
            else:
                result["item_count"] = 0
                result["sample_keys"] = []
                self.logger.info(f"{endpoint} returned status {response.status_code}")

            self.exploration_results[endpoint] = result

        self.logger.result(
            f"Exploration found {len(self.discovered_endpoints)} endpoints"
        )
        return self.exploration_results

    def reflect(self) -> str:
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is required before reflection can run")

        exploration_summary = json.dumps(self.exploration_results, indent=2)
        prompt = (
            "Analyze these API exploration results and decide what tests matter most. "
            "Focus on auth, edge cases, error handling, and input validation. "
            "Explain the priorities clearly and recommend a concise testing strategy.\n\n"
            f"Exploration results:\n{exploration_summary}"
        )

        self.logger.info("Sending exploration results to OpenAI for reflection")
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an API testing strategist helping an agent improve.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        self.reflection = response.choices[0].message.content.strip()
        self.logger.evolution(self.reflection)

        return self.reflection

    def evolve(self) -> dict:
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is required before evolution can run")
        if not self.reflection.strip():
            raise ValueError("Reflection must be recorded before evolution can run")

        prompt = (
            "Convert this API testing reflection into a structured testing strategy. "
            "Respond with JSON only. The JSON object must include these keys: "
            "endpoints_to_test, test_types, and priorities.\n\n"
            f"Reflection:\n{self.reflection}"
        )

        self.logger.info("Converting reflection into a structured testing strategy")
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You extract API testing strategies as valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_strategy = response.choices[0].message.content.strip()

        try:
            strategy = json.loads(raw_strategy)
        except json.JSONDecodeError as error:
            raise ValueError(f"OpenAI returned invalid JSON: {raw_strategy}") from error

        self.testing_strategy = {
            "endpoints_to_test": strategy.get("endpoints_to_test", []),
            "test_types": strategy.get("test_types", []),
            "priorities": strategy.get("priorities", []),
        }
        self.logger.evolution(json.dumps(self.testing_strategy, indent=2))

        return self.testing_strategy

    def execute(self) -> dict:
        if not self.testing_strategy:
            raise ValueError("Testing strategy must be created before execution can run")

        before_score = self.evaluator.record_before(
            endpoints_discovered=0,
            test_types_run=[],
            issues_found=0,
        )
        self.logger.result(f"Before execution score: {before_score}")

        endpoints_to_test = self.testing_strategy.get("endpoints_to_test", [])
        test_types = self.testing_strategy.get("test_types", [])

        if isinstance(endpoints_to_test, str):
            endpoints_to_test = [endpoints_to_test]
        if isinstance(test_types, str):
            test_types = [test_types]
        if not endpoints_to_test:
            endpoints_to_test = self.discovered_endpoints

        supported_test_types = ("auth", "edge_cases", "error_handling", "input_validation")
        test_types_to_run = [
            test_type.strip().lower().replace(" ", "_").replace("-", "_")
            for test_type in test_types
            if test_type.strip().lower().replace(" ", "_").replace("-", "_")
            in supported_test_types
        ]
        if not test_types_to_run:
            test_types_to_run = list(supported_test_types)

        self.execution_results = []
        issues_found = 0
        tests_run = set()
        normalized_endpoints = []

        for endpoint in endpoints_to_test:
            endpoint = str(endpoint).strip()
            if not endpoint:
                continue
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"

            normalized_endpoints.append(endpoint)
            url = f"{self.base_url}{endpoint}"

            for test_type in test_types_to_run:
                tests_run.add(test_type)
                self.logger.decision(f"Running {test_type} test against {endpoint}")

                result = {
                    "endpoint": endpoint,
                    "test_type": test_type,
                    "status_code": None,
                    "expected": "",
                    "passed": False,
                    "issue": False,
                }

                try:
                    if test_type == "auth":
                        result["expected"] = "Fake token should still return 200"
                        response = self.session.get(
                            url,
                            headers={"Authorization": "Bearer fake-token"},
                            timeout=10,
                        )
                        result["status_code"] = response.status_code
                        result["passed"] = response.status_code == 200

                    elif test_type == "edge_cases":
                        result["expected"] = "Non-existent resource should return 404"
                        edge_url = f"{url.rstrip('/')}/99999"
                        response = self.session.get(edge_url, timeout=10)
                        result["status_code"] = response.status_code
                        result["passed"] = response.status_code == 404

                    elif test_type == "error_handling":
                        result["expected"] = "Malformed content type should not cause a 5xx"
                        response = self.session.get(
                            url,
                            headers={"Content-Type": "not-a-real-content-type"},
                            timeout=10,
                        )
                        result["status_code"] = response.status_code
                        result["passed"] = response.status_code < 500

                    elif test_type == "input_validation":
                        result["expected"] = "Missing required fields should be rejected"
                        response = self.session.post(url, json={}, timeout=10)
                        result["status_code"] = response.status_code
                        result["passed"] = response.status_code in (400, 422)

                    result["issue"] = not result["passed"]

                except requests.RequestException as error:
                    result["error"] = str(error)
                    result["issue"] = True

                if result["issue"]:
                    issues_found += 1
                    self.logger.result(
                        f"Unexpected response for {test_type} on {endpoint}"
                    )
                else:
                    self.logger.info(f"{test_type} test passed on {endpoint}")

                self.execution_results.append(result)

        after_score = self.evaluator.record_after(
            endpoints_discovered=len(set(normalized_endpoints)),
            test_types_run=tests_run,
            issues_found=issues_found,
        )
        comparison = self.evaluator.compare()

        self.logger.result(f"After execution score: {after_score}")
        self.logger.result(f"Issues found: {issues_found}")
        self.logger.result(f"Score improvement: {comparison['improvement']}")

        return {
            "before_score": comparison["before_score"],
            "after_score": comparison["after_score"],
            "improvement": comparison["improvement"],
            "issues_found": issues_found,
            "results": self.execution_results,
        }
