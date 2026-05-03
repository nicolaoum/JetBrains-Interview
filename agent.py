import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from evaluator import Evaluator
from logger import Logger


class StemAgent:
    """Agent that learns and improves API testing strategies."""

    DEFAULT_ENDPOINTS = (
        "/posts",
        "/comments",
        "/albums",
        "/photos",
        "/todos",
        "/users",
    )

    def __init__(self, base_url: str = "https://jsonplaceholder.typicode.com") -> None:
        # Load local settings before reading keys or model names.
        load_dotenv()

        self.base_url = base_url.rstrip("/")
        self.logger = Logger()
        self.evaluator = Evaluator()
        self.session = requests.Session()

        # The OpenAI client is optional until a phase actually needs it.
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        # Keep run state on the agent so each phase can use earlier results.
        self.suggested_endpoints = []
        self.discovered_endpoints = []
        self.exploration_results = {}
        self.reflection = ""
        self.testing_strategy = {}
        self.execution_results = []

    def discover(self) -> list:
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is required before discovery can run")

        # Ask for JSON so endpoint parsing stays predictable.
        prompt = (
            "Suggest 8-10 likely REST API endpoint paths for this base URL. "
            "Respond with JSON only using one key called endpoints. "
            "The endpoints value must be a list of path strings like /users or /posts.\n\n"
            f"Base URL: {self.base_url}"
        )

        self.logger.info("Asking OpenAI to suggest likely API endpoints")
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You suggest likely REST API paths as valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_discovery = response.choices[0].message.content.strip()

        try:
            discovery = json.loads(raw_discovery)
        except json.JSONDecodeError as error:
            raise ValueError(f"OpenAI returned invalid JSON: {raw_discovery}") from error

        endpoints = discovery.get("endpoints", [])
        if isinstance(endpoints, str):
            endpoints = [endpoints]

        # Clean up the model output before using it in URLs.
        self.suggested_endpoints = []
        for endpoint in endpoints:
            endpoint = str(endpoint).strip()
            if not endpoint:
                continue
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"
            self.suggested_endpoints.append(endpoint)

        self.logger.decision(
            f"OpenAI suggested endpoints: {', '.join(self.suggested_endpoints)}"
        )

        return self.suggested_endpoints

    def explore(self) -> dict:
        # Use known JSONPlaceholder paths if discovery has not run yet.
        candidate_endpoints = self.suggested_endpoints or list(self.DEFAULT_ENDPOINTS)

        self.logger.info("Starting API exploration")
        self.discovered_endpoints = []
        self.exploration_results = {}

        for endpoint in candidate_endpoints:
            # Probe each endpoint and save both successes and failures.
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

                # Keep a tiny preview of the response shape for reflection.
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

        # Give the model the concrete exploration results, not just a summary.
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

        # Turn the written reflection into data the executor can use.
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

        # Keep only the fields the execution phase needs later.
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

        # Start from zero so the final improvement is easy to see.
        before_score = self.evaluator.record_before(
            endpoints_discovered=0,
            test_types_run=[],
            issues_found=0,
        )
        self.logger.result(f"Before execution score: {before_score}")

        endpoints_to_test = self.testing_strategy.get("endpoints_to_test", [])
        test_types = self.testing_strategy.get("test_types", [])

        # OpenAI can return strings or objects, so make simple lists here.
        if isinstance(endpoints_to_test, str):
            endpoints_to_test = [endpoints_to_test]
        if isinstance(test_types, dict):
            test_types = test_types.keys()
        if isinstance(test_types, str):
            test_types = [test_types]
        if not endpoints_to_test:
            endpoints_to_test = self.discovered_endpoints

        # Run only the supported tests; use all of them if the plan is vague.
        supported_test_types = ("auth", "edge_cases", "error_handling", "input_validation")
        test_types_to_run = [
            str(test_type).strip().lower().replace(" ", "_").replace("-", "_")
            for test_type in test_types
            if str(test_type).strip().lower().replace(" ", "_").replace("-", "_")
            in supported_test_types
        ]
        if not test_types_to_run:
            test_types_to_run = list(supported_test_types)

        self.execution_results = []
        issues_found = 0
        tests_run = set()
        normalized_endpoints = []

        for endpoint in endpoints_to_test:
            # Normalize endpoint paths before building each request URL.
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

                # Each test has one clear expectation to check.
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
                    # Network failures count as issues because the test did not finish.
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

        # Score the run using what actually happened.
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
