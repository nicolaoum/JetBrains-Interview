from agent import StemAgent


def main() -> None:
    agent = StemAgent()

    # Run the agent phases in the order they build on each other.
    agent.discover()
    agent.explore()
    agent.reflect()
    agent.evolve()
    summary = agent.execute()

    # Keep the terminal summary short; details are written to the log file.
    print("\nAgent Performance Summary")
    print("-------------------------")
    print(f"Before score: {summary['before_score']}")
    print(f"After score: {summary['after_score']}")
    print(f"Improvement: {summary['improvement']}")


if __name__ == "__main__":
    main()
