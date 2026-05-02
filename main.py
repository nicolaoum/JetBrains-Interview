from agent import StemAgent


def main() -> None:
    agent = StemAgent()

    agent.explore()
    agent.reflect()
    agent.evolve()
    summary = agent.execute()

    print("\nAgent Performance Summary")
    print("-------------------------")
    print(f"Before score: {summary['before_score']}")
    print(f"After score: {summary['after_score']}")
    print(f"Improvement: {summary['improvement']}")


if __name__ == "__main__":
    main()
