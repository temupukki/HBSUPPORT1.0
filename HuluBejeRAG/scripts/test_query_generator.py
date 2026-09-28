from app.llm.client import LLMClient
from app.llm.query_generator import QueryGenerator


def main():

    print("=" * 70)
    print("HULUBEJE LLM 1 TEST")
    print("=" * 70)

    client = LLMClient()

    generator = QueryGenerator(
        client=client
    )

    test_messages = [
        "How long does standard delivery take?",

        "የመደበኛ መላኪያ ስንት ቀን ይወስዳል?",

        "كم من الوقت يستغرق التوصيل العادي؟",
    ]

    for message in test_messages:

        print()
        print("-" * 70)
        print("CUSTOMER:")
        print(message)

        result = generator.understand(
            message
        )

        print()
        print("ORIGINAL LANGUAGE:")
        print(result.original_language)

        print()
        print("RETRIEVAL QUERY:")
        print(result.retrieval_query)

    print()
    print("=" * 70)
    print("LLM 1 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()