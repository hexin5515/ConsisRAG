def get_query_instruction(name: str) -> str:
    instructions = {
        "query_to_fact": "Represent the query for retrieving relevant facts.",
        "query_to_passage": "Represent the query for retrieving relevant passages.",
    }
    return instructions.get(name, "")
