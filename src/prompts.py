def format_examples(df, label_column, explanation_column, num_examples=9):
    """
    Format examples for use in few-shot prompting.

    Args:
        df: DataFrame with labeled examples.
        label_column: Column containing manual binary labels ('TT Manual' or 'CB Manual').
        explanation_column: Column with human-written explanation.
        num_examples: Number of examples to include in the prompt.

    Returns:
        A single string containing formatted examples.
    """
    example_texts = []

    for _, row in df.head(num_examples).iterrows():
        text = (
            f"Project Title: {row['Title']}\n"
            f"Project Description: {row['Description']}\n"
            f"Label: {row[label_column]}\n"
            f"Explanation: {row[explanation_column]}\n"
        )
        example_texts.append(text)

    return "\n\n".join(example_texts)


def get_role_prompt(tool):
    """
    Returns the system role prompt for TT or CB.
    """
    if tool == "TT":
        return (
            "You are an expert in international development projects with a specialization in technology transfer "
            "within the context of climate change mitigation and adaptation. You have extensive knowledge "
            "of the Swiss definitions of this concept and are skilled at evaluating project descriptions to determine their "
            "alignment with the definition. Your task is to provide a clear and definitive answer on whether a given project "
            "involves technology transfer, strictly based on the provided Swiss definition and the project "
            "details. You must answer only 'yes' or 'no' based on the project's alignment with the definition. "
            "You can only answer as 1 (yes) or 0 (no)."
        )
    elif tool == "CB":
        return (
            "You are an expert in international development projects with a specialization in capacity building "
            "within the context of climate change mitigation and adaptation. You have extensive knowledge "
            "of the Swiss definitions of this concept and are skilled at evaluating project descriptions to determine their "
            "alignment with the definition. Your task is to provide a clear and definitive answer on whether a given project "
            "involves capacity building, strictly based on the provided Swiss definition and the project "
            "details. You must answer only 'yes' or 'no' based on the project's alignment with the definition. "
            "You can only answer as 1 (yes) or 0 (no)."
        )


def get_context_prompt(tool, definition_text, example_texts):
    """
    Returns the full user context prompt for a given tool (TT or CB).
    """
    concept = "technology transfer" if tool == "TT" else "capacity building"

    context = (
        f"Please evaluate whether the following development project involves {concept} according to the Swiss definition. "
        f"Consider the project's title and description carefully, and determine if it meets the criteria for {concept} (1 (yes) or 0 (no)) "
        f"based on the definition provided in the following. Small action credits should always be marked as 0. "
        f"If title and description are too vague to make an assumption, rather mark as 0 as well."
        f"\n\nExamples:\n{example_texts}\n\n"
        f"Here is the Definition: {definition_text}\n\n"
    )

    return context


def build_prompts(train_df, definitions, num_examples=9):
    """
    Builds all prompts (roles and contexts) for TT and CB.

    Returns:
        context_tt, role_tt, context_cb, role_cb
    """
    examples_tt = format_examples(train_df, "TT Manual", "Arguments TT", num_examples)
    examples_cb = format_examples(train_df, "CB Manual", "Argments CB", num_examples)

    role_tt = get_role_prompt("TT")
    role_cb = get_role_prompt("CB")

    context_tt = get_context_prompt("TT", definitions["TT"], examples_tt)
    context_cb = get_context_prompt("CB", definitions["CB"], examples_cb)

    return context_tt, role_tt, context_cb, role_cb
