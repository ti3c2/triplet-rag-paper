import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Prompt(BaseModel):
    text: str
    name: str


class QAPair(BaseModel):
    question: str = Field()
    answer: str = Field()


class QAPairs(BaseModel):
    items: List[QAPair]


class Questions(BaseModel):
    questions: List[str] = Field()


class Answers(BaseModel):
    answers: List[str] = Field()


def format_json_schema(json_text: str) -> str:
    # Resolve errors with curly braces
    return json_text.replace("{", "{{").replace("}", "}}")


# Json schema with $defs
# qa_json_schema = json.dumps(QAPairs.model_json_schema())

# Json example
JSON_INDENT = 2
qa_json_schema = format_json_schema(
    json.dumps(
        QAPairs(
            items=[
                QAPair(question="<Question 1 text>", answer="<Answer 1 text>"),
                QAPair(question="<Question 2 text>", answer="<Answer 2 text>"),
            ]
        ).model_dump(),
        indent=JSON_INDENT,
    )
)

# Example for prompt_qa_general
qa_general_example = format_json_schema(
    json.dumps(
        QAPairs(
            items=[
                QAPair(
                    question="What happened in Shanghai's financial district?",
                    answer="Authorities evacuated several buildings in Shanghai's financial district after reports of structural instability.",
                ),
                QAPair(
                    question="Were any buildings evacuated in Shanghai?",
                    answer="Yes, multiple office towers were evacuated due to safety concerns.",
                ),
                QAPair(
                    question="Did any airports in China close?",
                    answer="Some regional airports suspended operations temporarily as a precaution.",
                ),
                QAPair(
                    question="How did the evacuation affect flights?",
                    answer="Flight schedules were disrupted nationwide, with many delays and cancellations reported.",
                ),
                QAPair(
                    question="Were there evacuations in other parts of China?",
                    answer="Evacuations were also reported in several nearby cities, including Hangzhou and Suzhou.",
                ),
                QAPair(
                    question="How did airlines respond to the situation?",
                    answer="Airlines provided free ticket changes and refunds for affected passengers.",
                ),
                QAPair(
                    question="Were any industrial sites affected in China?",
                    answer="Industrial operations in coastal regions were temporarily halted for safety inspections.",
                ),
            ]
        ).model_dump(),
        indent=JSON_INDENT,
    )
)

# Schema for queries and answers generated sequentially
q_json_schema = format_json_schema(
    json.dumps(
        Questions(questions=["<Question 1 text>", "<Question 2 text>"]).model_dump(),
        indent=JSON_INDENT,
    )
)
a_json_schema = format_json_schema(
    json.dumps(
        Answers(answers=["<Answer 1 text>", "<Answer 2 text>"]).model_dump(),
        indent=JSON_INDENT,
    )
)


class PromptSettings(BaseModel):
    """
    Settings for prompts; provides a registry of prompt templates and helpers.

    To add new prompts, add a new field with the name of the prompt and the text of the prompt,
    like this:

    prompt_data_morgana: str = Field(
        default=(
            "Generate numerous questions to properly capture all the important parts of the text.\n"
            "Generate in all CAPS.\n"
            f"Answer in this json schema:\n{qa_json_schema}\n\n"
            "Text:\n{chunk_text}"
        ),
        exclude=True,
    )

    Notes:
    - Don't forget to add formatting instructions, especially for models hosted by vllm.
    - The name of the prompt is configured in the .env with the key `QUESTION_GENERATION_PROMPT_NAME=`
      and corresponds to the name right after `prompt_` in the field name.
    """

    prompt_system_qgen: str = Field(
        default=(
            "You are a helpful assistant that generates questions and answers for a given text."
        ),
        exclude=True,
    )

    prompt_basic_nq: str = Field(
        default=(
            "Generate numerous questions to properly capture all the important parts of the text.\n"
            f"Answer in this json schema:\n{qa_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )
    prompt_complex_nq: str = Field(
        default=(
            "Read the following text and generate numerous factual question-answer pairs "
            "designed to resemble authentic user search queries and natural language variations. "
            "Each question should accurately and semantically capture important aspects of the text, "
            "with varying lengths and complexities that mirror real-world search patterns. "
            "Include both shorter, keyword-focused questions such as 'who founded Tesla Motors' and longer, "
            "natural style questions like 'when did Elon Musk first start Tesla company'. "
            "Incorporate 'how' and 'why' questions to reflect genuine user curiosity. "
            "Avoid using phrases like 'according to the text' and abstain from pronouns by specifying names or entities. "
            "Ensure questions are not overly formal or artificial, maintaining a natural query style. "
            f"Answer in this json schema:\n{qa_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )

    prompt_basic_multihop: str = Field(
        default=(
            "Generate enough multi-hop questions along with their answers to properly capture "
            "all the important parts of the text. These questions should require integrating multiple pieces "
            "of information to answer.\n"
            f"Answer in this json schema:\n{qa_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )
    prompt_complex_multihop: str = Field(
        default=(
            "Read the following text and generate complex, multi-hop questions that require "
            "integrating multiple pieces of information from the text to answer. "
            "The questions should involve reasoning and synthesis, referring to different parts or aspects of the text. "
            "Do not use phrases like 'according to the text', 'mentioned in the text', or 'in the text'. "
            "All questions should be one sentence long. Never use pronouns in questions; instead, use the actual names or entities. "
            f"Answer in this json schema:\n{qa_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )

    prompt_rag_system: str = Field(
        default="You are a helpful assistant that answers questions based on provided context.",
        exclude=True,
    )
    prompt_rag_response: str = Field(
        default=(
            "Based on the following retrieved context, provide a comprehensive answer to the question. "
            "Use only the information from the context and cite relevant parts. "
            "If the context doesn't contain enough information to answer the question, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        exclude=True,
    )

    prompt_rag_response_triple: str = Field(
        default=(
            "Based on the following retrieved context, provide a comprehensive answer to the question. "
            "Use only the information from the context and cite relevant parts. "
            "If the context doesn't contain enough information to answer the question, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {reference_question}\n\n"
            "Answer: {reference_answer}\n\n"
            "Question: {question}\n\n"
            "Answer: "
        ),
        exclude=True,
    )

    prompt_qa_general: str = Field(
        default=(
            "You are helping to build a question answering system using retrieved documents.\n"
            "\n"
            "Given the following document, generate a list of **search queries and their corresponding answers** "
            "that a user might naturally type into Google or ask an assistant — assuming they know nothing "
            "about the content beforehand.\n"
            "\n"
            "Guidelines for QUESTIONS:\n"
            '- Each question must be fully **standalone**. NEVER refer to "the event", "this situation", '
            'or any pronouns like "it", "they", or "this". Always use explicit entities or locations from the text '
            '(e.g., "Shanghai", "airports in China").\n'
            "- Focus on **general**, natural-sounding questions that seek information — not paraphrases of sentences.\n"
            "- Use **search query style**, like something a real person might ask out of curiosity.\n"
            '- Prefer broad, investigative formulations such as **"what happened", "were there", "how did", "why did"**, '
            "reflecting interest in causes, effects, or outcomes.\n"
            "- Cover **different angles**: locations, organizations, transportation, public response, economic impact, etc.\n"
            "\n"
            "Guidelines for ANSWERS:\n"
            "- Each answer must be **short (1-3 sentences)**, factual, and derived **only** from the given document.\n"
            "- Do **not** invent information that is not present in the text.\n"
            "- Use clear, concise language suitable for inclusion in a search engine snippet.\n"
            "- The answer should **directly respond** to the question.\n"
            '- If the document does not contain enough information to answer a question fully, write `"Not specified in the document."`\n'
            "\n"
            "Provide output strictly following this JSON schema:\n"
            f"```\n{qa_json_schema}\n```\n"
            "\n"
            "Text:\n"
            "{context}"
        ),
        exclude=True,
    )

    prompt_qa_general_example: str = Field(
        default=(
            "You are helping to build a question answering system using retrieved documents.\n"
            "\n"
            "Given the following document, generate a list of **search queries and their corresponding answers** "
            "that a user might naturally type into Google or ask an assistant — assuming they know nothing "
            "about the content beforehand.\n"
            "\n"
            "Guidelines for QUESTIONS:\n"
            '- Each question must be fully **standalone**. NEVER refer to "the event", "this situation", '
            'or any pronouns like "it", "they", or "this". Always use explicit entities or locations from the text '
            '(e.g., "Shanghai", "airports in China").\n'
            "- Focus on **general**, natural-sounding questions that seek information — not paraphrases of sentences.\n"
            "- Use **search query style**, like something a real person might ask out of curiosity.\n"
            '- Prefer broad, investigative formulations such as **"what happened", "were there", "how did", "why did"**, '
            "reflecting interest in causes, effects, or outcomes.\n"
            "- Cover **different angles**: locations, organizations, transportation, public response, economic impact, etc.\n"
            "\n"
            "Guidelines for ANSWERS:\n"
            "- Each answer must be **short (1-3 sentences)**, factual, and derived **only** from the given document.\n"
            "- Do **not** invent information that is not present in the text.\n"
            "- Use clear, concise language suitable for inclusion in a search engine snippet.\n"
            "- The answer should **directly respond** to the question.\n"
            '- If the document does not contain enough information to answer a question fully, write `"Not specified in the document."`\n'
            "\n"
            "Provide output strictly following this JSON schema:\n"
            f"```\n{qa_json_schema}\n```\n"
            "\n"
            "Example:\n"
            "If the document discusses evacuations, flight delays, and emergency response in China, the output should look like this:\n"
            f"```\n{qa_general_example}\n```\n"
            "\n"
            "Text:\n"
            "{context}"
        ),
        exclude=True,
    )

    prompt_sequential_question: str = Field(
        default=(
            "Generate numerous questions to properly capture all the important parts of the text.\n"
            f"Answer in this json schema:\n{q_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )

    prompt_sequential_question_complex: str = Field(
        default=(
            "Read the following text and generate numerous factual questions "
            "designed to resemble authentic user search queries and natural language variations. "
            "Each question should accurately and semantically capture important aspects of the text, "
            "with varying lengths and complexities that mirror real-world search patterns. "
            "Include both shorter, keyword-focused questions such as 'who founded Tesla Motors' and longer, "
            "natural style questions like 'when did Elon Musk first start Tesla company'. "
            "Incorporate 'how' and 'why' questions to reflect genuine user curiosity. "
            "Avoid using phrases like 'according to the text' and abstain from pronouns by specifying names or entities. "
            "Ensure questions are not overly formal or artificial, maintaining a natural query style.\n"
            f"Provide output strictly following this JSON schema:\n{q_json_schema}\n\n"
            "Text:\n{context}"
        ),
        exclude=True,
    )

    prompt_sequential_answer: str = Field(
        default=(
            "Based on the following retrieved context, provide a comprehensive answer to the questions. "
            "Never include direct references to the text, like 'according to the text'. "
            "If the context doesn't contain enough information to answer the question, say so. "
            "Each question should be one sentence long. Never use pronouns in questions; instead, use the actual names or entities.\n"
            f"Provide output strictly following this JSON schema:\n{a_json_schema}\n\n"
            "Context:\n{context}\n\n"
            "Questions:\n{context_questions}\n\n"
            "Answers:"
        ),
        exclude=True,
    )

    def get_prompt(
        self,
        prompt_name: str,
    ) -> Prompt:
        """
        Get a prompt by name. If not provided, use a generic default.

        Behavior:
        - If the provided name matches a field on this settings object (with or without the
          `prompt_` prefix), return that text.
        - Otherwise, assume the prompt is stored in the DB and return an empty text with the
          resolved name (without the `prompt_` prefix).
        - If no name is provided, default to `prompt_qa_general`.
        """
        attr_name = (
            prompt_name
            if prompt_name.startswith("prompt_")
            else f"prompt_{prompt_name}"
        )

        try:
            text = getattr(self, attr_name)
            return Prompt(text=text, name=attr_name)
        except AttributeError:
            # Fallback: treat the provided name as a DB-stored prompt.
            raw_name = (
                prompt_name[len("prompt_") :]
                if prompt_name.startswith("prompt_")
                else prompt_name
            )
            logger.warning(f"Prompt {raw_name} not found in code, using DB prompt")
            return Prompt(text="", name=raw_name)
