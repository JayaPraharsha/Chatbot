from db.access import get_content_by_id, get_title_by_id
from llms import content_extraction_llm
from nltk.tokenize import sent_tokenize
from prompts.content_extraction import extraction_prompt
from schemas import ContentExtractionSchema

content_extraction_llm = content_extraction_llm.with_structured_output(ContentExtractionSchema)
content_extraction_chain = extraction_prompt | content_extraction_llm


def convert_to_sentences(content: str):
    sentences = sent_tokenize(content.strip())
    numbered_content = [f"{i + 1}. {s}" for i, s in enumerate(sentences)]
    return numbered_content


def extract_content(content_id: str, query: str) -> tuple:
    extracted_content = get_title_by_id(content_id) + " \n "
    content = get_content_by_id(id)
    numbered_content = convert_to_sentences(content)
    result = content_extraction_chain.invoke(
        {"query": query, "numbered_sentences": numbered_content}
    )

    for span in result.extracted_contents:
        start_idx = span.start_idx
        end_idx = span.end_idx
        if start_idx != -1 and end_idx != -1:
            for sentence in numbered_content[start_idx - 1 : end_idx]:
                extracted_content += sentence + " \n "

    return extracted_content
