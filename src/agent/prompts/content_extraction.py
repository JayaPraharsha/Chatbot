from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_message = SystemMessagePromptTemplate.from_template("""
You are an intelligent assistant that extracts relevant information from text based on a user's query.

You are given a document that has been pre-processed into sentences, each assigned a unique number (starting from 0). Sentence i+1 immediately follows sentence i — all sentences are in sequential order.

Your task is to identify which sentence(s) or passage(s) are relevant to the user's query and return a structured output:

- For a relevant passage (multiple sentences), return (start_index, end_index), inclusive.
- For a single relevant sentence, return (index, index).
- Return (-1,-1) if there is no relevant content.
- Only include sentences relevant to the query.
- Do not include explanation, only the structured output.
- Make sure to include links to other articles, specific Wix resources, or pertinent external resources. 
- You must not return an empty list. Return (-1,-1) if there is no relevant content.                                                  
""")

# Human message
human_message = HumanMessagePromptTemplate.from_template("""
User Query: {query}

Sentences:
{numbered_sentences}
""")


extraction_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
