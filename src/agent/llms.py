from langchain_openai import ChatOpenAI

query_expander_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

reranker_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

content_extraction_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

generator_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# evaluator_llm = ChatOpenAI(model="gpt-4.1-mini")

factuality_evalutator_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
context_recall_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
