from langchain_openai import ChatOpenAI

query_expander_llm = ChatOpenAI(model="gpt-4.1-mini")

reranker_llm = ChatOpenAI(model="gpt-4.1-mini")

content_extraction_llm = ChatOpenAI(model="gpt-4.1-mini")

generator_llm = ChatOpenAI(model="gpt-4")

evaluator_llm = ChatOpenAI(model="gpt-4")
# factuality_evalutator_llm = ChatOpenAI(model="gpt-4.1-mini")
# context_recall_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini")
