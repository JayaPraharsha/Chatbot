from llms import generator_llm
from prompts.generation import generation_prompt

generation_chain = generation_prompt | generator_llm


def generate_answer(args_map: map) -> str:
    return generation_chain.invoke(args_map).content
