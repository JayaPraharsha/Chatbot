from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_message = SystemMessagePromptTemplate.from_template("""
You are a document reranker.  You receive a user query, and and five Document summaries.
You rerank documents based on their relevance to the user query. 
Your job is to rank these documents from most to least relevant.
Assign the value 1000, if a document is irrelevant.
""")

human_message = HumanMessagePromptTemplate.from_template("""
Here is the user query: \n
{query}
\n 
Rerank the given documents based on user query. Think Step-by-Step. Remember, you need to justify the order of the documents that you output.
1. DOCUMENT - 1
{Document_1}

                                                     
2. DOCUMENT - 2
{Document_2}


3. DOCUMENT - 3
{Document_3}
                                                     

4. DOCUMENT - 4
{Document_4}


5. DOCUMENT - 5
{Document_5}
                                                         

6. DOCUMENT - 6
{Document_6}


7. DOCUMENT - 7
{Document_7}  


8. DOCUMENT - 8  
{Document_8}


9. DOCUMENT - 9
{Document_9}
                                                         

10. DOCUMENT - 10
{Document_10}                            
""")


reranker_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
