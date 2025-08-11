from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_message = SystemMessagePromptTemplate.from_template("""
You are a search query expander. Use the provided background context from top-k retrieved documents—including titles and keywords—to rewrite the user's original query. 
Your goal is to append relevant keywords from the documents to the original user query. Don't change the original user query.
You have the flexibility to choose the number of keywords to add to the original user query.
Don't include keywords from all the documents. Your goal is to filter only the relevant documents, and derive context and keywords from them, and include them in the user query.
Don't add noise in the user query.
Do not invent any new keywords outside the context. If you think that you can create new keywords/phrases, rely only on the document content.
Output ONLY the expanded query and the justification.
""")


human_message = HumanMessagePromptTemplate.from_template("""
Here is the user query: \n
{query}
\n 
Expand the user query. Think Step-by-Step. Remember, you need to justify the expansion.
1. DOCUMENT - 1
{Document_1_Summary}
{Document_1_Keywords}
                                                         

                                                     
2. DOCUMENT - 2
{Document_2_Summary}
{Document_2_Keywords}


3. DOCUMENT - 3
{Document_3_Summary}
{Document_3_Keywords}
                                                     

4. DOCUMENT - 4
{Document_4_Summary}
{Document_4_Keywords}


5. DOCUMENT - 5
{Document_5_Summary}
{Document_5_Keywords}


6. DOCUMENT - 6
{Document_6_Summary}
{Document_6_Keywords}
                                                         

7. DOCUMENT - 7
{Document_7_Summary}
{Document_7_Keywords}
                                                         

8. DOCUMENT - 8
{Document_8_Summary}
{Document_8_Keywords}

                                                         
9. DOCUMENT - 9
{Document_9_Summary}
{Document_9_Keywords}
                                                         

10. DOCUMENT - 10
{Document_10_Summary}
{Document_10_Keywords}
""")


query_expansion_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
