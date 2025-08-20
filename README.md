# Chatbot

I am very much excited about this project.
I love generative AI.

Currently I am in feature 1 branch.

Currently I am in feature 2 branch.


RUN python -m nltk.downloader -d /usr/local/nltk_data punkt punkt_tab

EXPOSE 8001
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
