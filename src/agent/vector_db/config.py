import os

from dotenv import load_dotenv

load_dotenv()

attributes = ["content_id", "keywords", "article_index"]
index_name = "wix_qa_expert_ds"
text_key = "summary"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
