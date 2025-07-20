import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import gradio as gr

# ✅ Load environment variables
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

# ✅ Load book data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ✅ Load text file manually and wrap in Document
with open("tagged_description.txt", "r", encoding="utf-8") as f:
    content = f.read()
raw_documents = [Document(page_content=content)]

# ✅ Split documents (NO filtering needed)
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# ✅ Create vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
db_books = Chroma.from_documents(documents, embeddings)

# ✅ Book Recommendation Logic
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# ✅ Format results for Gradio
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            author_str = row["authors"]

        caption = f"{row['title']} by {author_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# ✅ Gradio UI
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description", placeholder="e.g. A story about friendship and courage")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotion", value="All")
        submit_button = gr.Button("Recommend")

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()
