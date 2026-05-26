# import gradio as gr
# import numpy as np
# import faiss

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# documents = []
# index = None


# def process_file(file):

#     global documents
#     global index

    
#     with open(file.name, "r", encoding="utf-8") as f:
#         text = f.read()

    
#     documents = [
#         line.strip()
#         for line in text.split("\n")
#         if line.strip()
#     ]

    
#     embeddings = model.encode(documents)

#     embeddings = np.array(embeddings).astype("float32")

   
#     dimension = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dimension)

#     index.add(embeddings)

#     return f"{len(documents)} documents uploaded successfully!"



# def semantic_search(query, top_k):

#     global documents
#     global index

    
#     if index is None:
#         return "Please upload a document first."

#     # Convert query to embedding
#     query_embedding = model.encode([query])

#     query_embedding = np.array(query_embedding).astype("float32")

#     # Search
#     distances, indices = index.search(query_embedding, top_k)

#     results = []

#     for i, idx in enumerate(indices[0]):

#         results.append(
#             f"""
# Result {i+1}

# Document:
# {documents[idx]}

# Distance:
# {distances[0][i]:.4f}
# """
#         )

#     return "\n".join(results)



# # GRADIO INTERFACE

# with gr.Blocks() as demo:

#     gr.Markdown("# Semantic Similarity Search")

#     gr.Markdown(
#         "Upload a text document and search using natural language."
#     )

#     file_input = gr.File(
#         label="Upload TXT File"
#     )

#     upload_output = gr.Textbox(
#         label="Upload Status"
#     )

#     upload_button = gr.Button("Process Document")

#     upload_button.click(
#         fn=process_file,
#         inputs=file_input,
#         outputs=upload_output
#     )

#     query_input = gr.Textbox(
#         label="Search Query",
#         placeholder="Example: machine learning"
#     )

#     top_k = gr.Slider(
#         minimum=1,
#         maximum=5,
#         value=3,
#         step=1,
#         label="Top K Results"
#     )

#     search_output = gr.Textbox(
#         label="Search Results",
#         lines=15
#     )

#     search_button = gr.Button("Search")

#     search_button.click(
#         fn=semantic_search,
#         inputs=[query_input, top_k],
#         outputs=search_output
#     )


# # Run app
# demo.launch()
import gradio as gr
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD MODEL
# ==========================================
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = []
index = None


# ==========================================
# PROCESS FILE
# ==========================================
def process_file(file):

    global documents
    global index

    with open(file.name, "r", encoding="utf-8") as f:
        text = f.read()

    documents = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    embeddings = model.encode(documents)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return f"{len(documents)} documents processed successfully."


# ==========================================
# SEMANTIC SEARCH
# ==========================================
def semantic_search(query, top_k):

    global documents
    global index

    if index is None:
        return "Please upload and process a document first."

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):

        results.append(
            f"""
Result {i+1}

Document:
{documents[idx]}

Distance Score:
{distances[0][i]:.4f}

----------------------------------------
"""
        )

    return "\n".join(results)


# ==========================================
# MINIMAL DARK THEME
# ==========================================
theme = gr.themes.Base().set(

    body_background_fill="#0b0f19",
    background_fill_primary="#0b0f19",

    block_background_fill="#111827",
    block_border_color="#1f2937",

    input_background_fill="#111827",
    input_border_color="#374151",

    button_primary_background_fill="#1f2937",
    button_primary_background_fill_hover="#374151",

    button_secondary_background_fill="#111827",

    color_accent_soft="#1f2937",

    body_text_color="#e5e7eb",
)


# ==========================================
# UI
# ==========================================
with gr.Blocks(
    theme=theme,
    title="Semantic Search",
    css="""
    .container {
        max-width: 850px;
        margin: auto;
        padding-top: 30px;
    }

    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }

    footer {
        visibility: hidden;
    }
    """
) as demo:

    with gr.Column(elem_classes="container"):

        gr.HTML(
            """
            <div class="title">
                Semantic Search
            </div>

            <div class="subtitle">
                Upload documents and search using natural language
            </div>
            """
        )

        with gr.Group():

            file_input = gr.File(
                label="Upload TXT File",
                file_types=[".txt"]
            )

            upload_button = gr.Button(
                "Process Document",
                variant="primary"
            )

            upload_output = gr.Textbox(
                label="Status",
                interactive=False
            )

        with gr.Group():

            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Search something...",
                lines=2
            )

            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Top Results"
            )

            search_button = gr.Button(
                "Search",
                variant="primary"
            )

            search_output = gr.Textbox(
                label="Results",
                lines=18
            )

    # ======================================
    # ACTIONS
    # ======================================
    upload_button.click(
        fn=process_file,
        inputs=file_input,
        outputs=upload_output
    )

    search_button.click(
        fn=semantic_search,
        inputs=[query_input, top_k],
        outputs=search_output
    )


# ==========================================
# RUN APP
# ==========================================
demo.launch()