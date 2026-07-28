import gradio as gr
import spaces
import torch
import torch.nn.functional as F
import urllib.parse
import random
from PIL import Image

from core.clip_adapter import ClipAdapter
from core.llm_translator import LLMQueryTranslator
from graph_db.connector import GraphDBConnector

# ─────────────────────────────────────────────────────────────
# Initialize pipeline components (CPU — model weights loaded here)
# ─────────────────────────────────────────────────────────────
print("Initializing Pipeline Components...")
clip_adapter = ClipAdapter()
llm_translator = LLMQueryTranslator(clip_adapter=clip_adapter)
graph_db = GraphDBConnector()
print("Components Ready!")


# ─────────────────────────────────────────────────────────────
# GPU-accelerated inference step (ZeroGPU: allocated on demand)
# ─────────────────────────────────────────────────────────────
@spaces.GPU
def encode_sketch(sketch_image: Image.Image) -> torch.Tensor:
    """Runs CLIP encoding on GPU via ZeroGPU."""
    return clip_adapter.encode_image(sketch_image)


# ─────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────
def process_sketch(sketch_data):
    """
    Takes a sketch from gr.Sketchpad and runs the full retrieval pipeline.
    Returns: concepts_html, confidence_html, translation, cypher_query, image_urls
    """
    if sketch_data is None:
        return _empty_state()

    # gr.Sketchpad returns a dict with 'background' and 'layers' in Gradio 4+
    # or a PIL Image / numpy array in older versions
    if isinstance(sketch_data, dict):
        # Composite layers onto background
        composite = sketch_data.get("composite") or sketch_data.get("background")
        if composite is None:
            return _empty_state()
        pil_image = Image.fromarray(composite).convert("RGB")
    elif hasattr(sketch_data, "convert"):
        pil_image = sketch_data.convert("RGB")
    else:
        import numpy as np
        pil_image = Image.fromarray(sketch_data).convert("RGB")

    # 1. CLIP Embedding (GPU via ZeroGPU)
    clip_embedding = encode_sketch(pil_image)

    # 2. Concepts + Confidence
    concepts = llm_translator._get_nearest_neighbors(clip_embedding)
    confidence = llm_translator.get_confidence(clip_embedding)

    # 3. Translation (direct mode — top concepts joined)
    translation = llm_translator.translate_vector(clip_embedding)

    # 4. Cypher Query
    cypher_query = graph_db.early_fusion_search(
        query_vector=clip_embedding.tolist(),
        metadata_constraints=None,
        top_k=5
    )

    # 5. Retrieve images
    image_results = graph_db.mock_execute_query(cypher_query, concepts=concepts)

    # Format outputs
    concepts_html = _render_concept_pills(concepts)
    confidence_html = _render_confidence(confidence)
    image_urls = [(r["url"], f"Score: {r['score']}") for r in image_results]

    return concepts_html, confidence_html, translation, cypher_query, image_urls


def _empty_state():
    return (
        "<em style='color:#64748b'>Draw something first...</em>",
        "",
        "—",
        "—",
        []
    )


def _render_concept_pills(concepts):
    colors = [
        ("#3b82f6", "#93c5fd"),
        ("#10b981", "#6ee7b7"),
        ("#8b5cf6", "#d8b4fe"),
        ("#f59e0b", "#fcd34d"),
        ("#ef4444", "#fca5a5"),
    ]
    pills = ""
    for i, concept in enumerate(concepts):
        bg, text = colors[i % len(colors)]
        pills += (
            f'<span style="display:inline-block;padding:4px 12px;margin:3px;'
            f'border-radius:999px;background:{bg}22;border:1px solid {bg}66;'
            f'color:{text};font-weight:600;font-size:0.85rem">{concept}</span>'
        )
    return pills


def _render_confidence(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.6:
        color = "#10b981"
        label = f"🟢 High confidence ({pct}%)"
    elif score >= 0.3:
        color = "#f59e0b"
        label = f"🟡 Medium confidence ({pct}%)"
    else:
        color = "#ef4444"
        label = f"🔴 Low confidence — add more detail ({pct}%)"

    bar = (
        f'<div style="background:#1e293b;border-radius:999px;height:10px;overflow:hidden;margin-bottom:6px">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:999px;'
        f'transition:width 0.5s ease"></div></div>'
        f'<span style="font-size:0.82rem;font-weight:600;color:{color}">{label}</span>'
    )
    return bar


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
.container { max-width: 1100px; margin: auto; }
#title { text-align: center; margin-bottom: 0.5rem; }
#title h1 {
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
}
#title p { color: #94a3b8; margin-top: 0; }
.panel {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem;
    backdrop-filter: blur(12px);
}
.code-block {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.75rem;
    font-family: monospace;
    font-size: 0.82rem;
    color: #34d399;
    white-space: pre-wrap;
    word-break: break-all;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Sketch-Based 2D Image Retrieval") as demo:

    gr.HTML("""
        <div id="title">
            <h1>✏️ Sketch-Based 2D Image Retrieval</h1>
            <p>Draw a concept → CLIP encodes it → Graph database query → Real image results</p>
        </div>
    """)

    with gr.Row(equal_height=False):
        # Left column: Drawing canvas
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### 🎨 Input Sketch")
            sketch_input = gr.Sketchpad(
                label="Draw here",
                type="pil",
                brush=gr.Brush(default_size=4, colors=["#000000", "#ef4444", "#3b82f6", "#10b981"]),
                height=400,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(sketch_input, value="🗑️ Clear")
                search_btn = gr.Button("🔍 Process Sketch", variant="primary")

        # Right column: Pipeline output
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### ⚡ Pipeline")

            gr.Markdown("**1. Extracted Concepts**")
            concepts_out = gr.HTML('<em style="color:#64748b">Awaiting input...</em>')

            gr.Markdown("**Recognition Confidence**")
            confidence_out = gr.HTML("")

            gr.Markdown("**2. Translation**")
            translation_out = gr.Textbox(
                label="", interactive=False,
                placeholder="Awaiting input...",
                show_label=False
            )

            gr.Markdown("**3. Generated Cypher Query**")
            cypher_out = gr.Code(
                label="", language="sql", interactive=False,
                show_label=False
            )

    # Bottom: Image gallery
    gr.Markdown("### 🖼️ Visual Matches")
    gallery_out = gr.Gallery(
        label="Retrieved Images",
        show_label=False,
        columns=5,
        height=280,
        object_fit="cover",
        allow_preview=True,
    )

    # Wire up the button
    search_btn.click(
        fn=process_sketch,
        inputs=[sketch_input],
        outputs=[concepts_out, confidence_out, translation_out, cypher_out, gallery_out],
    )

    gr.Markdown(
        "<p style='text-align:center;color:#475569;font-size:0.8rem;margin-top:1rem'>"
        "Pipeline: CLIP (openai/clip-vit-base-patch32) → QuickDraw vocab matching → "
        "Neo4j Cypher generation → Flickr image retrieval"
        "</p>"
    )


if __name__ == "__main__":
    demo.launch()
