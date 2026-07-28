---
title: Sketch-Based 2D Image Retrieval
emoji: ✏️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Draw a sketch and retrieve matching images via CLIP embeddings and graph database queries
---

# Sketch-Based 2D Image Retrieval

An interactive computer vision application that performs **sketch-to-image retrieval** using CLIP embeddings and a graph database query pipeline.

## How It Works

1. **Draw** a sketch on the canvas (cat, house, bicycle, etc.)
2. **CLIP** encodes your sketch into a normalized 512-D embedding vector
3. **Vocabulary matching** finds the closest concepts from a curated QuickDraw noun list
4. **Cypher query** is generated for a Neo4j HNSW vector index search
5. **Images** matching your sketch concept are retrieved and displayed

## Architecture

- **Frontend**: Vanilla HTML/CSS/JS with drawing canvas, brush tools, undo, and eraser
- **Backend**: Flask (Python) with Server-Sent Events (SSE) for streaming pipeline progress
- **Embedding Model**: `openai/clip-vit-base-patch32` via HuggingFace Transformers
- **Graph DB**: Neo4j Cypher query generation (mock execution with Flickr image retrieval)
- **Vocab**: 433 QuickDraw-specific sketch concepts, pre-embedded and cached

## Features

- 🎨 Drawing tools: pen, eraser, brush size, color swatches, undo (Ctrl+Z)
- ⚡ Real-time SSE streaming — each pipeline step updates as it completes
- 📊 Recognition confidence bar (Low / Medium / High)
- 🏷️ Color-coded concept pill badges
- 🖼️ Skeleton shimmer loading states + broken image fallback
