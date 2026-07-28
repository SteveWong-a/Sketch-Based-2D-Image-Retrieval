const canvas = document.getElementById('sketch-canvas');
const ctx = canvas.getContext('2d');

// --- Controls ---
const clearBtn = document.getElementById('clear-btn');
const searchBtn = document.getElementById('search-btn');
const undoBtn = document.getElementById('undo-btn');
const eraserBtn = document.getElementById('eraser-btn');
const penBtn = document.getElementById('pen-btn');
const brushSizeInput = document.getElementById('brush-size');
const brushSizeDisplay = document.getElementById('brush-size-display');
const statusInd = document.getElementById('status-indicator');
const embedOut = document.getElementById('embed-out');
const llmOut = document.getElementById('llm-out');
const cypherOut = document.getElementById('cypher-out');
const conceptTagsEl = document.getElementById('concept-tags');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceLabel = document.getElementById('confidence-label');
const galleryLabel = document.getElementById('gallery-query-label');

// --- Drawing state ---
let isDrawing = false;
let lastX = 0, lastY = 0;
let isEraser = false;
let currentColor = '#000000';
let brushSize = 3;
let strokeHistory = [];

// --- Canvas init ---
function initCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokeHistory = [];
}
initCanvas();

function saveStroke() {
    strokeHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (strokeHistory.length > 40) strokeHistory.shift();
}

// --- Brush size ---
brushSizeInput.addEventListener('input', () => {
    brushSize = parseInt(brushSizeInput.value);
    brushSizeDisplay.textContent = brushSize;
});

// --- Color swatches ---
document.querySelectorAll('.swatch').forEach(swatch => {
    swatch.addEventListener('click', () => {
        document.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
        swatch.classList.add('active');
        currentColor = swatch.dataset.color;
        setDrawMode();
    });
});

// --- Eraser / Pen toggle ---
function setDrawMode() {
    isEraser = false;
    eraserBtn.classList.remove('active');
    penBtn.classList.add('active');
    canvas.style.cursor = 'crosshair';
}

function setEraserMode() {
    isEraser = true;
    eraserBtn.classList.add('active');
    penBtn.classList.remove('active');
    canvas.style.cursor = 'cell';
}

penBtn.addEventListener('click', setDrawMode);
eraserBtn.addEventListener('click', setEraserMode);

// --- Undo ---
undoBtn.addEventListener('click', undo);
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo(); }
});
function undo() {
    if (strokeHistory.length > 0) ctx.putImageData(strokeHistory.pop(), 0, 0);
}

// --- Drawing logic ---
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches) return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function draw(e) {
    if (!isDrawing) return;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    if (isEraser) {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0,0,0,1)';
        ctx.lineWidth = brushSize * 3;
    } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = currentColor;
        ctx.lineWidth = brushSize;
    }
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    lastX = x; lastY = y;
}

canvas.addEventListener('mousedown', (e) => { saveStroke(); isDrawing = true; const {x,y} = getPos(e); lastX=x; lastY=y; });
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); saveStroke(); isDrawing = true; const {x,y} = getPos(e); lastX=x; lastY=y; });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', () => isDrawing = false);
clearBtn.addEventListener('click', initCanvas);

// --- Render concept tags ---
function renderConceptTags(concepts) {
    conceptTagsEl.innerHTML = '';
    if (!concepts || concepts.length === 0) {
        conceptTagsEl.innerHTML = '<span class="tag-placeholder">No concepts found</span>';
        return;
    }
    concepts.forEach((concept, i) => {
        const tag = document.createElement('span');
        tag.className = 'concept-tag';
        tag.style.animationDelay = `${i * 0.06}s`;
        tag.textContent = concept;
        conceptTagsEl.appendChild(tag);
    });
}

// --- Render confidence ---
function renderConfidence(score) {
    const pct = Math.round(score * 100);
    confidenceBar.style.width = `${pct}%`;

    if (score >= 0.6) {
        confidenceBar.style.background = 'linear-gradient(to right, #10b981, #34d399)';
        confidenceLabel.textContent = `🟢 High confidence (${pct}%)`;
        confidenceLabel.style.color = '#34d399';
    } else if (score >= 0.3) {
        confidenceBar.style.background = 'linear-gradient(to right, #f59e0b, #fcd34d)';
        confidenceLabel.textContent = `🟡 Medium confidence (${pct}%)`;
        confidenceLabel.style.color = '#fcd34d';
    } else {
        confidenceBar.style.background = 'linear-gradient(to right, #ef4444, #f87171)';
        confidenceLabel.textContent = `🔴 Low confidence — try adding more detail (${pct}%)`;
        confidenceLabel.style.color = '#f87171';
    }
}

// --- Skeleton shimmer cards ---
function renderSkeletons(count = 5) {
    const imageGrid = document.getElementById('image-grid');
    imageGrid.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const skel = document.createElement('div');
        skel.className = 'skeleton-card';
        imageGrid.appendChild(skel);
    }
}

// --- Render image cards ---
function renderImageCards(results) {
    const imageGrid = document.getElementById('image-grid');
    imageGrid.innerHTML = '';

    if (!results || results.length === 0) {
        imageGrid.innerHTML = '<div class="empty-state">No images matched the query.</div>';
        return;
    }

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'image-card';

        const img = document.createElement('img');
        img.alt = result.title;
        img.src = result.url;
        img.addEventListener('load', () => img.classList.add('loaded'));
        img.addEventListener('error', () => {
            img.style.display = 'none';
            const errDiv = document.createElement('div');
            errDiv.className = 'img-error';
            errDiv.innerHTML = `<span>🖼️</span><p>Image unavailable</p>`;
            card.appendChild(errDiv);
        });

        const overlay = document.createElement('div');
        overlay.className = 'image-card-overlay';
        overlay.innerHTML = `
            <span class="card-title">${result.title}</span>
            <span class="card-score">Score: ${result.score}</span>
        `;

        card.appendChild(img);
        card.appendChild(overlay);
        imageGrid.appendChild(card);
    });
}

// --- Reset UI state ---
function resetUI() {
    statusInd.className = 'status processing';
    statusInd.textContent = 'Processing...';
    embedOut.textContent = 'Extracting features...';
    llmOut.textContent = 'Waiting...';
    cypherOut.textContent = 'Waiting...';
    conceptTagsEl.innerHTML = '<span class="tag-placeholder">Analyzing...</span>';
    confidenceBar.style.width = '0%';
    confidenceLabel.textContent = '';
    galleryLabel.textContent = '';
    renderSkeletons(5);
}

// ──────────────────────────────────────────────────────────
// SSE Streaming: process sketch step by step
// ──────────────────────────────────────────────────────────
searchBtn.addEventListener('click', async () => {
    resetUI();
    searchBtn.disabled = true;

    const imageData = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/api/process/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const parts = buffer.split('\n\n');
            buffer = parts.pop(); // keep incomplete chunk

            for (const part of parts) {
                const eventLine = part.split('\n').find(l => l.startsWith('event:'));
                const dataLine = part.split('\n').find(l => l.startsWith('data:'));
                if (!eventLine || !dataLine) continue;

                const event = eventLine.replace('event:', '').trim();
                const payload = JSON.parse(dataLine.replace('data:', '').trim());

                if (event === 'clip_done') {
                    embedOut.textContent = `Extracted Vector Shape: [${payload.embedding_shape.join(', ')}]`;
                }

                if (event === 'concepts_done') {
                    renderConceptTags(payload.concepts);
                    renderConfidence(payload.confidence);
                }

                if (event === 'cypher_done') {
                    llmOut.textContent = payload.translation;
                    cypherOut.textContent = payload.cypher_query;
                    galleryLabel.textContent = `Showing results for: "${payload.translation}"`;
                }

                if (event === 'results_done') {
                    statusInd.className = 'status done';
                    statusInd.textContent = 'Complete';
                    renderImageCards(payload.results);
                }

                if (event === 'error') {
                    throw new Error(payload.message);
                }
            }
        }
    } catch (err) {
        statusInd.className = 'status idle';
        statusInd.textContent = 'Error';
        cypherOut.textContent = `Error: ${err.message}`;
        document.getElementById('image-grid').innerHTML = '<div class="empty-state">An error occurred. Please try again.</div>';
        console.error(err);
    } finally {
        searchBtn.disabled = false;
    }
});
