const canvas = document.getElementById('sketch-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const searchBtn = document.getElementById('search-btn');
const statusInd = document.getElementById('status-indicator');

const embedOut = document.getElementById('embed-out');
const llmOut = document.getElementById('llm-out');
const cypherOut = document.getElementById('cypher-out');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas with white background
function initCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

// Drawing logic
function draw(e) {
    if (!isDrawing) return;
    
    // Get mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.stroke();

    lastX = x;
    lastY = y;
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = e.clientX - rect.left;
    lastY = e.clientY - rect.top;
});

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// Touch support
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = e.touches[0].clientX - rect.left;
    lastY = e.touches[0].clientY - rect.top;
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
});
canvas.addEventListener('touchend', () => isDrawing = false);

// Clear Canvas
clearBtn.addEventListener('click', initCanvas);

// Send to API
searchBtn.addEventListener('click', async () => {
    statusInd.className = 'status processing';
    statusInd.textContent = 'Processing...';
    
    embedOut.textContent = 'Extracting features...';
    llmOut.textContent = 'Waiting for translation...';
    cypherOut.textContent = 'Waiting for constraints...';

    const imageData = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success) {
            statusInd.className = 'status done';
            statusInd.textContent = 'Complete';
            
            embedOut.textContent = `Extracted Vector Shape: [${data.embedding_shape.join(', ')}]`;
            llmOut.textContent = `[Prompt Generated]\n${data.concepts_prompt}\n\n[Local LLM Translation]\n${data.translation}`;
            cypherOut.textContent = data.cypher_query;
            
            // Render Image Grid
            const imageGrid = document.getElementById('image-grid');
            imageGrid.innerHTML = ''; // Clear previous
            
            if (data.results && data.results.length > 0) {
                data.results.forEach(result => {
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    card.innerHTML = `
                        <img src="${result.url}" alt="${result.title}">
                        <div class="image-card-overlay">
                            <span class="card-title">${result.title}</span>
                            <span class="card-score">Score: ${result.score}</span>
                        </div>
                    `;
                    imageGrid.appendChild(card);
                });
            } else {
                imageGrid.innerHTML = '<div class="empty-state">No images matched the query.</div>';
            }
            
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        statusInd.className = 'status idle';
        statusInd.textContent = 'Error';
        console.error(err);
        cypherOut.textContent = `Error processing sketch: ${err.message}`;
    }
});
