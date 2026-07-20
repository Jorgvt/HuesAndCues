// App Visualizer Frontend JavaScript Logic

let boardData = null;
let wordsList = [];
let currentWordIndex = 0;
let currentWordData = null;
let onlyColorblindFilter = false;

// Display settings
let showHeatmap = true;
let showCounts = true;
let showCBMarkers = true;

// DOM Elements
const wordSelect = document.getElementById('word-select');
const filterAllBtn = document.getElementById('filter-all');
const filterCbBtn = document.getElementById('filter-cb');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const randomBtn = document.getElementById('random-btn');

const toggleHeatmap = document.getElementById('toggle-heatmap');
const toggleCounts = document.getElementById('toggle-counts');
const toggleCb = document.getElementById('toggle-cb');

const wordTitle = document.getElementById('word-title');
const wordCategory = document.getElementById('word-category');
const totalPicksVal = document.getElementById('total-picks-val');
const cbPicksVal = document.getElementById('cb-picks-val');
const consensusVal = document.getElementById('consensus-val');
const consensusSwatch = document.getElementById('consensus-swatch');

const cbListContainer = document.getElementById('cb-list-container');
const boardGrid = document.getElementById('board-grid');
const cellTooltip = document.getElementById('cell-tooltip');

// Initialize App
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await fetchBoardData();
    await fetchWordsList();
    if (wordsList.length > 0) {
        await loadWord(wordsList[0].word);
    }
});

function setupEventListeners() {
    // Word filter tabs
    filterAllBtn.addEventListener('click', () => {
        if (!onlyColorblindFilter) return;
        onlyColorblindFilter = false;
        filterAllBtn.classList.add('active');
        filterCbBtn.classList.remove('active');
        fetchWordsList();
    });

    filterCbBtn.addEventListener('click', () => {
        if (onlyColorblindFilter) return;
        onlyColorblindFilter = true;
        filterCbBtn.classList.add('active');
        filterAllBtn.classList.remove('active');
        fetchWordsList();
    });

    // Word dropdown change
    wordSelect.addEventListener('change', (e) => {
        const selectedWord = e.target.value;
        if (selectedWord) {
            loadWord(selectedWord);
        }
    });

    // Navigation buttons
    prevBtn.addEventListener('click', () => {
        if (currentWordIndex > 0) {
            currentWordIndex--;
            wordSelect.selectedIndex = currentWordIndex;
            loadWord(wordsList[currentWordIndex].word);
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentWordIndex < wordsList.length - 1) {
            currentWordIndex++;
            wordSelect.selectedIndex = currentWordIndex;
            loadWord(wordsList[currentWordIndex].word);
        }
    });

    randomBtn.addEventListener('click', () => {
        if (wordsList.length > 0) {
            const randomIndex = Math.floor(Math.random() * wordsList.length);
            currentWordIndex = randomIndex;
            wordSelect.selectedIndex = currentWordIndex;
            loadWord(wordsList[currentWordIndex].word);
        }
    });

    // Keyboard Arrow Keys
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowLeft') {
            prevBtn.click();
        } else if (e.key === 'ArrowRight') {
            nextBtn.click();
        }
    });

    // Toggles
    toggleHeatmap.addEventListener('change', (e) => {
        showHeatmap = e.target.checked;
        updateBoardOverlay();
    });

    toggleCounts.addEventListener('change', (e) => {
        showCounts = e.target.checked;
        updateBoardOverlay();
    });

    toggleCb.addEventListener('change', (e) => {
        showCBMarkers = e.target.checked;
        updateBoardOverlay();
    });
}

// Fetch Board Structure (480 cells)
async function fetchBoardData() {
    try {
        const response = await fetch('/api/board');
        boardData = await response.json();
        renderBoardGrid();
    } catch (err) {
        console.error('Error fetching board data:', err);
    }
}

// Render Board Grid Skeleton
function renderBoardGrid() {
    if (!boardData) return;
    boardGrid.innerHTML = '';

    const { rows, cols, cells } = boardData;

    // Top Header Row (Labels 1 to 30)
    const emptyTopLeft = document.createElement('div');
    emptyTopLeft.className = 'axis-label';
    boardGrid.appendChild(emptyTopLeft);

    cols.forEach(col => {
        const label = document.createElement('div');
        label.className = 'axis-label';
        label.textContent = col;
        boardGrid.appendChild(label);
    });

    // 16 Grid Rows
    rows.forEach(row => {
        // Left Axis Label (A-P)
        const rowLabel = document.createElement('div');
        rowLabel.className = 'axis-label';
        rowLabel.textContent = row;
        boardGrid.appendChild(rowLabel);

        // 30 Cells for this row
        cols.forEach(col => {
            const coord = `${row}${col}`;
            const cellInfo = boardData.cell_map[coord];

            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${coord}`;
            cell.dataset.coord = coord;
            cell.style.backgroundColor = cellInfo.hex;

            // Overlay container for heatmap & count
            const overlay = document.createElement('div');
            overlay.className = 'cell-overlay';
            cell.appendChild(overlay);

            // Mouse hover tooltip events
            cell.addEventListener('mouseenter', (e) => handleCellHover(e, coord));
            cell.addEventListener('mouseleave', () => handleCellLeave());
            cell.addEventListener('mousemove', (e) => positionTooltip(e));

            boardGrid.appendChild(cell);
        });
    });

    // Bottom Header Row (Labels 1 to 30)
    const emptyBottomLeft = document.createElement('div');
    emptyBottomLeft.className = 'axis-label';
    boardGrid.appendChild(emptyBottomLeft);

    cols.forEach(col => {
        const label = document.createElement('div');
        label.className = 'axis-label';
        label.textContent = col;
        boardGrid.appendChild(label);
    });
}

// Fetch Words List for Dropdown
async function fetchWordsList() {
    try {
        const url = `/api/words?only_cb=${onlyColorblindFilter}`;
        const response = await fetch(url);
        wordsList = await response.json();
        
        // Populate dropdown
        wordSelect.innerHTML = '';
        wordsList.forEach((w, idx) => {
            const opt = document.createElement('option');
            opt.value = w.word;
            const cbBadge = w.has_colorblind ? ` (★ ${w.colorblind_responses} CB)` : '';
            opt.textContent = `${w.word} [${w.category}] - ${w.total_responses} picks${cbBadge}`;
            wordSelect.appendChild(opt);
        });

        // Set index
        currentWordIndex = 0;
        if (wordsList.length > 0) {
            wordSelect.selectedIndex = 0;
        }
    } catch (err) {
        console.error('Error fetching words list:', err);
    }
}

// Load full data for a single word
async function loadWord(wordName) {
    try {
        const response = await fetch(`/api/word/${encodeURIComponent(wordName)}`);
        if (!response.ok) return;
        
        currentWordData = await response.json();

        // Update currentWordIndex
        const idx = wordsList.findIndex(w => w.word === wordName);
        if (idx !== -1) {
            currentWordIndex = idx;
            wordSelect.selectedIndex = idx;
        }

        updateUI();
    } catch (err) {
        console.error('Error loading word:', err);
    }
}

// Update UI Components
function updateUI() {
    if (!currentWordData) return;

    // Word Header & Summary Card
    wordTitle.textContent = currentWordData.word;
    wordCategory.textContent = currentWordData.category || 'CATEGORY';
    totalPicksVal.textContent = currentWordData.total_responses;
    cbPicksVal.textContent = currentWordData.colorblind_responses_count;

    if (currentWordData.top_coords && currentWordData.top_coords.length > 0) {
        const top = currentWordData.top_coords[0];
        consensusVal.textContent = `${top.coordinate} (${top.count} picks, ${top.percentage}%)`;
        consensusSwatch.style.backgroundColor = top.hex;
    } else {
        consensusVal.textContent = 'N/A';
        consensusSwatch.style.backgroundColor = 'transparent';
    }

    // Colorblind Users List Breakdown
    renderColorblindList();

    // Board Grid Overlay (Heatmap & CB Markers)
    updateBoardOverlay();
}

// Update Heatmap and Markers on Board
function updateBoardOverlay() {
    if (!boardData || !currentWordData) return;

    const { counts_per_coord, max_count, colorblind_responses } = currentWordData;
    const cbCoordSet = new Set(colorblind_responses.map(r => r.coordinate));

    Object.keys(boardData.cell_map).forEach(coord => {
        const cell = document.getElementById(`cell-${coord}`);
        if (!cell) return;

        const overlay = cell.querySelector('.cell-overlay');
        overlay.innerHTML = '';
        overlay.style.backgroundColor = 'transparent';

        // Remove old cb marker if exists
        const existingCbMarker = cell.querySelector('.cb-marker');
        if (existingCbMarker) {
            existingCbMarker.remove();
        }

        const count = counts_per_coord[coord] || 0;

        // 1. Heatmap Overlay
        if (showHeatmap && count > 0) {
            const intensity = count / max_count;
            // Linear heat color from soft white glow to vibrant crimson/indigo
            const alpha = 0.25 + (intensity * 0.55);
            overlay.style.backgroundColor = `rgba(99, 102, 241, ${alpha})`;
            overlay.style.boxShadow = `inset 0 0 4px rgba(255, 255, 255, ${intensity})`;
        }

        // 2. Count Badges
        if (showCounts && count > 0) {
            const badge = document.createElement('span');
            badge.className = 'cell-count-badge';
            badge.textContent = count;
            overlay.appendChild(badge);
        }

        // 3. Colorblind Marker
        if (showCBMarkers && cbCoordSet.has(coord)) {
            const cbMarker = document.createElement('div');
            cbMarker.className = 'cb-marker';
            cell.appendChild(cbMarker);
        }
    });
}

// Render Colorblind Responses List Panel
function renderColorblindList() {
    cbListContainer.innerHTML = '';
    const cbResponses = currentWordData.colorblind_responses;

    if (!cbResponses || cbResponses.length === 0) {
        cbListContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.5rem 0;">No colorblind user responses recorded for this word.</div>';
        return;
    }

    cbResponses.forEach(res => {
        const card = document.createElement('div');
        card.className = 'cb-user-card';

        card.innerHTML = `
            <div class="cb-user-info">
                <div class="cb-user-id">User: ${res.short_userId}</div>
                <div class="cb-user-meta">Gender: ${res.gender} | Age: ${res.ageRange}</div>
            </div>
            <div class="cb-user-pick">
                <span class="coord-tag">${res.coordinate}</span>
                <span class="color-swatch-sm" style="background-color: ${res.hex};" title="RGB: ${res.rgb.join(', ')}"></span>
            </div>
        `;

        // Highlight cell on card hover
        card.addEventListener('mouseenter', () => {
            const cell = document.getElementById(`cell-${res.coordinate}`);
            if (cell) {
                cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                cell.style.transform = 'scale(1.5)';
                cell.style.zIndex = '100';
            }
        });

        card.addEventListener('mouseleave', () => {
            const cell = document.getElementById(`cell-${res.coordinate}`);
            if (cell) {
                cell.style.transform = '';
                cell.style.zIndex = '';
            }
        });

        cbListContainer.appendChild(card);
    });
}

// Tooltip Handling
function handleCellHover(event, coord) {
    if (!boardData || !currentWordData) return;

    const cellInfo = boardData.cell_map[coord];
    const totalPicks = currentWordData.counts_per_coord[coord] || 0;
    const totalWordPicks = currentWordData.total_responses;
    const pct = totalWordPicks > 0 ? ((totalPicks / totalWordPicks) * 100).toFixed(1) : 0;

    const cbPicks = currentWordData.colorblind_responses.filter(r => r.coordinate === coord);

    cellTooltip.innerHTML = `
        <div class="tooltip-header">
            <span>Coordinate: ${coord}</span>
            <span class="color-swatch-sm" style="background-color: ${cellInfo.hex}"></span>
        </div>
        <div class="tooltip-row"><span>RGB:</span> <span class="val">(${cellInfo.r}, ${cellInfo.g}, ${cellInfo.b})</span></div>
        <div class="tooltip-row"><span>Hex:</span> <span class="val">${cellInfo.hex}</span></div>
        <div class="tooltip-row"><span>Human Picks:</span> <span class="val">${totalPicks} (${pct}%)</span></div>
        <div class="tooltip-row"><span>Colorblind Picks:</span> <span class="val" style="color: var(--cb-gold);">${cbPicks.length}</span></div>
    `;

    cellTooltip.style.display = 'block';
    positionTooltip(event);
}

function positionTooltip(event) {
    const tooltipWidth = 200;
    const tooltipHeight = 130;
    
    let left = event.pageX + 15;
    let top = event.pageY + 15;

    if (left + tooltipWidth > window.innerWidth) {
        left = event.pageX - tooltipWidth - 15;
    }

    if (top + tooltipHeight > window.innerHeight + window.scrollY) {
        top = event.pageY - tooltipHeight - 15;
    }

    cellTooltip.style.left = `${left}px`;
    cellTooltip.style.top = `${top}px`;
}

function handleCellLeave() {
    cellTooltip.style.display = 'none';
}
