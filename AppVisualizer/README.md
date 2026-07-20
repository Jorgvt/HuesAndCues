# Hues & Cues App Visualizer

An interactive web application built with Flask, HTML5, CSS3, and JavaScript for visualizing and analyzing human participant responses to the **Hues and Cues** test, with a specific focus on comparing general human responses against colorblind participants.

---

## Features

1. **Interactive Hues & Cues Board Grid (16 × 30)**:
   - Full 480-cell board rendered with exact RGB colors loaded from [HC_RGB.csv](file:///Users/jorgvt/Developer/HuesAndCues/HC_RGB.csv).
   - Coordinates labelled on axes (A–P rows, 1–30 columns).
   - Interactive cell tooltips detailing coordinate name, RGB/Hex values, human pick percentage, and colorblind user response status.

2. **Single Colorboard Overlay & Heatmap**:
   - **Pick Heatmap**: Visualizes response density across human participants per word.
   - **Colorblind Response Markers**: Highlights coordinates picked by colorblind users (from [colorblind_ids.txt](file:///Users/jorgvt/Developer/HuesAndCues/colorblind_ids.txt)) with high-visibility glowing golden star icons.
   - **Toggle Controls**: Turn heatmaps, pick count badges, and colorblind markers on or off dynamically.

3. **Word-by-Word Analysis Navigator**:
   - Filter words: Browse all 177 words or filter to the 54 words answered by colorblind participants.
   - Navigation: Dropdown selector, Previous/Next controls, Random selection, and Keyboard Arrow navigation.

4. **Detailed Metrics & Demographic Breakdown**:
   - Displays consensus square (modal pick) per word.
   - Breakdown card showing colorblind participant details (User ID, Age Range, Gender, selected coordinate, color swatch).

---

## How to Run

Manage dependencies and execute the web server with `uv`:

```bash
uv run python AppVisualizer/app.py
```

Once started, open your web browser at:
`http://localhost:5000`

---

## Directory Structure

```
AppVisualizer/
├── README.md               # Visualizer documentation
├── app.py                  # Flask web server & API endpoints
├── data_loader.py          # Data ingestion (HC_RGB.csv, HC.csv, colorblind_ids.txt)
├── static/
│   ├── css/
│   │   └── style.css       # Custom dark UI styling & animations
│   └── js/
│       └── main.js         # Frontend interactive logic & grid renderer
└── templates/
    └── index.html          # Web application HTML interface
```
