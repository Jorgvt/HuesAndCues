import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_path(path):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    root_relative = os.path.join(BASE_DIR, path)
    if os.path.exists(root_relative):
        return root_relative
    return path

class DataLoader:
    def __init__(self, rgb_path="HC_RGB.csv", hc_path="HC.csv", cb_ids_path="colorblind_ids.txt"):
        self.rgb_path = resolve_path(rgb_path)
        self.hc_path = resolve_path(hc_path)
        self.cb_ids_path = resolve_path(cb_ids_path)
        
        self.rgb_df = None
        self.hc_df = None
        self.cb_ids = set()
        self.board_cells = []
        self.board_map = {}
        
        self.load_data()

        
    def load_data(self):
        # 1. Load Colorblind IDs
        if os.path.exists(self.cb_ids_path):
            with open(self.cb_ids_path, 'r', encoding='utf-8') as f:
                self.cb_ids = set(line.strip() for line in f if line.strip())
        
        # 2. Load RGB Board Grid
        if os.path.exists(self.rgb_path):
            self.rgb_df = pd.read_csv(self.rgb_path)
            # Ensure proper types
            self.rgb_df['coordenada_x'] = self.rgb_df['coordenada_x'].astype(str).str.strip()
            self.rgb_df['coordenada_y'] = self.rgb_df['coordenada_y'].astype(int)
            
            # Create cell dictionary and array
            rows = sorted(self.rgb_df['coordenada_x'].unique())
            row_map = {r: i for i, r in enumerate(rows)}
            
            for _, row in self.rgb_df.iterrows():
                x = row['coordenada_x']
                y = int(row['coordenada_y'])
                coord = f"{x}{y}"
                r, g, b = int(row['R']), int(row['G']), int(row['B'])
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                
                cell_data = {
                    "coord": coord,
                    "row": x,
                    "col": y,
                    "row_idx": row_map[x],
                    "col_idx": y - 1,
                    "r": r,
                    "g": g,
                    "b": b,
                    "hex": hex_color
                }
                self.board_cells.append(cell_data)
                self.board_map[coord] = cell_data

        # 3. Load Human Responses
        if os.path.exists(self.hc_path):
            self.hc_df = pd.read_csv(self.hc_path)
            self.hc_df['coordinate'] = self.hc_df['coordinate'].astype(str).str.strip()
            self.hc_df['is_colorblind'] = self.hc_df['userId'].isin(self.cb_ids)

    def get_board_grid(self):
        """Returns board setup details including dimensions and cell info."""
        rows = sorted(list(set(c['row'] for c in self.board_cells)))
        cols = sorted(list(set(c['col'] for c in self.board_cells)))
        return {
            "rows": rows,
            "cols": cols,
            "total_cells": len(self.board_cells),
            "cells": self.board_cells,
            "cell_map": self.board_map
        }

    def get_words_list(self):
        """Returns list of unique words with summary counts."""
        if self.hc_df is None:
            return []
            
        words_summary = []
        grouped = self.hc_df.groupby('word')
        
        for word, group in grouped:
            category = group['clueCategory'].iloc[0] if 'clueCategory' in group.columns else "UNKNOWN"
            total = len(group)
            cb_group = group[group['is_colorblind']]
            cb_count = len(cb_group)
            cb_users_count = cb_group['userId'].nunique()
            
            words_summary.append({
                "word": word,
                "category": category,
                "total_responses": total,
                "colorblind_responses": cb_count,
                "colorblind_users_count": cb_users_count,
                "has_colorblind": cb_count > 0
            })
            
        # Sort words alphabetically
        words_summary.sort(key=lambda x: x['word'])
        return words_summary

    def get_word_analysis(self, word_name):
        """Returns full response analysis for a single word."""
        if self.hc_df is None:
            return None
            
        word_data = self.hc_df[self.hc_df['word'] == word_name]
        if word_data.empty:
            return None
            
        category = word_data['clueCategory'].iloc[0] if 'clueCategory' in word_data.columns else ""
        clue_id = int(word_data['clueId'].iloc[0]) if 'clueId' in word_data.columns else None
        
        # Picks count per coordinate
        counts = word_data['coordinate'].value_counts().to_dict()
        max_count = max(counts.values()) if counts else 0
        
        # Colorblind responses
        cb_df = word_data[word_data['is_colorblind']]
        cb_responses = []
        for _, row in cb_df.iterrows():
            coord = row['coordinate']
            cell = self.board_map.get(coord, {})
            cb_responses.append({
                "userId": row['userId'],
                "short_userId": row['userId'][:8],
                "ageRange": row.get('ageRange', 'N/A'),
                "gender": row.get('gender', 'N/A'),
                "coordinate": coord,
                "hex": cell.get('hex', '#ffffff'),
                "rgb": [cell.get('r', 0), cell.get('g', 0), cell.get('b', 0)],
                "timestamp": row.get('timestamp', '')
            })
            
        # Non-colorblind responses count per coordinate
        non_cb_df = word_data[~word_data['is_colorblind']]
        non_cb_counts = non_cb_df['coordinate'].value_counts().to_dict()
        
        # Top consensus coordinates
        top_coords = []
        for coord, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            cell = self.board_map.get(coord, {})
            top_coords.append({
                "coordinate": coord,
                "count": count,
                "percentage": round((count / len(word_data)) * 100, 1),
                "hex": cell.get('hex', '#ffffff')
            })

        return {
            "word": word_name,
            "category": category,
            "clueId": clue_id,
            "total_responses": len(word_data),
            "colorblind_responses_count": len(cb_df),
            "non_colorblind_responses_count": len(non_cb_df),
            "counts_per_coord": counts,
            "non_cb_counts_per_coord": non_cb_counts,
            "max_count": max_count,
            "top_coords": top_coords,
            "colorblind_responses": cb_responses
        }

    def get_colorblind_summary(self):
        """Returns overall colorblind users summary statistics."""
        if self.hc_df is None or not self.cb_ids:
            return []
            
        cb_df = self.hc_df[self.hc_df['is_colorblind']]
        summary = []
        
        for user_id, group in cb_df.groupby('userId'):
            summary.append({
                "userId": user_id,
                "short_userId": user_id[:8],
                "total_words_answered": len(group),
                "unique_words_answered": group['word'].nunique(),
                "ageRange": group['ageRange'].iloc[0] if 'ageRange' in group.columns else 'N/A',
                "gender": group['gender'].iloc[0] if 'gender' in group.columns else 'N/A'
            })
            
        summary.sort(key=lambda x: x['total_words_answered'], reverse=True)
        return summary
