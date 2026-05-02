import sys
from pathlib import Path

# Allow `python scripts/extract_graph_png.py` from repo root without PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core.graph import build_graph


graph = build_graph()

png_data = graph.get_graph().draw_mermaid_png()

with open("graph_structure.png", "wb") as f:
    f.write(png_data)

print("그래프 이미지가 graph_structure.png로 저장되었습니다.")