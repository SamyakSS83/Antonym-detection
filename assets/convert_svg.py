#!/usr/bin/env python3
"""
Convert SVG files to PNG for LaTeX compatibility
"""

import cairosvg
import os

svg_files = [
    'dual_encoder_architecture.svg',
    'semantic_space_analysis.svg', 
    'training_pipeline.svg',
    'performance_comparison.svg',
    'cross_lingual_analysis.svg'
]

for svg_file in svg_files:
    svg_path = f'/home/threesamyak/hsl844/Antonym-detection/assets/{svg_file}'
    png_path = f'/home/threesamyak/hsl844/Antonym-detection/assets/{svg_file.replace(".svg", ".png")}'
    
    if os.path.exists(svg_path):
        print(f"Converting {svg_file} to PNG...")
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=1200, output_height=800)
        print(f"Created {png_path}")
    else:
        print(f"Warning: {svg_path} not found")

print("SVG to PNG conversion completed!")
