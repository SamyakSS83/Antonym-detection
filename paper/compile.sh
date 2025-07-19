#!/bin/bash

# Bhav-Net Paper Compilation Script
# Compiles the LaTeX paper with proper bibliography handling

echo "Compiling Bhav-Net paper..."

# Check if required files exist
if [ ! -f "bhav_net_paper.tex" ]; then
    echo "Error: bhav_net_paper.tex not found!"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "Error: references.bib not found!"
    exit 1
fi

# Clean previous compilation files
echo "Cleaning previous compilation files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls

# First LaTeX pass
echo "First LaTeX pass..."
pdflatex bhav_net_paper.tex

if [ $? -ne 0 ]; then
    echo "Error in first LaTeX pass!"
    exit 1
fi

# BibTeX pass
echo "Processing bibliography..."
bibtex bhav_net_paper

if [ $? -ne 0 ]; then
    echo "Error in BibTeX processing!"
    exit 1
fi

# Second LaTeX pass
echo "Second LaTeX pass..."
pdflatex bhav_net_paper.tex

if [ $? -ne 0 ]; then
    echo "Error in second LaTeX pass!"
    exit 1
fi

# Third LaTeX pass (for final references)
echo "Final LaTeX pass..."
pdflatex bhav_net_paper.tex

if [ $? -ne 0 ]; then
    echo "Error in final LaTeX pass!"
    exit 1
fi

echo "Compilation successful! Generated: bhav_net_paper.pdf"

# Optional: Clean auxiliary files
read -p "Clean auxiliary files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls
    echo "Auxiliary files cleaned."
fi

echo "Paper compilation complete!"
