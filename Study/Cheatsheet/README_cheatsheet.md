# Cheat Sheet — how to edit, compile & print

`cheatsheet.tex` → a **4-page A4** exam cheat sheet (= 2 sheets, front + back, which is exactly what the exam allows). `cheatsheet.pdf` is the latest build.

## Edit it
Everything is plain LaTeX. Useful custom commands:
- `\sub{...}` = red bold keyword.
- `\fb{...}` = yellow-highlighted inline formula.
- `\fbw{...}` = yellow-highlighted formula auto-scaled to column width (use for wide formulas).
- `\section{...}` = blue underlined heading.
- Layout is 3 columns (`multicol`); `\columnbreak` forces a new column; content otherwise flows automatically.

## Keep it exactly 4 pages
The build is currently 4 pages at `12pt`. If you add content and it spills to a 5th page, either:
- remove some lines, or
- drop the font: change `\documentclass[12pt]{extarticle}` → `[11pt]` (was 3 pages) or `[10pt]`.
If you want it fuller, raise the font or add more notes. Always re-check the page count after edits.

## Compile in Overleaf (recommended for you)
1. Go to overleaf.com → New Project → Upload Project (or Blank Project) and upload `cheatsheet.tex`.
2. Set the compiler to **pdfLaTeX** (Menu → Compiler → pdfLaTeX). 
3. Click **Recompile**, then **Download PDF** and print at **100% / Actual size** (do NOT "fit to page", or margins shrink).
4. Print **double-sided** → you get 2 physical A4 sheets, front+back.

If Overleaf complains about `extarticle`, change the first line to `\documentclass[12pt]{article}` (slightly different sizing, still fine).

## Compile locally (TinyTeX is already installed on this machine)
```powershell
cd "<this folder>"
pdflatex cheatsheet.tex
```
Output: `cheatsheet.pdf`.

## Allowed at the exam
2 A4 sheets, front+back = 4 pages, handwritten OR typed. This printout qualifies. Bring a non-programmable calculator.
