@echo off
echo Compiling LaTeX report...
pdflatex -interaction=nonstopmode project_report.tex
pdflatex -interaction=nonstopmode project_report.tex
echo.
echo Done! Check project_report.pdf
pause
