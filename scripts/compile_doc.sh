#!/bin/sh
pdflatex  --shell-escape $1
filename=`basename $1 .tex`
biber $filename
pdflatex  --shell-escape $1

if [ ! -d "../pdfs" ]; then
 mkdir "../pdfs"
fi
mv $filename.pdf  ../pdfs/

if [ ! -d "./logs" ]; then
 mkdir "./logs"
fi

for ext in aux bcf log run.xml bbl blg tex.bbl tex.blg; do 
 mv *.$ext ./logs/
done

Date=`date`
git add $filename.tex  ../pdfs/$filename.pdf
git add *.tex 
git commit -m "compiling $filenaem-$Date"
