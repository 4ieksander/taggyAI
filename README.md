# Taggy

Taggy is a simple command-line application for managing tags.


#### Tłumaczenie dokumentacji
<code>
cd docs_sphinx
</code>
<code>
sphinx-build -b gettext source/ locale/
</code>
<code>
sphinx-intl update -p locale -l pl
</code>
<code>
python .\translate_po.py
</code>

#### Budowanie dokumentacji
##### HTML
<code>
sphinx-build -b html source/ build/html
</code>    
##### PDF
<code>
sphinx-build -b latex source/ build/latex
</code>
##### DOCX
<code>
pandoc -s build/latex/taggy.tex -o taggy.docx
</code>
# Do PDF
make -C build/latex all-pdf
