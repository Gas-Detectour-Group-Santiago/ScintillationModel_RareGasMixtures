Archivos LaTeX reformateados:
- Se mantienen separadas las columnas de incertidumbre estadística y sistemática.
- Las captions se han reducido al título de la tabla.
- Los valores y errores se escriben mediante \num{...} en notación científica.
- Cuando un parámetro tiene incertidumbre estadística y sistemática exactamente nulas, ambas columnas se muestran como --.

Requiere en el preámbulo:
\usepackage{booktabs}
\usepackage{siunitx}
