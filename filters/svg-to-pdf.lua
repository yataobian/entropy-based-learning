-- For the PDF profile: point image references at a PDF sibling
-- (the render script converts every figures/*.svg first).

function Image(el)
  if not quarto.doc.is_format("pdf") then
    return el
  end
  el.src = el.src:gsub("%.svg$", ".pdf")
  return el
end
