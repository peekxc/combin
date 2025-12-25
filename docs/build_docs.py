import mkapi
import combin
from mkapi.renderer import TemplateKind, render, render_source, render_document, render_heading
from mkapi.page import Page
from mkapi.page import generate_module_markdown, generate_object_markdown, convert_markdown

generate_object_markdown("comb_to_rank", "combin.combinatorial")

render("comb_to_rank", "combin.combinatorial", level=0, namespace="combin")

print()
from mkapi.parser import Parser

parser = Parser.create("comb_to_rank", "combin.combinatorial")
name_set = parser.parse_name_set()
range()


# render_source(parser.obj)

# convert_markdown()
# generate_module_markdown(combin.comb_to_rank)

# p = Page.create_source("a/b.md", "mkapi.page")
# p.generate_markdown()
# m = p.convert_markdown("")
# dir(mkapi)
