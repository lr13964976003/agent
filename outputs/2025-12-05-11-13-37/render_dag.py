import graphviz

# Read the dot file
with open('../outputs/2025-12-05-11-13-37/complete_parallel_strategy.dot', 'r') as f:
    dot_content = f.read()

# Render to SVG
source = graphviz.Source(dot_content)
source.render('../outputs/2025-12-05-11-13-37/complete_parallel_strategy', format='svg', cleanup=True)
print("Generated SVG: complete_parallel_strategy.svg")
