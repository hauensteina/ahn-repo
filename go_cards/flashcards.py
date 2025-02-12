from pdb import set_trace as BP
import os
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import xml.etree.ElementTree as ET
import argparse

page_width, page_height = letter  # Letter size: 8.5" x 11"

#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate flashcards from svg files')
    parser.add_argument('--folder', type=str, help='folder containing the SVG files', required=True)
    args = parser.parse_args()
    
    generate_blank_svg('blank.svg')
    svg_files = [f'{args.folder}/{f}' for f in os.listdir(args.folder) if f.endswith('.svg')]
    # Pad with blank cards to make the number of cards a multiple of 4
    while len(svg_files) % 4 != 0:
        svg_files.append('blank.svg')
    output_pdf = f'{args.folder}/{args.folder}_flashcards.pdf'
    canv= canvas.Canvas(output_pdf, pagesize=letter)
    
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    # Go through cards in groups of 4
    for i in range(0, len(cards), 4):
        print(f"Processing cards {i+1} to {i+4}")
        cards_group = cards[i:i+4]
        svg_files_group = [f for f in svg_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]      
        generate_flashcards(canv, svg_files_group, output_pdf)

    canv.save()
    os.remove('blank.svg')
    print()
    print("Ignore any warnings")
    print(f"Flashcards generated in '{output_pdf}'")
    
#-------------------------------------------------------------
def generate_flashcards(canv, svg_files, output_pdf):
    """Generate a two-page PDF with the flashcards arranged for duplex printing."""

    get_first = lambda lst: lst[0] if lst else ''

    def draw_cutting_lines(c):
        c.setDash(5, 3)
        c.line(page_width/2, 0, page_width/2, page_height)
        c.line(0, page_height/2, page_width, page_height/2)
            
    # 'double_44/0004_f_2.svg' -> '0004'
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    # Swap the order of each pair in cards
    backs = [ cards[i+1] if i % 2 == 0 else cards[i-1] for i in range(len(cards)) ]
        
    # Arrange front side (first page)
    for i, card in enumerate(cards):
        front_1 = get_first([ x for x in svg_files if x.endswith(f"{card}_f_1.svg") ])
        front_2 = get_first([ x for x in svg_files if x.endswith(f"{card}_f_2.svg") ])
        row = i // 2
        col = i % 2

        draw_svg_on_canvas(canv, front_1, 2 * row, col)
        draw_svg_on_canvas(canv, front_2, 2 * row + 1, col)

    draw_cutting_lines(canv)
    canv.showPage()

    # Arrange back side (second page)
    for i, card in enumerate(backs):
        back_1 = get_first([ x for x in svg_files if x.endswith(f"{card}_b_1.svg") ])
        back_2 = get_first([ x for x in svg_files if x.endswith(f"{card}_b_2.svg") ])

        row = i // 2
        col = i % 2

        draw_svg_on_canvas(canv, back_1, 2 * row, col)
        draw_svg_on_canvas(canv, back_2, 2 * row + 1, col)

    draw_cutting_lines(canv)
    canv.showPage()
    
#-------------------------------------------------------------
def draw_rect(canvas, rect):
    x, y, width, height = rect
    canvas.setLineWidth(1)
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(x, y, width, height)

#-------------------------------------------------------------        
def draw_line(canvas, x1, y1, x2, y2):
    canvas.line(x1, y1, x2, y2)        

#----------------------------------------------------------------------------
def convert_svg_to_png_inkscape(svg_file, png_file, width, height):
    """Convert an SVG to PNG with a specific width and height using Inkscape."""
    command = [
        "inkscape", svg_file, 
        "--export-type=png", 
        "--export-filename=" + png_file, 
        f"--export-width={width}", 
        f"--export-height={height}",
        "--export-dpi=600",
        "--without-gui"
    ]
    subprocess.run(command, check=True)
    
#-------------------------------------------------------------
def draw_svg_on_canvas(canvas, svg_file, row, col):
    """Draw an SVG file on the ReportLab canvas using svglib."""
    if not os.path.exists(svg_file):
        return
    svg_width, svg_height = get_svg_dimensions(svg_file)
    hgap = page_width * 0.05
    vgap = page_height * 0.05
    diagram_width = (page_width - 2 * hgap) / 2
    diagram_height = (page_height - 4 * vgap) / 4
    x = hgap / 2 + col * (diagram_width + hgap)
    y = page_height - (row+1) * diagram_height - vgap / 2 - row * vgap
    #rect = (x, y, diagram_width, diagram_height)
    #draw_rect(canvas, rect)

    # Scale and center the SVG on the rectangle
    scale_x = diagram_width / svg_width
    scale_y = diagram_height / svg_height
    scale = min(scale_x, scale_y)
    scale = scale_y
    
    dx = (diagram_width - svg_width*scale) / 2
    dy = (diagram_height - svg_height*scale) / 2
    
    # Convert to png
    png_file = svg_file.replace('.svg', '.png')
    
    convert_svg_to_png_inkscape(svg_file, png_file, width=int(10*svg_width*scale), height=int(10*svg_height*scale))
    
    canvas.drawImage(png_file, x + dx, y + dy, width=svg_width*scale, height=svg_height*scale)
    os.remove(png_file)

#-------------------------------------------------------------        
def get_svg_dimensions(svg_file):
    """Extracts width and height from the <svg> tag in an SVG file."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    return float(width), float(height)

#-------------------------------------------------------------        
def generate_blank_svg(fname): 
    with open(fname, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100" height="100"></svg>')

        
main()