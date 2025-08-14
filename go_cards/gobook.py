from playwright.sync_api import sync_playwright
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import xml.etree.ElementTree as ET
from PIL import Image
import svgwrite
import textwrap
import json
import re

import base64
import os
import argparse
from pdb import set_trace as BP
from pprint import pprint

LETTER_WIDTH, LETTER_HEIGHT = letter  # Letter size: 8.5" x 11"
PAGE_HEIGHT = LETTER_HEIGHT 
PAGE_WIDTH = LETTER_WIDTH # * 0.80 # Cards should be narrower than a quarter letter size
SCALE_FACTOR = 0.85 # Cards should be smaller than a quarter letter size

#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate a Go book from svg files')
    parser.add_argument('--config', type=str, help='Json file with the chapters', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    title = config['title']
    chapters = config['chapters']    
    
    book_folder = args.config.split('/')[-1].split('.')[0]
    os.makedirs(book_folder, exist_ok=True)

    output_pdf = f'{book_folder}/{book_folder}_book.pdf'
    canv = canvas.Canvas(output_pdf, pagesize=letter)

    title_svg = f'{book_folder}/title.svg'
    create_book_title_index_svg(title, chapters, title_svg)
    batch_convert_svgs(book_folder, book_folder, 2.0)
    title_png = f'{book_folder}/title.png' 
    canv.drawImage(title_png, 0,0, width = PAGE_WIDTH, height = PAGE_HEIGHT)
    canv.showPage()
    
    # Add an empty page (back of index)
    canv.showPage()
        
    blank_svg = f'{book_folder}/blank.svg'
    generate_blank_svg(blank_svg)
    
    chapter_start_page = 1
    for chapidx,chapter in enumerate(chapters):
        chapter_pages = pages_in_chapter(chapter['folder'])
        generate_chapter(chapidx+1, chapter_start_page, chapter_pages, chapter['folder'], chapter['title'], canv)
        chapter_start_page += chapter_pages
    canv.save()
    
    print('Output is in ', output_pdf)
    print('Done.')

#-----------------------------------------
def pages_in_chapter(chapter_folder):
    """ Count the number of pages in a chapter """
    # Count the number of SVG files in the chapter folder
    svg_files = set(get_svgs(chapter_folder, basename_only=True))
    # Get the four letter prefixes into a set
    prefix_set = set([f.split('_')[0] for f in svg_files])    
    prefix_set.add('0000') # Add the title card
    ncards = len(prefix_set) # Number of cards
    
    # Four cards fill two pages
    num_pages = ((ncards - 1) // 4 + 1) * 2 
    return num_pages
    
#----------------------------------------------------    
def get_svgs(chapter_folder, basename_only=False):
    """ Get the list of SVG files in the chapter folder """
    svg_files = [f for f in os.listdir(chapter_folder) if f.endswith('.svg') and re.match(r'^\d{4}_', f) is not None]
    if not basename_only:
        svg_files = [f'{chapter_folder}/{f}' for f in svg_files]
    return svg_files
    
#------------------------------------------------------------------------------------------------
def generate_chapter( chapnum, chapter_start_page, chapter_pages, chapter_folder, title, canv):
    title_svg = f'{chapter_folder}/0000_f_1.svg'
    create_chapter_title_svg(chapnum, title, title_svg, LETTER_WIDTH, LETTER_HEIGHT, 30)
    blank_svg = f'{chapter_folder}/blank.svg'
    generate_blank_svg(blank_svg)

    svg_files = get_svgs(chapter_folder)
    png_files = [ f'{x.replace(".svg", ".png")}' for x in svg_files ]
    
    # Convert SVG to PNG
    batch_convert_svgs(chapter_folder, chapter_folder, 2.0)
        
    # output_pdf = f'{args.folder}/{args.folder}_flashcards.pdf'
    
    # #print_title(canv, args.title)   
    # #print_title(canv, "")
    
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    cards = [x for x in cards if not 'blank' in x]

    # Go through cards in groups of 4
    for i in range(0, len(cards), 4):
        pagenum = chapter_start_page + i // 2
        print(f"Processing cards {i+1} to {i+4}")
        
        # Four cards make a page
        if i+4 > len(cards):
            cards_group = cards[i:]
        else:
            cards_group = cards[i:i+4] 
        png_files_group = [f for f in png_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]      
        svg_files_group = [f for f in svg_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]
        captions_group = { f.replace('.svg', '.png'):get_caption(open(f).read()) for f in svg_files_group }
        
        generate_flashcards(canv, png_files_group, captions_group, i, chapnum, title, pagenum, f'{chapter_folder}/blank.png')

    os.remove(blank_svg)
    for f in png_files: os.remove(f)
    print()
    print(f"Generated chapter {title} in {chapter_folder}")

#----------------------------------------------------------------------------------
def create_chapter_title_svg(chapnum, title, output_path, width, height, font_size, max_chars=30):
    """ Generate and svg with chapter number and title """    
    dwg = svgwrite.Drawing(output_path, size=(f"{width}", f"{height}"))
    wrapped_lines = textwrap.wrap(title, width=max_chars)
    line_height = font_size * 1.3  
    total_text_height = len(wrapped_lines) * line_height
    start_y = (height - total_text_height) / 2 + font_size 
    
    wrapped_lines = [f'Chapter {chapnum}'] + wrapped_lines

    # Add each line to the SVG
    for i, line in enumerate(wrapped_lines):
        line_height = font_size * 1.3
        if i == 1: line_height = font_size * 1.5
        y_position = start_y + i * line_height  # Adjust y position for each line
        text_element = dwg.text(
            line,
            insert=("50%", y_position),  # Centered horizontally
            text_anchor="middle",  # Align text center
            font_size=font_size,
            font_family="Helvetica",
            fill="black"
        )
        dwg.add(text_element)

    # Save the SVG
    dwg.save()

#--------------------------------------------------------------------------
def create_book_title_index_svg(title, chapters, output_path):
    """ Generate an svg for the book title and index """
    dwg = svgwrite.Drawing(output_path, size=(f"{PAGE_WIDTH}", f"{PAGE_HEIGHT}"))
    title_y = '10%'
    title_elt = dwg.text(
        title,
        insert=("50%", title_y),  # Centered horizontally
        text_anchor="middle",  # Align text center
        font_size=30,
        font_family="Helvetica",
        fill="black"
    )
    dwg.add(title_elt)
    
    chapter_start_page = 1
    for idx, chapter in enumerate(chapters):
        chapter_title = chapter['title']
        chapter_num = idx + 1
        chapter_y = 10 + 5 + (idx) * 2
        chapter_y = f"{chapter_y}%"
        chapter_num_elt = dwg.text(
            f"{chapter_num}.",
            insert=("20%", chapter_y),  
            text_anchor="start",  # Align text left
            font_size=10,
            font_family="Helvetica",
            #font_weight="bold",
            fill="black"
        )
        dwg.add(chapter_num_elt)
        
        chapter_title_elt = dwg.text(
            f"{chapter_title}",
            insert=("23%", chapter_y),  
            text_anchor='start',
            font_size=10,
            font_family="Helvetica",
            fill="#555555"
        )
        dwg.add(chapter_title_elt)
    
        chapter_pages_elt = dwg.text(
            f"{chapter_start_page}",
            insert=("75%", chapter_y),  
            text_anchor="end",  # Align text right
            font_size=10,
            font_family="Helvetica",
            fill="#555555"
        )
        dwg.add(chapter_pages_elt)
        chapter_start_page += pages_in_chapter(chapter['folder'])

    # Save the SVG
    dwg.save()

#--------------------------------------------------------------------------------------------------------------
def generate_flashcards(canv, png_files, captions, numbering_offset, chapnum, chaptitle, pagenum, blank_png):
    """
        Draw two pages on a canvas.
        Four cards fit on a duplex sheet.
        The first page has the front side of the cards.
        The second page has the back side of the cards.
        Each side of a card has one or two diagrams.    
    """

    def centered_chap_num_and_title(canvas, chapnum, chaptitle):
        """ Center chapter number and title in one line at the top of the page """
        font = "Helvetica"
        size = 10
        txt = f"Chapter {chapnum}:  {chaptitle}"
        textwidth = canvas.stringWidth(txt, font, size)
        canvas.setFont(font, size)
        canvas.drawString(PAGE_WIDTH / 2 - textwidth / 2, PAGE_HEIGHT - 10, txt)
                
    get_first_png = lambda lst: lst[0] if lst else blank_png
            
    # 'double_44/0004_f_2.svg' -> '0004'
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in png_files ])))
    if len(cards) % 2 != 0:
        cards.append('xxxx')
        
    # Arrange front side (odd page)
    #--------------------------------
    scale_canvas(canv)
    
    if numbering_offset > 0:
        centered_chap_num_and_title(canv, chapnum, chaptitle)
    
    # Page num at the top right
    font = "Helvetica"
    size = 10
    textwidth = canv.stringWidth(str(pagenum), font, size)
    canv.setFont(font, size)
    canv.drawString(PAGE_WIDTH - textwidth - 10, PAGE_HEIGHT - 10, str(pagenum))

    for i, card in enumerate(cards):
        problem_num = i + numbering_offset
        front_dia_1 = get_first_png([ x for x in png_files if x.endswith(f"{card}_f_1.png") ]) # top
        front_dia_2 = get_first_png([ x for x in png_files if x.endswith(f"{card}_f_2.png") ]) # bot
        row = i // 2
        col = i % 2

        footer1 = f'{chapnum}-{problem_num}' 
        if 'blank.png' in front_dia_1: footer1 = ''
        if 'blank.png' not in front_dia_2: footer1 = ''
        if problem_num == 0: footer1 = ''
        footer2 = f'{chapnum}-{problem_num}'
        if 'blank.png' in front_dia_2: footer2 = ''
        if problem_num == 0: footer2 = ''
        draw_png_on_canvas(canv, front_dia_1, 2 * row, col, 
                           footer= footer1,
                           caption = captions.get(front_dia_1, ''))
        draw_png_on_canvas(canv, front_dia_2, 2 * row + 1, col, 
                            footer = footer2,
                            caption = captions.get(front_dia_2, ''))
        
    canv.showPage()

    # Arrange back side (even page)
    #--------------------------------
    scale_canvas(canv)
    
    centered_chap_num_and_title(canv, chapnum, chaptitle)
    
    # Page num at the top left
    font = "Helvetica"
    size = 10
    textwidth = canv.stringWidth(str(pagenum+1), font, size)
    canv.setFont(font, size)
    canv.drawString(10, PAGE_HEIGHT - 10, str(pagenum+1))
    
    for i, card in enumerate(cards):
        problem_num = i + numbering_offset
        #try:
        #    backcard = cards[i+1] if i%2 == 0 else cards[i-1]
        #except:
        #    backcard = 'xxxx'
        back_dia_1 = get_first_png([ x for x in png_files if x.endswith(f"{card}_b_1.png") ]) # top
        back_dia_2 = get_first_png([ x for x in png_files if x.endswith(f"{card}_b_2.png") ]) # bot

        row = i // 2
        col = i % 2

        footer1 = f'{chapnum}-{problem_num}' 
        if 'blank.png' in back_dia_1: footer1 = ''
        if 'blank.png' not in back_dia_2: footer1 = ''
        footer2 = f'{chapnum}-{problem_num}'
        if 'blank.png' in back_dia_2: footer2 = ''
        draw_png_on_canvas(canv, back_dia_1, 2 * row, col, 
                           footer= footer1,
                           caption = captions.get(back_dia_1, ''))
        draw_png_on_canvas(canv, back_dia_2, 2 * row + 1, col,
                           footer= footer2,
                           caption = captions.get(back_dia_2, ''))
                           
    canv.showPage()
        
#-------------------------------------------------------------
def draw_rect(canvas, rect):
    """ Useful for layout debugging """
    x, y, width, height = rect
    canvas.setLineWidth(1)
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(x, y, width, height)

#-------------------------------------------------------------        
def draw_line(canvas, x1, y1, x2, y2):
    canvas.line(x1, y1, x2, y2)        

#---------------------------------------------------------------------------
def batch_convert_svgs(svg_folder, output_folder, scale):
    """
    Convert SVG files to PNG.
    Increase scale if you want more pixels.
    """
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for filename in os.listdir(svg_folder):
            if not filename.endswith(".svg"): continue
            width, height = get_svg_size(os.path.join(svg_folder, filename))
            svg_path = os.path.join(svg_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".svg", ".png"))

            # Read SVG content
            with open(svg_path, "r") as f:
                svg_content = f.read()
                
            svg_data_uri = svg_to_data_uri(svg_content)  
            
            html = f'''
            <html>
            <body style="margin:0; padding:0; background-color:white;">
            <img src="{svg_data_uri}" width={width * scale}px">
            </body>
            </html>
            '''
            page.set_content(html)

            # Get image dimensions dynamically
            bbox = page.evaluate("document.querySelector('img').getBoundingClientRect();")
            page.set_viewport_size({
                "width": int(bbox["width"]),
                "height": int(bbox["height"])
            })
            # Take a screenshot
            page.screenshot(path=output_path, omit_background=True)
            print(f"Converted: {filename} at {output_path}")
            
        # Close the browser
        browser.close()
        
#--------------------------------------------------------------------------------
def draw_png_on_canvas(canvas, png_file, row, col, footer = '', caption = ''):
    """
    Position Go diagram on the ReportLab canvas.
    row in (0,1,2,3) 
    col in (0,1)
    y goes from bottom to top
    """
    if 'blank.png' in png_file: return
    caption_height = 10 #0
    footer_height = 15
    #if caption: caption_height = 10
    png_width, png_height = get_png_size(png_file)
    card_height = PAGE_HEIGHT / 2
    card_bottom = 0
    if row < 2: card_bottom = PAGE_HEIGHT + card_height
    middle = PAGE_HEIGHT / 2
    card_width = PAGE_WIDTH / 2
    hgap = PAGE_WIDTH * 0.05
    vgap = card_height * 0.05
    botmarg = card_height * 0.05
    topmarg = card_height * 0.05
    diagram_width = (PAGE_WIDTH - 2 * hgap) / 2
    diagram_height = (card_height - botmarg - topmarg - vgap) / 2 - caption_height # reduce height by caption if caption is there
    x = hgap / 2 + col * (diagram_width + hgap)
    if row == 0: y = PAGE_HEIGHT - topmarg - diagram_height - caption_height
    elif row == 1: y = PAGE_HEIGHT - topmarg - vgap - 2 * diagram_height - caption_height
    elif row == 2: y = middle - topmarg - diagram_height
    else: y = middle - topmarg - vgap - 2 * diagram_height - caption_height
                 
    #rect = (x, y, diagram_width, diagram_height)
    #draw_rect(canvas, rect)
    # Scale and center the diagram on the rectangle
    scale_x = diagram_width / png_width
    scale_y = diagram_height / png_height
    scale = min(scale_x, scale_y)
    if '0000_' in png_file: scale = max(scale_x, scale_y) # Title card
    #scale = scale_y
    
    dx = (diagram_width - png_width*scale) / 2
    dy = (diagram_height - png_height*scale) / 2
    dia_bottom = y + dy 
    caption_bottom = dia_bottom - caption_height
    footer_bottom = caption_bottom - footer_height
    
    canvas.drawImage(png_file, x + dx, dia_bottom, width=png_width*scale, height=png_height*scale)
    add_caption(canvas, caption, card_width, caption_bottom, col)
    
    if footer and not footer.startswith('0/'):
        # Draw footer
        font = "Helvetica"
        size = 10
        textwidth = canvas.stringWidth(footer, font, size)
        canvas.setFont(font, size)
        footer_y = y #y - 2 * caption_height
        #if caption: footer_y -= caption_height # below the caption
        canvas.drawString(card_width * col + card_width/2 - textwidth/2, 
                            footer_bottom, #(3 - row) / 2 * card_height + botmarg * 0.4, 
                            footer)

#-------------------------------------------------------------------------------
def add_caption(canvas, caption, card_width, caption_bottom, col):
    if not caption: return
    font = "Helvetica"
    size = 10
    textwidth = canvas.stringWidth(caption, font, size)
    canvas.setFont(font, size)
    canvas.drawString(card_width * col + card_width/2 - textwidth/2, 
                        caption_bottom, 
                        caption)

#--------------------------------    
def get_png_size(png_file):
    img = Image.open(png_file)
    width, height = img.size
    return width, height
        
#-------------------------------------------------------------        
def get_svg_size(svg_file):
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
  
#---------------------------------------------------------------------------
def scale_canvas(c):
    """Scale and center canvas"""
    scaled_width = PAGE_WIDTH * SCALE_FACTOR
    scaled_height = PAGE_HEIGHT * SCALE_FACTOR
    translate_x = (LETTER_WIDTH - scaled_width) / 2
    translate_y = (LETTER_HEIGHT - scaled_height) / 2
    c.translate(translate_x, translate_y)  # Move origin to center
    c.scale(SCALE_FACTOR, SCALE_FACTOR)    # Scale content

#-------------------------------------------------------------
def svg_to_data_uri(svg_content):
    """Convert SVG content to a base64 data URI"""
    svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f'data:image/svg+xml;base64,{svg_base64}'
        
#-------------------------------------------------------------        
def get_caption(txt):
    """ Parse the text between <caption> and </caption> from txt """
    start = txt.find('<caption>')
    if start == -1: return ''
    end = txt.find('</caption>')
    if end == -1: return ''
    res = txt[start+9:end].strip()
    return res
    
main()