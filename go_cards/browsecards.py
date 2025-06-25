
from pdb import set_trace as BP
import argparse
import http.server
import json
import os
import shutil
import tempfile
import json

from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time

#==========================================================
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = get_index_html()
            self.wfile.write(html.encode('utf-8'))
        elif '.svg' in self.path:
            # Cut off leading '/folder/' if it exists
            if self.path.startswith("/folder/"):
                self.path = self.path.split("/folder/")[-1]
            # serve the SVG file 
            try:
                with open(self.path, 'rb') as f:
                    self.send_response(200)
                    self.send_header("Content-type", "image/svg+xml")
                    self.end_headers()
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_error(404, "File not found")
        elif self.path.startswith("/folder/"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            folder = self.path.split("/folder/")[-1]
            print(">>>>>>>>>>>>>>>>>>>>>>>>", folder)
            html = get_folder_html(folder)
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_error(404, "Folder not found")    

# -----------------
def main():
    parser = argparse.ArgumentParser(
        description='Browse through svg flashcards in a folder')
    parser.add_argument('--folder', type=str,
                        help='folder containing the SVG files')
    args = parser.parse_args()

# -----------------------------
def get_index_html():
    """
    Get a list of all folders with files like 0001_f_1.svg in it and 
    present an html page with list of links to those folders. Clicking on a link will 
    send a request to the server to 
    """
    folders = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('_f_1.svg'):
                folder = os.path.relpath(root, '.')
                if folder not in folders:
                    folders.append(folder)
    folders = sorted(folders)
    html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Flashcard Browser</title>
            <style>
            body { text-align: center; font-family: sans-serif; font-size: 2em;}
            a, a:visited { display: block; margin: 1%%; text-decoration: none; color: #0000ee;}
            a:hover { color: #0000ee; text-decoration:underline; }
  
        </style>
        </head>
        <body>
        <h1>Flashcard Browser</h1>
        %s
        </body>
        </html>
    """
    links = []
    for folder in folders:
        folder_name = os.path.split(folder)[-1]
        title = folder_name.replace("_", " ").title()
        link = f'<a href="folder/{folder}">{title}</a>'
        links.append(link)
    links_html = '\n'.join(links)
    html = html % links_html    
    return html
        

# ---------------------------------
def get_folder_html( folder):
    print(f'>>>>>>>{folder}')
    BLANK_SVG = 'blank.svg'
    generate_blank_svg(BLANK_SVG)

    svg_files = [
        f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.svg')]
    cards = sorted(
        list(set([os.path.split(x)[-1].split('_')[0] for x in svg_files])))
    cards = [x for x in cards if not 'blank' in x and not '0000' in x and not 'title' in x ]

    svg_pairs = []
    for idx, card in enumerate(cards):
        svgs = []
        captions = []
        for dia in ['f_1', 'f_2', 'b_1', 'b_2']:
            fname = f"{folder}/{card}_{dia}.svg"
            svg = fname if os.path.exists(fname) else BLANK_SVG
            caption = get_caption(open(svg, 'r').read())
            captions.append(caption)
            svgs.append(svg)

        front_1, front_2 = (svgs[0], captions[0]), (svgs[1], captions[1])
        back_1, back_2 = (svgs[2], captions[2]), (svgs[3], captions[3])
        if back_2[0] == BLANK_SVG:
            back_1, back_2 = back_2, back_1

        svg_pairs.append((front_1, front_2))
        svg_pairs.append((back_1, back_2))
    
    html = generate_folder_html(svg_pairs, folder)
    return html

# ----------------------------------------------
def generate_folder_html(svg_pairs, folder):
    title = folder
    svg_pairs_base = [(pair[0][0], pair[1][0]) for pair in svg_pairs]
    svg_pairs_base_json = json.dumps(svg_pairs_base)
    caption_pairs = [(pair[0][1], pair[1][1]) for pair in svg_pairs]
    caption_pairs = json.dumps(caption_pairs)

    cards = [ x[0] if not 'blank' in x[0] else x[1] for x in svg_pairs_base ]
    cards = [ x.split('/')[-1] for x in cards ]
    cards = [ x.split('_')[0] for x in cards ]
    #viewer = os.path.join(folder, "index.html")
    html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Flashcard Browser</title>
            <style>
            body { text-align: center; font-family: sans-serif; font-size: 2em;}
            figure { display: inline-block; margin: 1%%; }
            img { width: 45vw; height: auto; border: none; }
            figcaption { margin-top: 0.5em; color: #444; }
            a, a:visited { text-decoration: none; color: #0000ee }
            a:hover { color: #0000ee; text-decoration:underline; }
          </style>
        </head>
        <body>
        <div id="heading"  style="text-align: left; margin-top: 30px; margin-left:120px ">
            <bold>%s</bold>
        </div>
        <div id="image-container">
        </div>
        <script>
            const svgPairs = %s;
            const cards = %s;
            const captionPairs = %s;
            const title = "%s";
            let index = 0;
            let probnum = 1;
            let nprobs = svgPairs.length / 2;

            function render() {
                const pair = svgPairs[index];
                const captionPair = captionPairs[index];
                        const card = cards[index];
                document.getElementById("image-container").innerHTML = `
                <figure>
                <img src="${pair[0]}">
                <figcaption>${captionPair[0]}</figcaption>
                </figure>
                <figure>
                <img src="${pair[1]}">
                <figcaption>${captionPair[1]}</figcaption>
                </figure>`;
                document.getElementById("heading").innerHTML = `
                <span>
                <a href="/">Back</a>
                &nbsp;
                <bold>${title} ${probnum}/${nprobs}</bold> (files ${card}_)
                </span>`;
            } // render()

            document.addEventListener("keydown", (e) => {
            if (e.key === "ArrowRight") {
                let up = function() {
                    index++;
                    index %%= svgPairs.length;
                    probnum = Math.floor(index / 2) + 1;
                }
                up();
                // Up again if both sides are blank
                if (svgPairs[index][0] == "blank.svg" && svgPairs[index][1] == "blank.svg") {
                    up();
                }
                render();
            } else if (e.key === "ArrowLeft") {
                let down = function() {
                    index--; 
                    if (index < 0) {
                        index = svgPairs.length - 1;
                    }
                    probnum = Math.floor(index / 2) + 1;
                }
                down();
                // Down again if both sides are blank
                if (svgPairs[index][0] == "blank.svg" && svgPairs[index][1] == "blank.svg") {
                    down();
                }
                render();
            }
            });

            render();
        </script>
        </body>
        </html>    
    """ % (title, svg_pairs_base_json, cards, caption_pairs, title)
    return html

# -------------------------------------------------------------
def get_caption(txt):
    """ Parse the text between <caption> and </caption> from txt """
    start = txt.find('<caption>')
    if start == -1:
        return ''
    end = txt.find('</caption>')
    if end == -1:
        return ''
    res = txt[start+9:end].strip()
    return res

# -------------------------------------------------------------
def generate_blank_svg(fname):
    with open(fname, 'w') as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100" height="100"></svg>')

# -------------------------------------------------------------
if __name__ == "__main__":
    #threading.Thread(target=start_server, daemon=True).start()
    #time.sleep(1)  # Give the server a moment to start
    server = HTTPServer(("localhost", 8000), MyHandler)
    webbrowser.open("http://localhost:8000/")
    print("Server running at http://localhost:8000")
    server.serve_forever()
