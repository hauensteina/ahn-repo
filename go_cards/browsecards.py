
from pdb import set_trace as BP
import argparse
import http.server
import json
import os
import shutil
import socketserver
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
import tempfile
import webbrowser

# -----------------
def main():
    parser = argparse.ArgumentParser(
        description='Browse through svg flashcards in a folder')
    parser.add_argument('--folder', type=str,
                        help='folder containing the SVG files', required=True)
    args = parser.parse_args()

    blank_svg = f'{args.folder}/blank.svg'
    generate_blank_svg(blank_svg)

    svg_files = [
        f'{args.folder}/{f}' for f in os.listdir(args.folder) if f.endswith('.svg')]
    cards = sorted(
        list(set([os.path.split(x)[-1].split('_')[0] for x in svg_files])))
    cards = [x for x in cards if not 'blank' in x and not '0000' in x and not 'title' in x ]
    print(cards)

    svg_pairs = []
    for idx, card in enumerate(cards):
        svgs = []
        captions = []
        for dia in ['f_1', 'f_2', 'b_1', 'b_2']:
            fname = f"{args.folder}/{card}_{dia}.svg"
            svg = fname if os.path.exists(fname) else blank_svg
            caption = get_caption(open(svg, 'r').read())
            captions.append(caption)
            svgs.append(svg)

        front_1, front_2 = (svgs[0], captions[0]), (svgs[1], captions[1])
        back_1, back_2 = (svgs[2], captions[2]), (svgs[3], captions[3])
        if back_2[0] == blank_svg:
            back_1, back_2 = back_2, back_1

        svg_pairs.append((front_1, front_2))
        svg_pairs.append((back_1, back_2))

    htmldir = os.path.join(args.folder, "html")
    os.makedirs(htmldir, exist_ok=True)
    indexhtml = generate_html_files(svg_pairs, htmldir)
    serve_html(htmldir, indexhtml)

# -------------------------------------------------------------
def generate_blank_svg(fname):
    with open(fname, 'w') as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100" height="100"></svg>')

# ----------------------------------------------
def generate_html_files(svg_pairs, folder):
    title = os.path.split(folder)[-2]
    svg_pairs_base = [(os.path.basename(pair[0][0]),
                       os.path.basename(pair[1][0])) for pair in svg_pairs]
    svg_pairs_base_json = json.dumps(svg_pairs_base)
    caption_pairs = [(pair[0][1], pair[1][1]) for pair in svg_pairs]
    caption_pairs = json.dumps(caption_pairs)
    cards = [ x[0] if not 'blank' in x[0] else x[1] for x in svg_pairs_base ]
    cards = [ x.split('_')[0] for x in cards ]
    viewer = os.path.join(folder, "index.html")
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
        </style>
        </head>
        <body>
        <div id="heading">
            <bold>%s</bold>
        </div>
        <div id="image-container">
        <img src="0020_f_1.svg"><img src="0020_f_2.svg">
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
			<bold>${title} ${probnum}/${nprobs}</bold> (files ${card}_)`;
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
    with open(viewer, "w") as f:
        f.write(html)

    # Copy the SVG files into the temp folder
    for pair in svg_pairs:
        for svg in pair:
            shutil.copy(svg[0], os.path.join(folder, os.path.basename(svg[0])))

    return viewer

# ----------------------------------------------
def serve_html(folder, html_file):
    pwd = os.getcwd()
    os.chdir(folder)
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    with ReusableTCPServer(("", port), handler) as httpd:
        webbrowser.open(
            f"http://localhost:{port}/{os.path.basename(html_file)}")
        print(f"Serving at http://localhost:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")
            os.chdir(pwd)
            shutil.rmtree(folder)
            print("Temporary html folder removed.")

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


if __name__ == "__main__":
    main()
