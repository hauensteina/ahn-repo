
from flask import Flask, render_template, url_for

app = Flask(__name__)

posts = [
    {
        'author':'Joe Schmoe',
        'title':'About Me',
        'content':'Got the virus',
        'date_posted':'July 6, 2020'
    },
    {
        'author':'Jane Doe',
        'title':'Whatever',
        'content':'Never be passive aggressive',
        'date_posted':'July 6, 2020'
    }
]

@app.route('/')
@app.route('/home')
def home():
    return render_template( 'home.tmpl', posts=posts)

@app.route('/about')
def about():
    return render_template( 'about.tmpl')

if __name__ == '__main__':
    app.run( debug=True)
