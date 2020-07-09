
from flask import Flask, render_template, url_for, flash, redirect
from forms import RegistrationForm, LoginForm

app = Flask(__name__)

app.config['SECRET_KEY'] = '769d3f053b8f29f33041d5b7fda804b3'

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

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash( f'Account created for {form.username.data}!', 'success')
        return redirect( url_for( 'home'))
    return render_template( 'register.tmpl', title='Register', form=form)

@app.route('/login')
def login():
    form = LoginForm()
    return render_template( 'login.tmpl', title='Login', form=form)

if __name__ == '__main__':
    app.run( debug=True)
