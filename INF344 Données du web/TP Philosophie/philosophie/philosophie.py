#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Ne pas se soucier de ces imports
import setpath
from flask import Flask, render_template, session, request, redirect, flash
from getpage import getPage

app = Flask(__name__)

app.secret_key = "TODO: mettre une valeur secrète ici"

global cache
cache = {}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/game', methods=['GET'])
def game(title=None, items=None, score=None):
    if session["article"] in cache.keys():
        title1, items1 = cache[session["article"]]
        session["test_cache"] = 1
    else :
        session["test_cache"] = 0
        title1, items1 = getPage(session["article"])
        cache[session["article"]] = title1, items1
    return render_template('game.html', title=title1, items=items1, score=session["score"])


@app.route('/new-game', methods=['POST'])
def new_game():
    session["article"] = request.form["nom_page"]
    session["score"] = 0
    return redirect('/game')


@app.route('/move', methods=['POST'])
def next_page():
    #if request.form["score"] != session["score"]:
    #    flash('Mouvement annulé')
    #    return redirect('/')
    
    new_page = request.form["new_page"]
    
    if new_page == "Philosophie":
        flash('Bravo vous avez gagné')
        return redirect('/')
    else :
        session["article"] = request.form["new_page"]
        session["score"] += 1
        return redirect('/game')
# Si vous définissez de nouvelles routes, faites-le ici


if __name__ == '__main__':
    app.run(debug=True)

