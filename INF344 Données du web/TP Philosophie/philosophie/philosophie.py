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
        if title1=="Philosophie":
            flash("C'est un peu de la triche...", 'not_error')
            session["score"] = 1
            return redirect('/')

        if title1 == "":
            flash("La page n'existe pas, retournez à l'école", "error")
            return redirect('/')
        elif not items1:
            flash('Aucun résultat, vous êtes très nul.', "error")
            return redirect('/')

        cache[session["article"]] = title1, items1
    return render_template('game.html', title=title1, items=items1, score=session["score"])


@app.route('/new-game', methods=['POST'])
def new_game():
    session["article"] = request.form["nom_page"]
    session["score"] = 0
    return redirect('/game')


@app.route('/move', methods=['POST'])
def next_page():
    new_page = request.form["new_page"]

    if int(request.form["score"]) != int(session["score"]):
        if session["article"] != request.form["new_page"]:
            flash('Mouvement annulé', "error")
            return redirect('/game')

    elif new_page == "Philosophie":
        if new_page in cache[session["article"]][1]:
            flash("Pas très dur ce jeu...", "not_error")
            return redirect('/')
        else :
            flash("Tricheur !", "error")
            return redirect("/game")
    else :
        session["article"] = request.form["new_page"]
        session["score"] += 1
        return redirect('/game')
# Si vous définissez de nouvelles routes, faites-le ici


if __name__ == '__main__':
    app.run(debug=True)
