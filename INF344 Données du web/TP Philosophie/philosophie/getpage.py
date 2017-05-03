#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Ne pas se soucier de ces imports
import setpath
from bs4 import BeautifulSoup
from json import loads
from urllib.request import urlopen
from urllib.parse import urlencode
from pprint import pprint
from urllib.parse import unquote
from urllib.parse import urldefrag

# Si vous écrivez des fonctions en plus, faites-le ici


def getJSON(page):
    params = urlencode({
      'format': 'json',  # TODO: compléter ceci
      'action': 'parse',  # TODO: compléter ceci
      'prop': 'text',  # TODO: compléter ceci
      'redirects' : "true",
      'page': page})
    API = "https://fr.wikipedia.org/w/api.php"  # TODO: changer ceci
    response = urlopen(API + "?" + params)
    return response.read().decode('utf-8')


def getRawPage(page):
    parsed = loads(getJSON(page))
    try:
        title = parsed["parse"]["title"]  # TODO: remplacer ceci
        content = parsed["parse"]["text"]["*"]  # TODO: remplacer ceci
        return title, content
    except KeyError:
        # La page demandée n'existe pas
        return None, None


def getPage(page):
    page = page.replace(" ", "_")
    title, json = getRawPage(page)
    soup = BeautifulSoup(json, 'html.parser')
    liste_p = soup.find_all("p", recursive=False)
    liste_a=[]
    
    for item in liste_p:
        item.find_all("a", href=True)
        liste_a += [elem for elem in item.find_all("a", href=True)]

    new_list = []
    
    for item in liste_a:
        try: 
            if item["href"].split("/")[1]=="wiki":
                elemt = unquote(urldefrag(item["href"].split("/")[2])[0]).replace("_", " ")
                if elemt not in new_list:
                    if "Aide:" not in elemt:
                        new_list.append(elemt)
        except:
            continue
    return title, new_list[:10]   # TODO: écrire ceci


if __name__ == '__main__':
    # Ce code est exécuté lorsque l'on exécute le fichier

    print("Ça fonctionne !")

    # Voici des idées pour tester vos fonctions :
    print(getPage("Histoire"))
    # print(getRawPage("Utilisateur:A3nm/INF344"))
    # print(getRawPage("Histoire"))
