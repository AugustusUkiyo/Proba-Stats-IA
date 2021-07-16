#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:34:15 2020

@author: ukiyo
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import time

#           MODELISATION ET FONCTIONS SIMPLES
        
class Bateau:
    def __init__(self, type):
        self.type = type
    
    def convert_int(self):
        if self.type == 'Porte-avion':
            return 5
        elif self.type == 'Croiseur':
            return 4
        elif self.type == 'Contre-torpilleur':
            return 3
        elif self.type == 'Torpilleur':
            return 2              
        elif self.type == 'Sous-marin':
            return 3.4
        
def peut_placer(grille,bateau,position,direction):
    """rend vrai s'il est possible de placer le bateau sur la grille
       position est couple (a,b)
       direction est: 1 pour horizontal et 2 pour vertical
    """
    x = position[0]
    y = position[1]
    if y + bateau > 10 and x + bateau > 10 :
        return False
    if direction == 1: # test horizontalement
        if y + int(bateau) > 10:
            return False
        for i in range(int(bateau)): # 1 a 4
            if grille[x][y+i] != 0:
                return False
        return True
    else:
        if x + int(bateau) > 10:
            return False
        for i in range(int(bateau)):
            if grille[x+i,y] != 0:
                return False
        return True

def place(grille,bateau,position,direction):
    """rend la grille modifiee"""
    x = position[0]
    y = position[1]
    if peut_placer(grille,bateau,position,direction) == True :
        if direction == 1: # test horizontalement
            for i in range(int(bateau)): # 1 a 4
                grille[x][y+i] = bateau
            return True
        else:
            for i in range(int(bateau)):
                grille[x+i,y] = bateau
            return True
    else:
        return False

def place_alea(grille,bateau):
    """place aleatoirement le bateau dans la grille"""
    x = random.randint(0,9)
    y = random.randint(0,9)
    position = (x,y)
    direction = random.randint(1,2)
    while(True):
        if place(grille,bateau,position,direction) == True:
            break
        else:
            x = random.randint(0,9)
            y = random.randint(0,9)
            position = (x,y)
            direction = random.randint(1,2)

def affiche(grille):
    """affiche la grille de jeu"""
    return plt.imshow(grille)

def eq(grilleA,grilleB):
    """tester l'egalite entre deux grilles"""
    return np.array_equal(grilleA,grilleB)

def genere_grille():
    """rend une grille avec les bateaux disposes de maniere aleatoire"""
    bateau1 = Bateau('Porte-avion').convert_int()
    bateau2 = Bateau('Croiseur').convert_int()
    bateau3 = Bateau('Contre-torpilleur').convert_int()
    bateau4 = Bateau('Torpilleur').convert_int()
    bateau5 = Bateau('Sous-marin').convert_int()
    liste_bateau = [bateau1,bateau2,bateau3,bateau4,bateau5]
    grille= np.zeros((10,10))
    for bateau in liste_bateau:
        place_alea(grille,bateau)
    return grille
    
#               COMBINATOIRE DU JEU
# Question 1
# Borne superieur
# Combinaison
# 2^100
# 1.267e+30

# Question 2
bateau = Bateau('Porte-avion').convert_int()
grille= np.zeros((10,10))

# 2.2
def q2_2(grille,bateau):
    """
        permet de dénombrer le nombre de façons de placer un bateau donné sur une grille vide
    """
    nb = 0
    for i in range(10):
        for j in range(10):
            if peut_placer(grille,bateau,(i,j),1) == True:
                nb += 1
            if peut_placer(grille,bateau,(i,j),2) == True:
                nb += 1
    return nb

q2_2(grille,bateau)    

def q2_3(grille,liste_bateaux):
    """
        permet de dénombrer le nombre de façon de placer une liste de bateaux sur une grille vide
    """
    nb = 0   
    if len(liste_bateaux) == 1:
        cp_grille = np.copy(grille)
        return q2_2(cp_grille,liste_bateaux[0])
    if len(liste_bateaux) == 2:
        for i in range(10):
            for j in range(10):
                grille1 = np.copy(grille)
                if place(grille1,liste_bateaux[0],(i,j),1) == True: # 1er bateau place horizontalement
                    nb += q2_2(grille1, liste_bateaux[1])
                grille2 = np.copy(grille)
                if place(grille2,liste_bateaux[0],(i,j),2) == True: # 1er bateau place verticalement
                    nb += q2_2(grille2, liste_bateaux[1])
        return nb
    if len(liste_bateaux) >= 3:    
        for i in range(10):
            for j in range(10):
                grille1 = np.copy(grille)
                if place(grille1,liste_bateaux[0],(i,j),1) == True:
                    nb += q2_3(grille1, liste_bateaux[1:])
                grille2 = np.copy(grille)
                if place(grille1,liste_bateaux[0],(i,j),2) == True:
                    nb += q2_3(grille2, liste_bateaux[1:])
        return nb
    
# test 1 bateau    
bateau = Bateau('Porte-avion').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[]
liste_bateaux.append(bateau)
q2_3(grille,liste_bateaux=[bateau])
# 120

# test 2 bateaux    
bateau1 = Bateau('Porte-avion').convert_int()
bateau2 = Bateau('Croiseur').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[bateau1,bateau2]
start = time.time()
print(q2_3(grille,liste_bateaux))
end = time.time()
print(end - start)
# 14400

# test 3 bateaux
bateau1 = Bateau('Porte-avion').convert_int()
bateau2 = Bateau('Croiseur').convert_int()
bateau3 = Bateau('Contre-torpilleur').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[bateau1,bateau2,bateau3]
start = time.time()
print(q2_3(grille,liste_bateaux))
end = time.time()
print(end - start)
# 1 413 432

# test 4 bateaux
bateau1 = Bateau('Porte-avion').convert_int()
bateau2 = Bateau('Croiseur').convert_int()
bateau3 = Bateau('Contre-torpilleur').convert_int()
bateau4 = Bateau('Torpilleur').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[bateau1,bateau2,bateau3,bateau4]
start = time.time()
print(q2_3(grille,liste_bateaux))
end = time.time()
print(end - start)
# 150 562 118
# 399 s
 
# test 5 bateaux
bateau1 = Bateau('Porte-avion').convert_int()
bateau2 = Bateau('Croiseur').convert_int()
bateau3 = Bateau('Contre-torpilleur').convert_int()
bateau4 = Bateau('Torpilleur').convert_int()
bateau5 = Bateau('Sous-marin').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[bateau1,bateau2,bateau3,bateau4,bateau5]
start = time.time()
print(q2_3(grille,liste_bateaux))
end = time.time()
print(end - start)
# 11 459 281 354
# 44189.09546470642 s

def q2_4(grille_donne=genere_grille()):
    """
         prend en paramètre une grille, génère des grilles aléatoirement jusqu’à 
         ce que la grille générée soit égale à la grille passée en paramètre et 
         qui renvoie le nombre de grilles générées
    """
    nb = 1
    grille_genere = genere_grille()
    while eq(grille_donne,grille_genere) == False:
        grille_genere = genere_grille()
        nb += 1
    return nb
q2_4()
# il faut presque 1 jour pour trouver cette solution
# 213 847 875

def q2_5(grille, liste_bateaux):
    """
        approximer le nombre total de grilles pour une liste de bateaux
    """
    nb = 1
    for bateau in liste_bateaux:
        nb *= q2_2(grille,bateau)
        place_alea(grille,bateau)
    return nb

bateau1 = Bateau('Porte-avion').convert_int()
bateau2 = Bateau('Croiseur').convert_int()
bateau3 = Bateau('Contre-torpilleur').convert_int()
bateau4 = Bateau('Torpilleur').convert_int()
bateau5 = Bateau('Sous-marin').convert_int()
grille= np.zeros((10,10))
liste_bateaux=[bateau1,bateau2,bateau3,bateau4,bateau5]
q2_5(grille,liste_bateaux)
# 35 280 000 000

#           MODELISATION PROBABILISTE DU JEU

class Bataille:
    def __init__(self, grille, liste_bateaux):
        self.grille = grille
        self.position_bateaux = []
        self.liste_bateaux = liste_bateaux
        self.vide = []
        self.touche = []
        self.liste_annexe = []
        for bateau in self.liste_bateaux:
            p_bateau = list(zip(*np.where(grille == bateau)))
            self.position_bateaux.append(p_bateau)
        #print(self.position_bateaux)
        
    def joue(self, position):
        x = position[0]
        y = position[1]
        # 10 indique le bateau est touche, 9 indique la case deja joue
        if self.grille[x,y] == 0:
            self.grille[x,y] = 9
            return False
        else:
            self.grille[x,y] = 10
            return True
    
    def victoire(self):
        total_point = 0
        for i in range(10):
            for j in range(10):
                if self.grille[i,j] == 10:
                    total_point +=1
        if total_point == 17:
            return True
        return False
    
    def observer(self):
        self.vide = []
        self.touche = []
        for i in range(10):
            for j in range(10):
                if self.grille[i,j] != 10 and self.grille[i,j] != 9: # case vide
                    self.vide.append((i,j))
                if self.grille[i,j] == 10: # case contient bateau touchee
                    self.touche.append((i,j))            
    
    def trouve_bateau_coule(self):
        #update la liste des bateaux
        liste_touche = list(self.touche)
        valeur = 0
        for p_bateau in self.position_bateaux:
            # Verifie si toutes les positions de bateau sont dans la liste touchee
            if all(position in liste_touche for position in p_bateau):
                # supprimer le bateau dans la liste des bateaux non coules
                #print('bateau', len(p_bateau),'position',p_bateau)
                if len(p_bateau) == 3 and 3.0 not in self.liste_bateaux and 3.4 in self.liste_bateaux :
                    self.liste_bateaux.remove(3.4)
                    valeur = 3.4
                else:
                    valeur = len(p_bateau)
                # supprimer toutes les positions de ce bateau dans la liste touchee
                for position in p_bateau:
                    liste_touche.remove(position)
                self.position_bateaux.remove(p_bateau)
        return valeur
                
                
    def reset(self):
        return
    
class Joueur:
    def __init__(self,name):
        self.name = name
        
    def prendre_position_ale(self,liste_vide):
        random.shuffle(liste_vide)
        return liste_vide.pop(0)
    
    def prendre_position_connexe(self, position_touche, grille):
        x = position_touche[0]
        y = position_touche[1]
        liste_choix_poissibles = [(x,y+1),(x,y-1),(x+1,y),(x-1,y)]
        liste_retourne = []
        for elem in liste_choix_poissibles:
            if elem[0] < 10 and elem[0] >= 0 and elem[1] < 10 and elem[1] >= 0 and grille[elem[0],elem[1]] != 9 and grille[elem[0],elem[1]] != 10:
                liste_retourne.append(elem)
        return liste_retourne
    
    def prendre_position_proba(self, liste_vide, liste_bateaux, grille):
        grille_proba = np.zeros((10,10))
        for bateau in liste_bateaux:
            for position in liste_vide: # calculer la densite de chaque case dans la liste des cases non visitees
                x = position[0]
                y = position[1]
                # test horizontal
                if y + int(bateau) <= 10:
                    for i in range(int(bateau)):
                        if grille[x][y+i] == 9 or grille[x][y+i] == 10:
                            break
                        if i == int(bateau) - 1:
                            for j in range(int(bateau)):
                                grille_proba[x,y+j] += 1
                # test vertical
                if x + int(bateau) <= 10:
                    for i in range(int(bateau)):
                        if grille[x+i][y] == 9 or grille[x+i][y] == 10:
                            break
                        if i == int(bateau) - 1:
                            for j in range(int(bateau)):
                                grille_proba[x+j,y] += 1
        
        # si la liste de proba est vide, on va jouer aleatoirement
        if eq(grille_proba,np.zeros((10,10))) == True:
            return (-1,-1)
        # trouver index de la valeur max
        ind = np.unravel_index(np.argmax(grille_proba, axis=None), grille_proba.shape)
        return ind
    
    def prendre_position_proba_connexe(self, liste_connexe, liste_bateaux, N):
        """
            retourner la position connexe avec la probabilite la plus probable
        """
        # creer un dictionaire pour compter le nbr de fois la case est touchée avec N grilles differantes
        dict_case = {}.fromkeys(liste_connexe, 0)
        for i in range(N):
            for case in liste_connexe:
                grille = genere_grille()
                jeu_test = Bataille(grille,liste_bateaux)
                touche = jeu_test.joue(case)
                if touche == True:
                    dict_case[case] += 1
        # Retourner la case avec la valeur la plus touchée
        case = max(dict_case, key=dict_case.get)
        # Enlever cette case dans la liste connexe
        liste_connexe.remove(case)
        return case             
        
    def prendre_position_monte_carlo(self):
        pass

def test_ale(n):
    """
    stimuler le jeu un nombre de fois donnee n
    """
    list_res = []
    # liste bateaux
    bateau1 = Bateau('Porte-avion').convert_int()
    bateau2 = Bateau('Croiseur').convert_int()
    bateau3 = Bateau('Contre-torpilleur').convert_int()
    bateau4 = Bateau('Torpilleur').convert_int()
    bateau5 = Bateau('Sous-marin').convert_int()
    liste_bateau = [bateau1,bateau2,bateau3,bateau4,bateau5]
    for i in range(n):
        grille_test = genere_grille() 
        jeu = Bataille(grille_test,liste_bateau)    
        j1 = Joueur('Test')
        nbr_essai = 0
        while (jeu.victoire() == False):
            nbr_essai += 1
            jeu.observer()
            x,y = j1.prendre_position_ale(jeu.vide)
            jeu.joue((x,y))
        list_res.append(nbr_essai)
    # Tracer la distribution
    plt.figure(figsize=(10,6))
    sns.distplot(list_res,hist=True,color = 'red')
    plt.xlim(10, 100)
    plt.xlabel('Nombre de coups joués')
    plt.ylabel('Probabilité')
    plt.savefig('Aleatoire.png')
    plt.show()
    print(int(sum(list_res)/len(list_res)))   
    
# test version aleatoire 95    
test_ale(100000)

def test_heuristique(n):
    list_res = []
    # liste bateaux
    bateau1 = Bateau('Porte-avion').convert_int()
    bateau2 = Bateau('Croiseur').convert_int()
    bateau3 = Bateau('Contre-torpilleur').convert_int()
    bateau4 = Bateau('Torpilleur').convert_int()
    bateau5 = Bateau('Sous-marin').convert_int()
    liste_bateau = [bateau1,bateau2,bateau3,bateau4,bateau5]
    for i in range(n):
        grille_test = genere_grille()
        jeu = Bataille(grille_test,liste_bateau)    
        j1 = Joueur('Test')
        nbr_essai = 0
        l = []
        while (jeu.victoire() == False):
            nbr_essai += 1
            jeu.observer()
            x,y = j1.prendre_position_ale(jeu.vide)
            touche1 = jeu.joue((x,y))
            if touche1 == True:       
                # prendre les cases annexes
                l += j1.prendre_position_connexe((x,y),jeu.grille)
            while len(l) != 0 and jeu.victoire() == False: # prendre la position dans la liste annexe tant qu'il reste encore des cases dans cette liste
                case = l.pop(0)
                touche2 = jeu.joue(case)
                nbr_essai += 1
                if touche2 == True and jeu.victoire() == False:
                    l += j1.prendre_position_connexe(case,jeu.grille)
        list_res.append(nbr_essai)
    # Tracer la distribution
    plt.figure(figsize=(10,6))
    sns.distplot(list_res,hist=True,color = 'red')
    plt.xlim(10, 100)
    plt.xlabel('Nombre de coups joués')
    plt.ylabel('Probabilité')
    plt.savefig('Heuristique.png')
    plt.show()
    print(int(sum(list_res)/len(list_res)))         
                    
# test version heuristique 60
test_heuristique(100000)                     

def test_proba_simple(n):
    """
        on suppose que la position des bateaux est indépendante
    """
    list_res = []
    # liste bateaux
    bateau1 = Bateau('Porte-avion').convert_int()
    bateau2 = Bateau('Croiseur').convert_int()
    bateau3 = Bateau('Contre-torpilleur').convert_int()
    bateau4 = Bateau('Torpilleur').convert_int()
    bateau5 = Bateau('Sous-marin').convert_int()
    liste_bateaux = [bateau1,bateau2,bateau3,bateau4,bateau5]
    for i in range(n):
        grille_test = genere_grille() 
        jeu = Bataille(grille_test,liste_bateaux)    
        j1 = Joueur('Test')
        nbr_essai = 0
        #print(grille_test)
        l = []
        while (jeu.victoire() == False):
            nbr_essai += 1
            jeu.observer()
            jeu.trouve_bateau_coule()
            x,y = j1.prendre_position_proba(jeu.vide, jeu.liste_bateaux, jeu.grille)
            if (x,y) == (-1,-1):
                x,y = j1.prendre_position_ale(jeu.vide)
            touche1 = jeu.joue((x,y))
            if touche1 == True:       
                # prendre les cases annexes
                l += j1.prendre_position_connexe((x,y),jeu.grille)
            while len(l) != 0 and jeu.victoire() == False: # prendre la position dans la liste annexe tant qu'il reste encore des cases dans cette liste
                case = l.pop(0)
                touche2 = jeu.joue(case)
                nbr_essai += 1
                if touche2 == True and jeu.victoire() == False:
                    l += j1.prendre_position_connexe(case,jeu.grille)  
        list_res.append(nbr_essai)
    # Tracer la distribution
    plt.figure(figsize=(10,6))
    sns.distplot(list_res,hist=True,color = 'red')
    plt.xlim(10, 100)
    plt.xlabel('Nombre de coups joués')
    plt.ylabel('Probabilité')
    plt.savefig('Proba_simple.png')
    plt.show()
    print(int(sum(list_res)/len(list_res)))   

# test version Proba_simple 53     
test_proba_simple(10000)        
            
def test_proba_ameliore(n):
    """
        la position des bateaux est dépendante
    """
    list_res = []
    # liste bateaux
    bateau1 = Bateau('Porte-avion').convert_int()
    bateau2 = Bateau('Croiseur').convert_int()
    bateau3 = Bateau('Contre-torpilleur').convert_int()
    bateau4 = Bateau('Torpilleur').convert_int()
    bateau5 = Bateau('Sous-marin').convert_int()
    liste_bateaux = [bateau1,bateau2,bateau3,bateau4,bateau5]
    for i in range(n):
        grille_test = genere_grille()
        jeu = Bataille(grille_test,liste_bateaux)    
        j1 = Joueur('Test')
        nbr_essai = 0
        while (jeu.victoire() == False):
            jeu.observer()
            jeu.trouve_bateau_coule()
            x,y = j1.prendre_position_proba(jeu.vide, jeu.liste_bateaux, jeu.grille)
            if (x,y) == (-1,-1):
                x,y = j1.prendre_position_ale(jeu.vide)
            touche1 = jeu.joue((x,y))
            nbr_essai += 1
            jeu.observer()
            if touche1 == True and jeu.victoire() == False and jeu.trouve_bateau_coule() == 0:       
                l = []
                l = j1.prendre_position_connexe((x,y),jeu.grille)
                while len(l) != 0 and jeu.victoire() == False and jeu.trouve_bateau_coule() == 0: # prendre la position dans la liste annexe tant qu'il reste encore des cases dans cette liste
                    case = j1.prendre_position_proba_connexe(l, jeu.liste_bateaux, 1000)
                    nbr_essai += 1
                    touche2 = jeu.joue(case)
                    jeu.observer()
                    if touche2 == True and jeu.trouve_bateau_coule() == 0:
                        l += j1.prendre_position_connexe(case,jeu.grille)
        list_res.append(nbr_essai)
    # Tracer la distribution
    plt.figure(figsize=(10,6))
    sns.distplot(list_res,hist=True,color = 'red')
    plt.xlim(10, 100)
    plt.xlabel('Nombre de coups joués')
    plt.ylabel('Probabilité')
    plt.savefig('Proba_ameliore.png')
    plt.show()
    print(int(sum(list_res)/len(list_res)))          

# test version Proba_ameliore 51 
test_proba_ameliore(1000)        
        
# N=10 51, N=100 51, N=1000 51         
        
        
        
        
        
        
        