import streamlit as st
import cv2
import warnings
warnings.filterwarnings('ignore')

import sys

sys.path.insert(0, '/Users/manu/Desktop/ML_Projects/DataScientest/DoctorPy')

from utils.utils_st import *

# Project title
st.title('Projet COVID-19')

# The sidebar that will be located on the left and give several options
st.sidebar.subheader('Sommaire')

# Table of contents
add_selectbox = st.sidebar.radio(
    "",
    ("Présentation du projet", "Analyse exploratoire", "Biais", "Différents preprocessings", "Unet", "Prédiction", "Conclusion")
)

st.header(add_selectbox)

# To make it more esthetic for those using dark mode, and there will eventually be a language option 
st.sidebar.subheader('Préférences')

dmode = st.sidebar.checkbox('Mode sombre pour les graphes')
if dmode:
    plt.style.use('dark_background')
    sns.set_palette('pastel')
    plt.rcParams.update({
        "savefig.facecolor": "#0F1116",
        "axes.facecolor": "#0F1116"})
else:
    plt.style.use('default')

# Bibliography section
links = '<ul>' \
        '<li><a href="https://datascientest.com" style="text-decoration:None;">DataScientest</a></li>' \
        '<li><a href="https://arxiv.org/abs/1505.04597" style="text-decoration:None;">U-Net</a></li>' \
        '<li><a href="https://github.com/DataScientest/DoctorPy" style="text-decoration:None;">Repo GitHub</a></li>' \
        '</ul>'

st.sidebar.subheader('Liens utiles')
st.sidebar.markdown(links, unsafe_allow_html=True)


# Basic introduction
if add_selectbox == 'Présentation du projet':
    st.markdown("Ce projet, proposé par DataScientest, a pour but de classifier des radiographies pulmonaires de \
    patients dans trois catégories :")
    st.markdown("COVID-19, NORMAL, Viral Pneumonia")

    st.subheader('Présentation de l\'équipe')
    st.markdown('**Mentor** Paul Dechorgnat')
    st.markdown("**Membre de l'équipe** Amuli Inès")
    st.markdown("**Membre de l'équipe** Ngoma Algy")
    st.markdown("**Membre de l'équipe** Potrel Manu")
    st.markdown("**Membre de l'équipe** Turlet Frédéric")

    st.subheader('Lien vers les données')
    st.markdown('Les données ont été prises sur le site de Kaggle.')


    # Plotting a few examples
    st.subheader('Exemples de radiographies')
    if st.button('Générer quelques images'):
        fig = plot_images()
        st.pyplot(fig)


# EDA
# This section contains a fair amount of exploration
# The names of the various buttons and tools will be self explanatory
elif add_selectbox == "Analyse exploratoire":

    st.markdown("Le jeu de données en question est un jeu relativement équilibré. \
    En effet, il y 1143 images COVID-19, 1342 images NORMAL et 1345 images Viral Pneumonia.")
    st.markdown('Commençons par visualiser quelques images pour chaque classe.')
    fig = plot_images()
    stplot = st.empty()
    stplot = st.pyplot(fig)

    # Random images generation
    if st.button('Générer de nouvelles images'):
        fig = plot_images()
        stplot.pyplot(fig)

    st.markdown("Il semble y avoir de grandes similarités entre les images NORMAL et Viral Pneumonia. "\
                "Cela peut avoir un effet discriminant sur les images COVID-19. Nous allons étudier ce phénomène.")

    st.markdown("Les images labellisées COVID-19 semblent être de qualité inférieure, "
                "avec parfois des indications sur les radiographies, "
                "des cables et des tubes qui sont fréquemment présents "
                "et des des tracés flous blancs dans les zones sombres.")

    # Intensity mean
    st.subheader('Moyenne de pixels')
    mean_txt_pre = "Nous allons dans un premier temps comparer les moyennes de pixels des images de chaque classe."
    st.markdown(mean_txt_pre)
    fig = load_eda('mean')
    st.pyplot(fig)
    mean_txt_post = "Les trois distributions semblent être issues de loi gaussiennes distinctes mais, " \
                    "à nouveau, il semble que celle de la classe COVID-19 soit celle qui se distingue le plus."
    st.markdown(mean_txt_post)

    # Contrast
    st.subheader('Ecart-type')

    std_txt_pre = "Nous allons maintenant étudier la distribution des écart-types, " \
                  "qui peut être assimilé à une mesure du contraste des images."
    st.markdown(std_txt_pre)
    fig = load_eda('std')
    st.pyplot(fig)
    std_txt_post = "Les distributions semblent toujours suivre des lois normales, " \
                   "toujours avec des différences plus remarquables au niveau de la classe COVID-19."
    st.markdown(std_txt_post)

    # Shannon entropy
    st.subheader('Entropie')

    entropy_txt_pre = "Afin d'étudier la complexité des images, il est possible d'en étudier l'entropie."
    st.markdown(entropy_txt_pre)
    fig = load_eda('entropy')
    st.pyplot(fig)
    entropy_txt_post = "Il ne semble pas ici y avoir de problème particulier."
    st.markdown(entropy_txt_post)

    # Threshold investigation, the choice of 50 comes from deeper exploration
    st.subheader('Seuil à 50')

    thres_txt_pre = "Nous avons pu observer que les images COVID-19 se distinguent. " \
                    "On peut voir certaines indications directement sur les images, " \
                    "mais il faut également tenter de les identifier précisément et les quantifier. " \
                    "Voyant que ces images ont plus de tracés blancs sur les zones sombres, " \
                    "un indicateur potentiel pourrait être la quantité de pixels en dessous d'un certain seuil " \
                    "(i.e la quantité de pixels très sombres, qui devrait être basse dans le cas COVID-19)."
    st.markdown(thres_txt_pre)
    fig = load_eda('thres_inf_50')
    st.pyplot(fig)
    thres_txt_post = "Le résultat obtenu est le résultat attendu. " \
                     "Les images COVID-19 possèdent en effet beaucoup moins de pixels d'intensité basse."
    st.markdown(thres_txt_post)

    # Edge intensity study to see if it represents a relevant indicator
    st.subheader('Intensité au bord')

    edge_txt_pre = "Maintenant que nous avons pu détecter un indicateur encore plus concret de discrimination, " \
                   "nous allons tenter d'isoler (si possible) les zones affectées. " \
                   "Cela pourrait donner des indices pour un choix de preprocessing. " \
                   "Le test effectué ici correspond à l'intensité des pixels sur un cadre au bord de l'image."
    st.markdown(edge_txt_pre)
    fig = load_eda('edge_brightness')
    st.pyplot(fig)
    edge_txt_post = "Nous retrouvons le résultat précédent, cette fois-ci appliqué au bord " \
                    "et non à toute l'image : il semble que les bords des images COVID-19 soient " \
                    "particulièrement plus lumineux."
    st.markdown(edge_txt_post)

    # Completes the edge intensity study
    st.subheader('Intensité au centre')

    center_txt_pre = "Dans ce dernier test, nous procédons au même test que précédemment mais sur la " \
                     "zone complémentaire, c'est-à-dire au centre de l'image."
    st.markdown(center_txt_pre)
    fig = load_eda('center_brightness')
    st.pyplot(fig)
    center_txt_post = "Les différences sont ici beaucoup moins claires. " \
                      "C'est une bonne chose car cela nous permet de nous orienter dans le choix du preprocessing : " \
                      "Il semble que, en ne considérant que le centre de l'image, nous nous débarassons d'un biais " \
                      "sans pour autant perdre de l'information (les bords de l'image ne sont pas les zones " \
                      "qui nous intéressent)."

    # Concluding and introducing statistical results
    st.subheader('Conclusion')

    ccl = "Cette analyse a fait ressortir la présence de biais ainsi que quelques indices pour le preprocessing "\
        "afin de contrer ces biais dont la présence est confirmée non seulement en visualisant des graphes mais "\
        "également par des tests statistiques (ANOVA). La prochaine section est dédiée à l'étude de ces derniers."
    st.markdown(ccl)


# The bias study section
elif add_selectbox == 'Biais':

    text = "Nous avons donc pu constater la présence de forts biais dans le jeu de données." \
           "Il serait intéressant de pouvoir évaluer les biais présents dans le jeu de données." \
           "Dans ce but, deux méthodes seront utilisées (LDA et t-SNE)."
    st.markdown(text)

    # LDA
    st.subheader('LDA')

    lda_txt_pre = "L'algorithme que nous allons utiliser ici est un algorithme de réduction de dimension. " \
              "Il présente la particularité d'utiliser les labels dans son entraînement, ce qui va lui " \
              "permettre de linéairement séparer les classes lors des projections orthogonales. " \
              "On va donc pouvoir quantifier les biais à l'aide d'un algorithme de classification sur les projections."

    st.markdown(lda_txt_pre)

    fig = plot_lda()

    st.pyplot(fig)

    lda_txt_post = "Visuellement, les clusters ressortent même sur l'ensemble de validation. " \
                   "Un RandomForestClassifier non tuné, entraîné sur l'ensemble d'entraînement, obtient un score de 85% de précision."
    st.markdown(lda_txt_post)


    # t-SNE Manifold Learning
    st.subheader('t-SNE')

    tsne_txt_pre = "Passons maintenant à notre deuxième algorithme. " \
               "Celui-ci fonctionne différemment car ses projections ne sont pas linéaires, " \
               "elles sont basées sur la théories des variétés différentielles " \
               "et permettent donc de repérer des formes de clusters plus compliquées."
    st.markdown(tsne_txt_pre)

    fig = plot_tsne()

    st.pyplot(fig)

    tsne_txt_post = "Nous constatons à nouveau des clusters, bien que moins connexes que les précédents."

    st.subheader('Conclusion')

    ccl = "Nous avons donc mis en place deux façons de représenter (et même mesurer) certains biais de notre jeu. " \
          "Cela va nous permettre d'évaluer différents preprocessings et d'en choisir un pour notre modèle final. " \
          "Cependant, cela ne signifie pas forcément que, même si nous trouvons un preprocessing satisfaisant " \
          "vis à vis de nos deux indicateurs, le modèle que nous entraînerons sera parfaitement débiaisé. " \
          "Cela signifiera seulement que les biais que nous avons réussi à amoindrir " \
          "<u>les biais que nous avons réussi à quantifier</u>."

    st.markdown(ccl, unsafe_allow_html=True)


# Studying the various preprocessings to fight off the biases
elif add_selectbox == "Différents preprocessings":

    # The user will select what to visualize
    preprocess_choice = st.selectbox('Choisissez votre version de preprocessing',
                            (
                                'Zoomée',
                                'Filtre d\'intensité',
                                'Filtre de contraste',
                                'Contraste + intensité',
                                'Intensité + contraste',
                                'Filtre CLAHE'
                            ))

    if preprocess_choice == 'Zoomée':
        func = img_zoom

    elif preprocess_choice == 'Filtre d\'intensité':
        func = standardize_img_brightness

    elif preprocess_choice == 'Filtre de contraste':
        func = standardize_img_contrast

    elif preprocess_choice == 'Contraste + intensité':
        func = ctrst_brght

    elif preprocess_choice == 'Intensité + contraste':
        func = brght_ctrst

    elif preprocess_choice == 'Filtre CLAHE':
        func = clahe_preprocessing


    # Actual preprocessing visualization
    if st.button('Affichage du preprocessing'):

        f = plot_img_preprocessing(func=func)
        f.set_size_inches(11, 8)
        st.pyplot(f)

    # Boxplot based on the criterion and versions chosen by the user
    st.subheader('Affichage du boxplot')

    st.markdown('Vous pouvez sélectionner ici le critère de comparaison du diagramme en boîte à afficher, \
                ainsi que les versions du preprocessing.')

    criterion = st.selectbox('Critère',
                             (
                                 'Intensité sur le bord',
                                 'Moyenne',
                                 'Ecart-type'
                             ))

    versions = st.multiselect('Versions',
                            (
                                'Originale',
                                'Zoomée',
                                'Filtre d\'intensité',
                                'Filtre de contraste',
                                'Contraste + intensité',
                                'Intensité + contraste',
                                'Filtre CLAHE'
                            ),
                              default='Originale')

    # Actual plot
    if st.button('Afficher le boxplot'):
        fig = plot_boxplot_preprocess(criterion, versions)
        st.pyplot(fig)

# U-Net section
elif add_selectbox == 'Unet':

    # Intro to U-Net
    unet_txt = 'Le U-Net est un réseau de neurones à convolutions initialement développé pour la segmentation '\
        'd\'images médicales. Vous pouvez en apprendre plus sur son fonctionnement sur le site de '\
        '<a href="https://datascientest.com/u-net">DataScientest</a> ou sur le lien donné dans le menu.'

    st.markdown(unet_txt, unsafe_allow_html=True)

    if st.button('Afficher les images filtrées'):
        fig = display_unet()
        st.pyplot(fig)

    our_unet = 'Dans notre cas, il fut entraîné sur seulement 139 images. Cela laisse donc une marge d\'amélioration'\
        'dans le futur.\n'\
        'Comme on peut le constater, ce n\est pas encore une solution idéale mais cela constitue une piste.'

    st.markdown(our_unet)

    # U-Net visualization
    fig = plot_unet_lda()
    st.pyplot(fig)

# The inference section
elif add_selectbox == 'Prédiction':
    
    # Intro
    prediction_txt = 'Le jeu de données étant extrêmement biaisé, pour tous les preprocessings testés, tous les modèles \
    obtenaient des performances excellentes, avec une précision supérieure à 90%, que ce soit pour l\'ensemble de \
    validation ou celui d\'entraînement. Le preprocessing choisi au final fut donc celui qui semblait le mieux gérer ces \
    biais et le modèle est celui dont l\'interprétabilité (Grad-CAM) semblait la plus pertinente.'

    st.markdown(prediction_txt)

    # File selector for the user to try our model on their own images
    uploaded_file = st.file_uploader("Upload png or jpeg files",type=['png','jpeg'])

    if st.button('Prédiction') and uploaded_file is not None:

        fig, lab, p = predict(uploaded_file)

        st.pyplot(fig)

        st.markdown('Le modèle évalue cette image comme correspondant au label ' + lab + ' avec une certitude de ' + \
        str(round(p*100)) + '%.')

# Conclusion of the project
elif add_selectbox == 'Conclusion':

    ccl_text_1 = 'Il semble que malgré les preprocessings testés, il ne soit pas aisé de se débarasser de tous \
    les biais. En effet, les LDA continuent de montrer que les classes peuvent être majoritairement trouvées par \
    simple réduction de dimension. Cela explique la matrice de confusion que nous obtenons ci-dessous.'
    ccl_text_2 = 'Cette hypothèse de jeu biaisé se confirme à l\'aide d\'un jeu de données supplémentaire \
    (qui contient déjà nos données actuelles mais qui rajoute des images labelisées `COVID-19` et `NORMAL`). \
    Lorsqu\'on observe la matrice de confusion de notre modèle sur ce jeu, on constate qu\'il différencie très \
    mal les images `COVID-19` et `NORMAL` (qui sont les seules classes contenant des images étrangères à notre \
    jeu initial) et que les images `Viral Pneumonia` restent médiocrement reconnues.'

    st.markdown(ccl_text_1)

    cm_test = load_csv('data/cm_test.csv')
    st.dataframe(cm_test)

    st.markdown(ccl_text_2) 

    cm_test = load_csv('data/cm_sup.csv')
    st.dataframe(cm_test)