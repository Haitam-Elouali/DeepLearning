# Logos de la page de titre

Déposez ici les logos officiels pour qu'ils apparaissent en grand sur la première
page du rapport PDF :

- `logo_emsi.png` (ou `.jpg`, `.pdf`) — logo de l'école ;
- tout autre fichier dont le nom commence par `logo` (partenaire, laboratoire…).

`scripts/build_latex.py` insère automatiquement **tous** les fichiers `logo*`
trouvés dans ce dossier, à une hauteur de 3,4 cm, côte à côte.

Sans fichier, la page de titre affiche uniquement le bloc typographique
« EMSI — École Marocaine des Sciences de l'Ingénieur » : aucun logo n'est
fabriqué à la place de l'original.
