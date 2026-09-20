# rp-bot-server

Ce projet fournit le service local de génération d’images de
[`rp-bot`](https://github.com/oHminod/rp-bot). Il transforme l’avatar d’un
personnage et le contexte d’un message en une image SDXL tout en préservant son
identité grâce à [PuLID](https://github.com/ToTheBeginning/PuLID).

`rp-bot-server` est le nom de ce projet ; PuLID désigne la technologie
d’identité utilisée. Les noms techniques existants (`pulid_app`, `pulid-gen`,
`PuLID_models`, scripts et archives `pulid-<version>.tar.gz`) sont conservés
pour assurer la compatibilité avec les installations et avec `rp-bot`.

À partir de 0.1.2, `PULID_MODELS_ROOT` et l'option Python `--models-root`
désignent le dossier final exact des modèles, quel que soit son nom. Le dossier
est créé s'il manque ; aucun sous-dossier `PuLID_models` n'est ajouté et aucune
question d'emplacement n'est posée. Les chemins relatifs partent de la racine du
projet. `--models-root` prime sur la variable d'environnement, qui prime sur la
configuration locale. Sans chemin explicite, le parcours interactif habituel
(choix d'un parent ou d'un dossier `PuLID_models`) reste disponible.

Le service peut également être utilisé sans `rp-bot` grâce à un frontend web
basique inclus dans le dépôt. Une CLI et une API HTTP sont disponibles pour les
usages avancés. L’ensemble fonctionne sans ComfyUI.

## Sommaire

- [Ce que fournit le projet](#ce-que-fournit-le-projet)
- [Installation rapide](#installation-rapide)
  - [1. Choisir le parcours](#1-choisir-le-parcours)
  - [2. Installer sur macOS](#2-installer-sur-macos)
  - [3. Installer sous Windows](#3-installer-sous-windows)
  - [4. Choisir le checkpoint SDXL](#4-choisir-le-checkpoint-sdxl)
- [Utilisation avec rp-bot](#utilisation-avec-rp-bot)
  - [1. Démarrer rp-bot-server](#1-démarrer-rp-bot-server)
  - [2. Configurer le prompt SDXL](#2-configurer-le-prompt-sdxl-dans-rp-bot)
  - [3. Connecter le service d’image](#3-connecter-le-service-dimage)
  - [4. Générer depuis une conversation](#4-générer-depuis-une-conversation)
- [Frontend autonome](#frontend-autonome)
- [Krea v2 sans identité](#krea-v2-sans-identité)
- [Embeddings de texte pour rp-bot](#embeddings-de-texte-pour-rp-bot)
- [Vérifier l’installation](#vérifier-linstallation)
- [Ajouter un checkpoint SDXL](#ajouter-un-checkpoint-sdxl)
- [Génération en ligne de commande](#génération-en-ligne-de-commande)
- [Mémoire GPU et performances](#mémoire-gpu-et-performances)
- [Fichiers et stockage](#fichiers-et-stockage)
- [Dépannage](#dépannage)
- [Documentation avancée](#documentation-avancée)

## Ce que fournit le projet

| Usage | Interface | Adresse par défaut |
|---|---|---|
| Génération depuis une conversation `rp-bot` | Action PuLID dans le chat | `http://127.0.0.1:12693` |
| Génération manuelle | Frontend web inclus | `http://127.0.0.1:8888` |
| Embeddings de texte pour la mémoire de `rp-bot` | API compatible OpenAI | `http://127.0.0.1:12693/v1` |
| Scripts et automatisations | CLI `pulid-gen` | terminal |

Le backend prend en charge Apple Silicon avec MPS, les GPU NVIDIA avec CUDA et
le CPU. Sur macOS, la détection faciale InsightFace reste exécutée sur CPU.

## Installation rapide

### Prérequis

- une connexion Internet lors de la première installation ;
- aucun Python, uv ni compilateur à préinstaller ;
- sur macOS, un Mac Apple Silicon avec macOS 14 ou supérieur ;
- sous Windows, un GPU NVIDIA et un pilote compatible avec CUDA 13 ;
- au moins 20 Go disponibles, davantage si plusieurs checkpoints SDXL sont
  installés.

Les modèles sont volumineux et ne sont pas versionnés avec le projet.
L’installateur propose par défaut un dossier `PuLID_models` à la racine du
projet, mais permet de choisir un autre emplacement, par exemple sur un SSD
externe. Il y place les checkpoints, PuLID, AntelopeV2, BGE-M3 et tous les
caches lourds.

Git n’est requis que pour le parcours de développement depuis un clone. Une
installation gérée utilise l’archive de release et ne requiert pas Git.

### 1. Choisir le parcours

Pour développer `rp-bot-server`, cloner le dépôt dans le dossier local `PuLID` :

Dans un terminal :

```bash
git clone --branch dev https://github.com/oHminod/rp-bot-server.git PuLID
cd PuLID
```

Pour une installation gérée, télécharger `pulid-<version>.tar.gz` depuis les
[releases de rp-bot-server](https://github.com/oHminod/rp-bot-server/releases),
puis l’extraire dans un nouveau dossier. L’archive contient les mêmes
installateurs et lanceurs que le clone,
sans historique Git, tests, caches, configurations locales ni modèles.

### 2. Installer sur macOS

Le clone inclut la wheel Metal précompilée et son manifeste sous `runtime/wheels/`,
comme l’archive de distribution. Elle est versionnée directement dans Git, sans
Git LFS. Aucune copie manuelle ni compilation locale n’est nécessaire ;
l’installateur vérifie son SHA-256 avant de l’utiliser.

```bash
./install_macos.sh
```

Depuis une archive de release, ou pour reproduire le profil géré depuis un
clone :

```bash
./install_production_macos.sh
# équivalent : ./install_macos.sh --production
```

Le parcours historique reste une installation éditable avec les dépendances de
développement. Le profil production est non éditable et exclut l’extra `dev`.

À la première exécution, acceptez le dossier de modèles proposé ou indiquez un
emplacement personnalisé. Le chemin peut désigner directement un dossier nommé
`PuLID_models` ou son dossier parent.

### 3. Installer sous Windows

Depuis un clone de développement, dans l’Explorateur ou `cmd.exe` :

```bat
install_windows.bat
```

Depuis une archive de release, ou pour reproduire le profil géré depuis un
clone :

```bat
install_production_windows.bat
rem équivalent : install_windows.bat --production
```

Le pare-feu Windows n’est pas modifié par défaut. Pour préparer volontairement
un accès depuis le réseau privé, utilisez l’option avancée
`install_windows.bat --network`, puis confirmez l’ouverture du port `12693`.
Cette option se combine avec `--production` et reste idempotente.

InsightFace 1.0.1 est installé depuis sa wheel officielle. L’installateur exige
cette distribution binaire sur macOS comme sous Windows.

#### Licence d’AntelopeV2

Avant le premier téléchargement, l’installateur indique que les poids
AntelopeV2 sont réservés à la recherche non commerciale et demande une
acceptation explicite. Tout usage commercial exige une licence distincte auprès
d’InsightFace. Consultez les
[conditions officielles](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md).
L’option `--accept-insightface-license` est réservée aux installations
automatisées où ces conditions ont déjà été acceptées.

### 4. Choisir le checkpoint SDXL

L’installateur demande d’abord si vous souhaitez ajouter un modèle SDXL tout de
suite. Vous pouvez différer cette étape : l’installation des autres composants
se termine normalement, mais un checkpoint reste nécessaire avant de générer
une image.

Si vous choisissez de l’ajouter immédiatement, placez votre checkpoint
`.safetensors` dans le dossier `checkpoints` de `PuLID_models` lorsque
l’installateur le demande, ou acceptez le téléchargement de SDXL Base 1.0. Pour
l’ajouter ultérieurement, déposez le fichier dans ce même dossier puis relancez
le script d’installation.

L’installation complète recrée `.venv` ; fermer le serveur avant de la
relancer. Les modèles valides restent réutilisés. Python **3.11.16** est toujours
installé par uv **0.12.10** dans `PuLID_models/other/uv-python-<plateforme>` ;
les outils du PATH et le Python système ne sont jamais choisis. Le chemin utilisé
contient la version complète, sans dépendre de la jonction mineure `cpython-3.11-*`.
Les binaires uv sont également propres au projet, sous `other/uv-<plateforme>-bin`.

Sous Windows, pour mettre à jour une installation existante après avoir récupéré
le nouveau code, fermer le serveur et le frontend puis lancer :

```powershell
.\install_windows.bat --update
```

Ce mode conserve `.venv` et son profil production/développement. Il ajoute les
dépendances manquantes et met à niveau celles dont la version dans `uv.lock` a
changé ; les dépendances déjà conformes et les paquets supplémentaires restent
installés. Le petit paquet applicatif PuLID est réinstallé pour actualiser aussi
le code en profil production. La DLL CPU portable déjà conforme est réutilisée.
Ce mode prépare aussi les petits fichiers partagés de configuration/tokenizer
Qwen3-VL pour Krea : il télécharge ceux qui manquent ou répare ceux dont l'empreinte
est incorrecte. Les fichiers conformes sont réutilisés sans accès réseau.
Il ne télécharge aucun poids de modèle, ne lance ni les tests de génération/BGE,
ni la configuration réseau, et ne modifie pas `config/local.yaml`.

Python et uv doivent déjà correspondre aux versions requises. Un environnement
absent ou incompatible provoque un arrêt explicite, sans recréation automatique ;
l'installation complète reste disponible avec `install_windows.bat` sans option.
`--update` s'utilise seul, sans option de profil ni `--network`.

`uv.lock` fixe les dépendances directes, indirectes et de construction, leurs
sources et SHA-256. Windows utilise PyTorch 2.13.0+cu130, Torchvision 0.28.0+cu130
et llama-cpp-python 0.3.35 CUDA 13.0 avec la DLL CPU portable vérifiée. macOS utilise
notre wheel llama-cpp-python 0.3.35 précompilée avec Metal, fournie dans l’archive.
L’installateur ne résout rien de nouveau (`uv sync --frozen`) et vérifie d’abord
que le manifeste lie bien le verrou au `pyproject.toml`, aux versions des outils
et au manifeste de la wheel. Les outils de construction Python sont eux aussi
verrouillés ; aucun compilateur C/C++ n’est utilisé sur le poste utilisateur.

L'installation télécharge les dépendances verrouillées sans réutiliser de paquets
globaux. Python démarre en mode isolé (`-I`) ; les paquets utilisateur et les
variables `PYTHONPATH`/`PYTHONHOME` n'interviennent pas. uv ignore les configurations
locales héritées et globales (`--no-config`), les options nécessaires étant
passées explicitement. Les lanceurs du backend et du frontend utilisent uniquement
le Python de `rp-bot-server`. Le chargement CUDA recherche les DLL de la wheel
PyTorch locale et du pilote NVIDIA, sans utiliser un CUDA Toolkit global. Les composants de
l'OS et le pilote graphique restent des prérequis de la machine compatible.

BGE tourne sur GPU Metal avec `n_gpu_layers=-1` sur macOS. Une wheel sans le
backend GPU requis provoque une erreur explicite, sans repli CPU automatique.
Le mode CPU explicite reste disponible ; InsightFace/ONNX peut rester sur CPU.

`rp-bot-server` peut être déplacé **sans réinstallation** : arrêtez le backend et le
frontend, déplacez le dossier complet (y compris `.venv` et `PuLID_models`), puis
utilisez les lanceurs habituels. Ils retrouvent le Python géré et réajustent
localement `pyvenv.cfg`, le lien Python macOS et le chemin des sources en mode
développement. Sur Windows, les lanceurs `python.exe` et `pythonw.exe` de `.venv`
proviennent du dossier `Lib/venv/scripts/nt` du Python géré : ils lisent `pyvenv.cfg`
et remplacent les trampolines uv contenant un ancien chemin absolu. Aucun appel à uv,
téléchargement ou changement de dépendances
n’est effectué. Les nouvelles installations utilisent aussi `uv venv --relocatable`.
Un `models_root` interne au projet est enregistré en relatif.

Le déplacement conserve le même OS et la même architecture ; une autre machine
doit satisfaire les mêmes prérequis système/GPU. Les modèles/Python placés dans
un dossier externe restent à leur chemin absolu : déplacer ce dossier séparément,
ou changer sa lettre de lecteur, demande une reconfiguration. Les chemins absolus
personnalisés dans la configuration ou les variables d’environnement ne sont pas
réécrits. Le dossier doit être accessible en écriture au premier démarrage après
le déplacement. Recréez les raccourcis pointant vers l’ancien emplacement.

Pour une installation antérieure à ce mécanisme, mettez les sources à jour et
lancez une fois `rp-bot-server` **avant de déplacer le dossier**, afin d’enregistrer le
chemin portable du Python géré. Après un déplacement, passez d’abord par le
lanceur backend ou frontend avant d’utiliser directement `.venv` en ligne de
commande. Une modification des versions ou du verrou exige toujours une
réinstallation ; un déplacement seul conserve les paquets déjà installés.

Si une installation utilisant déjà `.venv/pulid-python-path` échoue après déplacement
avec `uv trampoline failed to spawn Python child process`, mettre les sources à
jour puis relancer `start_windows.bat` suffit à appliquer ce correctif localement.
Il utilise le Python géré déplacé, sans recréer `.venv` ni toucher aux modèles.

## Utilisation avec rp-bot

`rp-bot` est l’interface principale prévue pour ce service. Il prépare le prompt
à partir du message et de la scène, envoie l’avatar de l’auteur comme référence,
puis conserve l’image générée dans la discussion.

### 1. Démarrer rp-bot-server

Sur macOS :

```bash
./start_pulid_server.sh
```

Sur Windows, le plus simple est de double-cliquer sur `start_windows.bat` dans
l’Explorateur de fichiers. Le script ouvre lui-même sa fenêtre de terminal.
L’exécution depuis `cmd.exe` reste également possible :

```bat
start_windows.bat
```

Le serveur écoute par défaut sur `127.0.0.1:12693`, y compris sous Windows. Il
n’est donc accessible que depuis la machine locale.
Le script Windows active automatiquement `model_cpu_offload` pour Krea v2,
adapté aux cartes de 12 Go. Aucun argument supplémentaire n'est nécessaire.
`--krea2-offload none` permet de le désactiver explicitement ; les réglages
SDXL/BGE restent indépendants.

Pour un réseau privé de confiance uniquement, après configuration explicite du
pare-feu, le mode avancé Windows écoute sur `0.0.0.0` et active CORS ouvert :

```bat
start_windows.bat --network
```

L’équivalent direct sur macOS est explicite lui aussi :

```bash
./start_pulid_server.sh --network
```

Pour créer un raccourci sur le Bureau sans déplacer le script :

1. faites un clic droit sur `start_windows.bat` ;
2. sous Windows 11, choisissez **Afficher plus d’options** ;
3. choisissez **Envoyer vers > Bureau (créer un raccourci)** ;
4. renommez éventuellement le raccourci en **rp-bot-server**.

Conservez le fichier `.bat` dans le dossier du projet : créez un raccourci au
lieu de le copier sur le Bureau. Si le dossier PuLID est déplacé, recréez le
raccourci.

Laissez ce terminal ouvert pendant l’utilisation de `rp-bot`. `Ctrl+C` arrête
le service.

### 2. Configurer le prompt SDXL dans rp-bot

Dans `rp-bot`, ouvrez **Réglages > Tâches LLM et prompts**, puis configurez la
tâche **Prompt d’image SDXL** avec un fournisseur et un modèle de texte actifs.
Cette tâche transforme le message de roleplay en prompt visuel adapté au
checkpoint sélectionné.

### 3. Connecter le service d’image

Ouvrez **Réglages > Modèles d’image**, puis la section **Génération SDXL avec
PuLID** :

1. activez la génération SDXL avec PuLID ;
2. renseignez l’URL du serveur ;
3. choisissez le checkpoint, la méthode de sampling et les autres paramètres ;
4. cliquez sur **Enregistrer**.

Utilisez l’une de ces adresses :

- `http://127.0.0.1:12693` si `rp-bot-server` et `rp-bot` tournent sur la même machine ;
- `http://<IP_DU_PC>:12693` si `rp-bot-server` tourne sur un PC Windows du réseau local.

Le badge **Catalogue disponible** confirme que `rp-bot` atteint le serveur. Le
bouton **Actualiser** recharge la liste des checkpoints et des samplers.

### 4. Générer depuis une conversation

Le personnage doit disposer d’un avatar contenant un visage exploitable. Dans
le chat, après un message du personnage :

1. survolez ou sélectionnez la bulle du message ;
2. cliquez sur l’action au visage intitulée **Générer avec PuLID** ;
3. attendez la fin de la génération.

Les réglages rapides **Génération SDXL PuLID** sont également disponibles dans
le panneau du chat. L’option **Inspecter le prompt** permet de relire et modifier
les prompts positif et négatif avant leur envoi.

L’image finale apparaît dans la galerie de la conversation. `rp-bot` la stocke
dans son propre dossier `.local-data/generated-images` avec le checkpoint, le
sampler, les sigmas et la seed réellement utilisés.

## Frontend autonome

Le frontend inclus permet de générer une image sans lancer `rp-bot`. Démarrez
d’abord le backend, puis ouvrez un second terminal.

Sur macOS, dans deux terminaux distincts :

```bash
# Terminal 1
./start_pulid_server.sh

# Terminal 2
./start_frontend_macos.sh
```

Sur Windows, double-cliquez successivement sur les deux scripts ; chacun ouvre
sa propre fenêtre de terminal :

```bat
rem Terminal 1
start_windows.bat

rem Terminal 2
start_frontend_windows.bat
```

Vous pouvez créer de la même manière deux raccourcis sur le Bureau, par exemple
**rp-bot-server** pour `start_windows.bat` et **Frontend rp-bot-server** pour
`start_frontend_windows.bat`. Démarrez toujours le serveur avant le frontend.

Ouvrez ensuite [http://localhost:8888](http://localhost:8888).

Le formulaire demande une image de référence, le nom du personnage, un prompt
et un checkpoint. Il permet aussi de régler le prompt négatif, Clip Skip 2, le
CFG, les steps, la force d’identité, le sampler, les sigmas et la seed.

Les derniers réglages sont enregistrés dans le `localStorage` du navigateur.
La photo de référence est conservée dans `IndexedDB` et automatiquement
restaurée au prochain chargement de la page. Le bouton **Oublier la photo**
supprime cette référence sans modifier les autres réglages. Le lien
**Effacer les données locales** retire la référence et l’ensemble des réglages
du navigateur.

Le PNG généré n’est jamais persisté : il reste disponible pour l’aperçu et le
téléchargement uniquement jusqu’au rechargement ou à la fermeture de la page.

Contrairement à la CLI, le frontend ne crée pas automatiquement de fichier dans
`outputs/`. Les données persistées restent propres au navigateur et à l’adresse
`http://localhost:8888`.

Pour cibler un backend situé ailleurs :

```bash
./start_frontend_macos.sh --backend-url http://192.168.1.20:12693
```

L’équivalent Windows accepte la même option :

```bat
start_frontend_windows.bat --backend-url http://192.168.1.20:12693
```

## Embeddings de texte pour rp-bot

Le même serveur expose BGE-M3 au format OpenAI pour la mémoire vectorielle de
`rp-bot` :

- `GET /v1/models` liste le modèle `text-embedding-bge-m3` ;
- `POST /v1/embeddings` calcule des vecteurs de 1024 dimensions.

Si le fournisseur **LM Studio local** de `rp-bot` sert uniquement aux
embeddings, configurez-le ainsi dans **Réglages > Fournisseurs** :

```text
URL de base : http://127.0.0.1:12693/v1
Clé API : vide
Activé : oui
```

Puis configurez la tâche **Embeddings locaux** dans **Réglages > Tâches LLM et
prompts** :

```text
fournisseur : LM Studio local
modèle : text-embedding-bge-m3
format : embedding
activé : oui
```

Si LM Studio sert également au chat, ne remplacez pas son URL : `rp-bot-server`
n’expose pas `/v1/chat/completions`. La procédure pour séparer les fournisseurs
et reconstruire les index LanceDB est détaillée dans
[`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md).

## Vérifier l’installation

Les lanceurs appellent directement le Python de `.venv`, après contrôle de sa
provenance et des chemins. Pour vérifier l’installation macOS :

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pulid_app.cli doctor --allow-missing-sdxl
.venv/bin/python scripts/inspect_models.py --show-cache-env --fail-on-internal-cache --allow-missing-sdxl
```

Sous Windows :

```bat
.venv\Scripts\python.exe scripts\check_environment.py
.venv\Scripts\python.exe -m pulid_app.cli doctor --allow-missing-sdxl
```

`doctor` contrôle les checkpoints, les modèles de visage, PuLID, BGE-M3, les
permissions, le device et les dépendances critiques sans lancer une génération
complète.

Pour la supervision gérée, les routes suivantes ne chargent aucun modèle :

```bash
curl http://127.0.0.1:12693/health
curl http://127.0.0.1:12693/version
curl http://127.0.0.1:12693/capabilities
```

Elles exposent séparément la version de `rp-bot-server`, dérivée de
`pyproject.toml`, et la version SemVer du contrat API.

## Ajouter un checkpoint SDXL

Déposez le fichier `.safetensors` dans :

```text
<PuLID_models>/checkpoints/
```

Le VAE du checkpoint par défaut
`realvisxlV50_v50LightningBakedvae.safetensors` est intégré : aucun VAE externe
n’est nécessaire. Le serveur charge toujours un fichier local explicite et ne
télécharge jamais implicitement un modèle SDXL pendant une génération.

Après l’ajout, utilisez **Actualiser** dans `rp-bot` ou rechargez le frontend
autonome. Le nom du modèle est affiché sans l’extension `.safetensors`.

## Génération en ligne de commande

La CLI est utile pour tester le pipeline indépendamment des interfaces web :

```bash
pulid-gen generate \
  --reference inputs/noemie.webp \
  --character noemie \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --method dpmpp_2m_sde \
  --sigmas karras \
  --cfg 4.5 \
  --strength 0.8 \
  --steps 20 \
  --seed 42
```

La commande écrit le PNG et un manifeste JSON adjacent dans `outputs/`. Pour
sélectionner un autre checkpoint, ajoutez `--model NOM_DU_FICHIER` sans
`.safetensors`.

## Mémoire GPU et performances

Le serveur garde le pipeline chargé entre les requêtes CUDA utilisant le même
checkpoint. Les modes suivants permettent d’arbitrer entre vitesse et mémoire,
notamment lorsque SDXL et BGE-M3 partagent le GPU :

| Démarrage | Comportement |
|---|---|
| sans option | BGE-M3 et SDXL utilisent le GPU sans offload ; calculs CUDA concurrents |
| `--serialized-cuda` | conserve les modèles sur le GPU mais sérialise leurs calculs |
| `--partial` | déplace les encodeurs CLIP et le VAE SDXL sur CPU pendant un embedding |
| `--full` | décharge tout le pipeline SDXL pendant un embedding |
| `--CPU` | exécute BGE-M3 sur CPU et laisse SDXL sur le GPU |

Exemples :

```bash
./start_pulid_server.sh --partial
```

```bat
start_windows.bat --serialized-cuda
start_windows.bat --CPU
```

Si le mode par défaut provoque une erreur de mémoire CUDA, essayez d’abord
`--serialized-cuda`, puis `--partial` ou `--full`.

## Fichiers et stockage

```text
PuLID/
├── outputs/                 # images et JSON créés par la CLI
├── cache/identity/          # petits caches d’identité ArcFace
└── config/local.yaml        # configuration locale générée, ignorée par Git

<PuLID_models>/
├── checkpoints/            # checkpoints SDXL locaux
├── antelopev2/             # modèles InsightFace
├── text_embedding/         # BGE-M3 au format GGUF
├── sources/PuLID/          # code officiel épinglé
├── huggingface/            # cache Hugging Face externe
├── torch/                  # cache PyTorch externe
└── other/                  # autres caches lourds
```

Le backend HTTP conserve le PNG en mémoire et le renvoie au client. Il ne crée
pas d’image ni de manifeste dans `outputs/`. Le frontend ou `rp-bot` est
responsable de l’enregistrement du PNG reçu.

## Dépannage

| Symptôme | Action recommandée |
|---|---|
| uv affiche `everything's installed!` puis `Installation de uv ... impossible` | Mettre à jour `dev` et relancer `install_windows.bat --development`. L'ancien bootstrap pouvait lire un `$LASTEXITCODE` vide ou périmé après le script PowerShell. Le contrôle vérifie désormais le fichier uv puis son code de sortie et sa version. Si uv est déjà installé, relancer l'ancien installateur permet également de dépasser ce premier blocage. |
| Catalogue PuLID indisponible dans `rp-bot` | Vérifier que le serveur est démarré et que l’URL se termine par `:12693`, sans `/v1` |
| Serveur distant inaccessible | Sur un réseau privé uniquement, exécuter `install_windows.bat --network`, puis `start_windows.bat --network` et utiliser l’IPv4 privée affichée |
| Aucun visage détecté | Choisir un avatar net, de face et suffisamment grand |
| Plusieurs visages détectés | Recadrer l’avatar afin qu’un seul visage soit visible |
| Checkpoint introuvable | Vérifier le fichier sous `<PuLID_models>/checkpoints/`, puis actualiser le catalogue |
| Erreur de mémoire CUDA | Redémarrer avec `--serialized-cuda`, `--partial` ou `--full` |
| `llama.dll` ou erreur Windows `0xc000001d` | Mettre à jour le pilote NVIDIA puis relancer `install_windows.bat` |
| Installation incomplète | Relancer l’installateur, puis exécuter `pulid-gen doctor` |

Le serveur ne possède ni authentification ni gestion d’utilisateurs. Ne
l’exposez pas à Internet ; limitez son accès à la machine locale ou à un réseau
privé de confiance.

## Documentation avancée

- [`API_FRONTEND_INTEGRATION.md`](API_FRONTEND_INTEGRATION.md) : contrat HTTP,
  paramètres de génération et exemples d’intégration ;
- [`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md) :
  configuration détaillée de BGE-M3 dans `rp-bot` ;
- [`PULID_CODEX_IMPLEMENTATION_PLAN.md`](PULID_CODEX_IMPLEMENTATION_PLAN.md) :
  architecture, phases d’implémentation et validation technique.
- [`RELEASE.md`](RELEASE.md) : archive déterministe, SHA-256, profil production
  sans Git et procédure de publication.

Pour exécuter les tests unitaires, sans réseau ni modèle lourd :

```bash
pytest -m unit
```

AntelopeV2 est distribué pour la recherche non commerciale. Consultez la
licence InsightFace avant tout autre usage.


## Krea v2 sans identité

Le serveur fournit `POST /generate/krea2`, basé sur le workflow
`krea2_simple.json` (copie fournie dans Downloads ; `/mnt/data` absent sur macOS).
Il utilise Krea 2, Qwen3-VL-4B et le VAE Qwen Image via Diffusers/Transformers,
sans ComfyUI, PuLID ni InsightFace. Les dépendances d'inférence verrouillées du
projet incluent déjà les classes nécessaires.

Sur une installation existante, préparer uniquement les composants Krea :

```bash
.venv/bin/pulid-install --krea2-only --models-root /Volumes/SSD/Documents/PuLID_models
```

Sous Windows, utiliser `.venv\Scripts\pulid-install.exe` et la racine externe
choisie à l'installation. L'installation générale prépare aussi ces dossiers.
Le mode `--krea2-only` conserve les autres réglages de la configuration locale.

```text
PuLID_models/
├── krea2/checkpoints/
│   └── krea2.safetensors                   # fourni manuellement
├── text_encoders/qwen3vl/
│   ├── qwen3vl_4b_bf16.safetensors         # fourni manuellement
│   └── config/                            # préparé automatiquement, partagé
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       └── chat_template.json
└── vae/
    └── qwen_image_vae.safetensors          # téléchargé uniquement si absent
```

Les six fichiers de `config/` sont téléchargés automatiquement depuis
[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)
à une révision épinglée, avec vérification SHA-256, pendant l'installation
complète, `pulid-install --krea2-only` et `install_windows.bat --update`.
Ce dossier est partagé par tous les encodeurs **Qwen3-VL-4B compatibles**,
quelle que soit leur quantification prise en charge ou leur nom de fichier.
Il suffit de déposer vos poids dans `text_encoders/qwen3vl/` : aucune copie
manuelle de configuration n'est nécessaire. Cela ne rend pas les variantes
Qwen d'autres architectures ou tailles compatibles avec Krea.

Seuls les fichiers manquants ou dont l'empreinte est incorrecte sont téléchargés
ou remplacés ; une configuration complète et conforme ne nécessite aucune
requête réseau. Le chemin `krea2.text_encoder_config_dir` et sa surcharge
d'environnement sont respectés. Pour préparer uniquement ces fichiers sur une
installation existante, sans toucher au YAML, au VAE ni aux dépendances :

```powershell
.\.venv\Scripts\python.exe -m pulid_app.installer --qwen3vl-config-only
```

Sur macOS : `.venv/bin/python -m pulid_app.installer --qwen3vl-config-only`.
Le checkpoint Krea et les poids Qwen3-VL ne sont **jamais téléchargés automatiquement**.
Le VAE est téléchargé depuis
[Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/qwen_image_vae.safetensors),
à une révision épinglée, avec contrôle SHA-256. Un fichier déjà présent et invalide
est signalé ; il n'est pas remplacé automatiquement. Les requêtes HTTP restent hors ligne.

Formats pris en charge : checkpoints `.safetensors` **BF16, FP16 ou FP32**,
**NVFP4 Comfy** (E2M1 compacté, échelles par blocs de 16 et échelle globale),
et **FP8 E4M3 scaled Comfy**. Les clés peuvent être natives Krea 2 ou Diffusers ;
l'encodeur Qwen3-VL-4B doit être un fichier unique avec les clés de sa branche
texte HF ou ComfyUI. Le VAE reste en BF16/FP16/FP32.

Les noms ci-dessus sont des exemples et des préférences de configuration :
**aucun renommage n'est nécessaire**. Placez vos checkpoints `.safetensors`
directement dans `krea2/checkpoints/` et vos encodeurs dans
`text_encoders/qwen3vl/`, avec leurs noms d'origine. Les extensions en majuscules,
les espaces et les accents sont acceptés. `config/` reste le dossier partagé
de configuration/tokenizer Qwen3-VL-4B-Instruct.

`GET /models/krea2` expose deux listes `models` et `text_encoders`, avec le nom
complet de chaque fichier et un indicateur `default`. Le fichier configuré est
prioritaire s'il existe ; sinon le premier fichier dans l'ordre alphabétique
insensible à la casse sert de défaut. L'inventaire est actualisé à chaque appel,
sans téléchargement ni lecture des poids. Les sous-dossiers ne sont pas parcourus.
Pour rp-bot, envoyer les noms choisis dans les champs `model` et `text_encoder`
de `POST /generate/krea2` ; voir [le contrat API](API_FRONTEND_INTEGRATION.md#génération-krea-v2-sans-identité).

Le format est identifié par le contenu, indépendamment du nom du fichier.
Les architectures et formats compatibles restent ceux décrits ci-dessus.
Les déclarations `_quantization_metadata` version 1.0 et les marqueurs
`comfy_quant` sont reconnus, y compris lorsque seul le nom des poids porte
le préfixe `model.diffusion_model.`. Une échelle absente, des dimensions
incompatibles ou un format inconnu produisent une erreur explicite avant
le chargement du composant. INT8, ConvRot, MXFP8 et les anciennes variantes
FP8 sans ces déclarations ne sont pas pris en charge.

**Les poids restent quantifiés en RAM et en VRAM.** Les couches NVFP4 sont
déquantifiées temporairement au moment de leur calcul ; aucune copie complète
du modèle en BF16/FP16 n'est créée, ni conservée sur disque. Les couches FP8
utilisent le calcul CUDA natif sur Ada (dont RTX 4070 SUPER) ou plus récent
si le noyau et les dimensions le permettent, avec un repli par couche.
Le marqueur `full_precision_matrix_mult` est respecté. Cette stratégie suit
le principe de stockage compact et de calcul à la demande de ComfyUI, sans
en importer le runtime. Sur CUDA, le décodage par couche utilise les noyaux
précompilés **Comfy Kitchen 0.2.35**, également utilisés par ComfyUI. Un noyau
absent produit une erreur explicite au chargement ; aucun repli vers la boucle
PyTorch lente n'est effectué sur CUDA. Les multiplications NVFP4 natives de
Blackwell ne sont pas utilisées. L'attention Krea CUDA sélectionne explicitement
un noyau fusionné (Flash, memory-efficient ou cuDNN) et journalise son nom.
Les têtes K/V sont dépliées lorsque Flash ne prend pas en charge le GQA,
notamment avec un masque. Le backend `math` est interdit pour ces appels :
dans PyTorch 2.13 sur Ada, il est prioritaire sur cuDNN et peut matérialiser
plusieurs Gio de matrices d'attention, même quand cuDNN est disponible.
Si aucun noyau fusionné n'est utilisable, la génération échoue explicitement
avec les dimensions concernées et la commande de diagnostic. Les réglages
SDPA précédents sont restaurés après chaque appel ; CPU et MPS gardent leur chemin habituel.

Après une mise à jour du code, lancer `install_windows.bat --update` pour installer
les dépendances verrouillées, dont Comfy Kitchen. Cela ne télécharge pas les
poids Krea ni Qwen3-VL. Le lancement Windows sélectionne déjà l'offload Krea
par composant, adapté à une carte CUDA de 12 Go :

```powershell
.\start_windows.bat
```

Le script prépare les bibliothèques CUDA et écoute sur `127.0.0.1:12693`,
l'adresse attendue par le frontend léger. L'exécutable `pulid-server` utilise
également `12693` par défaut ; `--port` permet de le modifier explicitement.

Ce mode charge successivement Qwen, Krea puis le VAE. Si un composant compacté
et la marge de calcul estimée dépassent la VRAM libre, Krea utilise l'offload
par sous-module d'Accelerate. Les transferts des couches quantifiées restent
compactés. La marge est une estimation, pas une garantie pour toutes les
résolutions ; réduire la résolution si la mémoire manque. `--krea2-offload none`
conserve tous les composants Krea sur le device. L'option `--krea2-offload`
s'applique uniquement à Krea ; si elle est omise, Krea hérite de `--offload`
ou de la configuration. MPS et CPU utilisent `none`, avec déquantification
temporaire par couche (CPU en float32).

Sur CUDA, le serveur conserve le pipeline après une génération réussie et le
réutilise pour les requêtes Krea suivantes, même si le prompt, la seed ou la
résolution changent. Il ne relit pas les poids et ne vide pas systématiquement
le cache CUDA. Avec l'offload par composant, les composants restent en VRAM
tant que leur coexistence laisse la marge de calcul estimée. Krea peut ainsi
rester sur le GPU pendant et après le décodage VAE. Avant d'utiliser un autre
composant, le serveur vérifie la mémoire libre et ne déplace des poids sur CPU
que si nécessaire, en privilégiant la conservation du débruiteur. Aucun
déchargement systématique n'a lieu au début ou à la fin de chaque image.
Le terminal indique les composants conservés en VRAM et les évictions.

Deux encodages texte au maximum sont gardés en RAM (prompt et branche CFG vide),
sans fichier sur disque. Relancer le même prompt avec une autre seed, résolution
ou nombre de steps évite ainsi de rappeler Qwen et, si Krea et le VAE tiennent
ensemble, de retransférer leurs poids. La longueur du conditionnement fait
partie de la clé du cache ; changer de checkpoint ou d'encodeur le réinitialise.
Sur 12 Go, un nouveau prompt peut toujours nécessiter d'évincer Krea pour charger
Qwen. Les très gros modèles utilisent encore l'offload par sous-module et ne
restent donc pas intégralement en VRAM. Aucune réinstallation n'est nécessaire.

Krea est libéré avant une génération SDXL ou un embedding BGE sur GPU, après
une erreur, et à l'arrêt du serveur. Un embedding BGE sur CPU ou une requête
de découverte (`/health`, `/models`, `/capabilities`) ne l'évince pas.
CPU/MPS gardent le nettoyage après chaque requête. Les images restent uniquement
dans la réponse HTTP en mémoire.

Le terminal distingue le chargement, l'encodage Qwen, chaque step de diffusion
et le décodage VAE, avec leurs durées. Si le premier step semble bloqué,
relever ces lignes et l'utilisation GPU pendant l'attente :

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

Ce benchmark compare les décodeurs CUDA et PyTorch sur une matrice synthétique,
sans charger de modèle ni générer de fichier. Il ne mesure pas la génération
complète et ne garantit pas une durée identique à ComfyUI :

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_krea2_cuda.py
```

Pour mesurer **l'attention seule** aux dimensions du workflow 1248×832
(cas maximal de 5080 tokens : 4056 image + 1024 texte, 48 têtes Q et 12 têtes K/V), arrêter le serveur puis lancer :

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_krea2_cuda.py --attention
```

Le JSON indique `attention_ms`, le pic de mémoire allouée par PyTorch et
`sdpa_operators`, les opérateurs réellement exécutés, relevés par le profiler
sans fichier de trace. On attend un opérateur `efficient_attention`,
`flash_attention` ou `cudnn_attention`, jamais `attention_math`.
Ce test ne charge aucun poids et n'est pas une mesure du step complet.
Un correctif du code d'attention ne change pas les dépendances : dans une
installation de développement, récupérer le code puis redémarrer suffit.

Les quatre chemins sont centralisés dans la section `krea2` de
`config/default.yaml` et `config/local.yaml`, relatifs à `models_root`.
Ils peuvent être surchargés par `PULID_KREA2_CHECKPOINT`,
`PULID_KREA2_TEXT_ENCODER`, `PULID_KREA2_TEXT_ENCODER_CONFIG_DIR` et
`PULID_KREA2_VAE`. Les chemins absolus doivent rester sous la racine des modèles.
Une ancienne configuration sans section `krea2` reçoit ces chemins par défaut.

Dans le **frontend léger**, sélectionnez « Krea v2 · texte vers image » dans
« Moteur de génération ». Le portrait, le personnage et les contrôles d'identité
disparaissent. Choisissez « Modèle Krea v2 » et « Encodeur Qwen3-VL » dans les
réglages avancés ; le bouton de reconnexion actualise les listes après l'ajout
de fichiers. Les sélections sont mémorisées séparément de SDXL. Sur CUDA, la même
paire réutilise le pipeline ; changer l'un des deux fichiers le recharge.
Entrez le prompt puis générez : valeurs initiales 1248 × 832,
10 steps, CFG 1, Euler, Beta, denoise 1. Les réglages de chaque moteur sont
mémorisés séparément. Krea reste accessible même si aucun checkpoint SDXL n'est
installé. L'aperçu n'est pas sauvegardé automatiquement ; le bouton de
téléchargement reste disponible.

Krea utilise un plafond de **1024 tokens** (environ 1019 pour le texte et 5 pour
le suffixe), avec une limite de **8000 caractères** dans l'API et le frontend.
Les tokens au-delà de 512 participent à la génération ; le dépassement de la
fenêtre de 1024 reste tronqué automatiquement. La longueur est calculée pour
chaque prompt : un texte de 100 tokens produit 105 positions de conditionnement,
sans remplissage jusqu'à 1024. Qwen et la diffusion traitent cette longueur réduite ;
le préfixe système, le suffixe assistant et les tokens utiles sont conservés.
Le terminal affiche la longueur effective. Avec CFG différent de 1, la branche
négative vide est remplie à cette même longueur pour garder les positions alignées.
Les prompts longs demandent davantage de calcul et de mémoire. SDXL conserve
ses propres limites. Le benchmark d'attention ci-dessus reste un test au plafond
de 1024 positions, pas une mesure représentative de tous les prompts.

L'API renvoie le PNG directement en mémoire, sans écrire dans `outputs/` ni créer
de JSON/cache d'identité. **rp-bot conserve les images reçues**, comme avec SDXL.
Voir [le contrat HTTP](API_FRONTEND_INTEGRATION.md#génération-krea-v2-sans-identité)
pour les champs, les limites, les en-têtes et les erreurs.

Références techniques : [Krea 2 officiel](https://github.com/krea-ai/krea-2),
[pipeline Diffusers](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/krea2/pipeline_krea2.py),
[scheduler beta de référence](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py),
[format NVFP4](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/nvfp4.md),
[noyaux Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen),
[sélection SDPA PyTorch 2.13](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp),
[priorités SDPA par défaut](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/Context.h).
Les tests utilisent des composants factices et de petits composants Diffusers
avec des poids synthétiques, y compris les formats quantifiés et l'offload.
Pour vérifier les noyaux CUDA sur le PC, les tests ne téléchargent aucun modèle :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_quantized_krea2.py tests/test_krea2_cuda_performance.py -q
```

Les tests GPU sont ignorés lorsque le matériel correspondant est absent.
Une génération avec les poids réels nécessite leur installation manuelle.
