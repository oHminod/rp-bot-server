# PuLID pour rp-bot

Ce projet fournit le service local de génération d’images de
[`rp-bot`](https://github.com/oHminod/rp-bot). Il transforme l’avatar d’un
personnage et le contexte d’un message en une image SDXL tout en préservant son
identité grâce à PuLID.

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
  - [1. Démarrer PuLID](#1-démarrer-pulid)
  - [2. Configurer le prompt SDXL](#2-configurer-le-prompt-sdxl-dans-rp-bot)
  - [3. Connecter le service d’image](#3-connecter-le-service-dimage)
  - [4. Générer depuis une conversation](#4-générer-depuis-une-conversation)
- [Frontend autonome](#frontend-autonome)
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

Pour développer PuLID, cloner le dépôt :

Dans un terminal :

```bash
git clone --branch dev https://github.com/oHminod/PuLID.git
cd PuLID
```

Pour une installation gérée, extraire `pulid-<version>.tar.gz` dans un nouveau
dossier. L’archive contient les mêmes installateurs et lanceurs que le clone,
sans historique Git, tests, caches, configurations locales ni modèles.

### 2. Installer sur macOS

Depuis un clone de développement, ajouter d’abord la wheel Metal et son manifeste
issus de la même archive PuLID sous `runtime/wheels/` (ou la construire suivant
[RELEASE.md](RELEASE.md)). La wheel n’est pas versionnée dans Git.

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

L’installation recrée systématiquement `.venv` ; fermer le serveur avant de la
relancer. Les modèles valides restent réutilisés. Python **3.11.16** est toujours
installé par uv **0.12.10** dans `PuLID_models/other/uv-python-<plateforme>` ;
les outils du PATH et le Python système ne sont jamais choisis. Le chemin utilisé
contient la version complète, sans dépendre de la jonction mineure `cpython-3.11-*`.
Les binaires uv sont également propres à PuLID, sous `other/uv-<plateforme>-bin`.

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
le Python PuLID. Le chargement CUDA recherche les DLL de la wheel PyTorch locale
et du pilote NVIDIA, sans utiliser un CUDA Toolkit global. Les composants de
l'OS et le pilote graphique restent des prérequis de la machine compatible.

BGE tourne sur GPU Metal avec `n_gpu_layers=-1` sur macOS. Une wheel sans le
backend GPU requis provoque une erreur explicite, sans repli CPU automatique.
Le mode CPU explicite reste disponible ; InsightFace/ONNX peut rester sur CPU.

Après déplacement de PuLID, relancer l’installateur pour reconstruire `.venv` et
ses chemins. Le lanceur détecte un environnement déplacé ou périmé et indique
cette réparation. Un `models_root` interne au projet est enregistré en relatif.
La [recette Windows](WINDOWS_INSTALL_TEST.md) fournit les commandes Git, le test
sans outils préinstallés, la reproduction d’une jonction cassée et le déplacement.

## Utilisation avec rp-bot

`rp-bot` est l’interface principale prévue pour ce service. Il prépare le prompt
à partir du message et de la scène, envoie l’avatar de l’auteur comme référence,
puis conserve l’image générée dans la discussion.

### 1. Démarrer PuLID

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
4. renommez éventuellement le raccourci en **Serveur PuLID**.

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

- `http://127.0.0.1:12693` si PuLID et `rp-bot` tournent sur la même machine ;
- `http://<IP_DU_PC>:12693` si PuLID tourne sur un PC Windows du réseau local.

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
**Serveur PuLID** pour `start_windows.bat` et **Frontend PuLID** pour
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

Si LM Studio sert également au chat, ne remplacez pas son URL : le backend PuLID
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

Elles exposent séparément la version PuLID, dérivée de `pyproject.toml`, et la
version SemVer du contrat API.

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
