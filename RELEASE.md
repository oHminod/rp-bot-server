# Publier PuLID

`pyproject.toml` est l’unique source de la version PuLID. Le nom de l’archive,
sa racine, `release-metadata.json` et les commandes de construction lisent tous
`project.version`; aucune version applicative n’est recopiée dans les scripts.

## Construire les artefacts

Depuis un clone ou depuis une précédente archive extraite :

```bash
python scripts/build_release.py
```

La commande crée :

```text
dist/pulid-<version>.tar.gz
dist/checksums-sha256.txt
```

L’archive est déterministe pour un même contenu : chemins triés, date à l’époque
Unix, propriétaires normalisés et en-tête gzip stable. Elle contient les sources,
le frontend, la configuration par défaut et les installateurs, mais exclut
notamment `.git`, `.github`, `tests`, les scripts `test_*`, environnements
virtuels, caches, `config/local.yaml`, entrées/sorties locales et fichiers de
modèles (`.safetensors`, `.onnx`, `.gguf`, `.pt`, `.pth`, `.ckpt`, `.bin`).

Vérifier l’empreinte sur macOS ou Linux :

```bash
cd dist
shasum -a 256 -c checksums-sha256.txt
```

Sur Windows PowerShell :

```powershell
$line = (Get-Content dist\checksums-sha256.txt).Split(' ', 2)
(Get-FileHash -Algorithm SHA256 (Join-Path dist $line[1].Trim())).Hash.ToLowerInvariant() -eq $line[0]
```

## Vérifier l’installation sans Git

Extraire l’archive dans un nouveau dossier, puis utiliser le profil production :

```bash
./install_production_macos.sh
```

ou sous Windows :

```bat
install_production_windows.bat
```

Ces wrappers réutilisent respectivement `install_macos.sh --production` et
`install_windows.bat --production`. Le profil production installe les extras
`inference`, `pulid`, `server` et `embeddings` sans installation éditable et sans
extra `dev`. Les modèles et `config/local.yaml` restent externes à l’archive ;
les installateurs réutilisent les fichiers valides déjà présents et ne les
téléchargent de nouveau que s’ils manquent ou échouent à leur contrôle
d’intégrité.

## Première GitHub Release

Avant une première release publique :

1. choisir et ajouter la licence du dépôt ;
2. vérifier l’état juridique de la redistribution du code, sans ajouter aucun
   modèle ou poids à l’archive ;
3. exécuter les tests unitaires et les validations documentées dans le README ;
4. reconstruire les artefacts sur un commit propre et vérifier le SHA-256 ;
5. signer l’archive et `checksums-sha256.txt` selon la politique de signature du
   canal stable ;
6. créer seulement ensuite le tag et la GitHub Release avec l’autorisation du
   propriétaire.

Pour préparer le nom du tag sans dupliquer la version :

```bash
VERSION="$(python -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')"
echo "v${VERSION}"
```

La création du tag, le push et la publication ne font volontairement pas partie
de `build_release.py`.

## Wheel macOS précompilée et verrou d'installation

L'archive inclut désormais `runtime/wheels/llama_cpp_python-0.3.35-py3-none-macosx_11_0_arm64.whl`,
son manifeste SHA-256, `uv.lock`, `runtime/lock-manifest.json`, `.python-version`
et `.uv-version`. `build_release.py` refuse une wheel absente ou corrompue.
Aucun poids de modèle n'est inclus. La wheel conserve les licences embarquées
par la distribution amont.

Sur le Mac de préparation uniquement, avec Xcode/Command Line Tools installé :

```bash
UV="${PWD}/PuLID_models/other/uv-macos-bin/uv"
PYTHON="${PWD}/PuLID_models/other/uv-python-macos/cpython-3.11.16-macos-aarch64-none/bin/python3.11"
export UV_PYTHON_INSTALL_DIR="${PWD}/PuLID_models/other/uv-python-macos"
export UV_CACHE_DIR="${PWD}/PuLID_models/other/uv-macos"
"${PYTHON}" scripts/build_macos_wheel.py --uv "${UV}"
"${PYTHON}" scripts/lock_environment.py --uv "${UV}"
"${PYTHON}" scripts/build_release.py
```

Adapter ces chemins si les outils de préparation sont dans un autre dossier.
Le script exige les versions exactes de Python et uv du projet, télécharge la
source PyPI 0.3.35 avec SHA-256 vérifié et installe les outils listés dans
`requirements/wheel-build.txt` avec vérification de leurs hashes. Il compile en
arm64, avec `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`,
`GGML_ACCELERATE=ON` et `GGML_NATIVE=OFF`. Il n'utilise pas les éventuels CMake ou
Ninja du PATH : leurs versions sont verrouillées dans un environnement temporaire.
Le compilateur Apple et la version de macOS sont enregistrés dans le manifeste.
La construction n'est pas promise identique bit pour bit entre versions de Xcode ;
la wheel distribuée est un artefact unique identifié par son empreinte.

Toute reconstruction de la wheel doit être suivie de `lock_environment.py`,
puis d'une installation et d'un calcul BGE Metal réels avant distribution.
Les empreintes des fichiers texte normalisent CRLF en LF pour les clones Windows.
Le manifeste permet `uv sync --frozen` sans résoudre des sources d'une autre
plateforme : Windows n'a pas besoin du binaire macOS dans son clone Git.
Les versions de Python disponibles sont figées par release uv ; la sélection
explicite et le mode géré suivent la [documentation uv](https://docs.astral.sh/uv/concepts/python-versions/).

La wheel reste un artefact non suivi par Git. Un clone macOS doit la recevoir
avec le manifeste correspondant depuis la même archive, ou la reconstruire et
régénérer le verrou. L'utilisateur final de l'archive n'a aucune compilation à
faire et n'a besoin ni de Python système, ni de uv global, ni de Xcode.

## Validation de cette tranche (5 septembre 2026)

- Python géré 3.11.16, uv 0.12.10 ; environnement de validation séparé sous `/tmp`.
- Wheel Metal construite sur macOS 15.7.7, Apple M1 Max, Apple clang 17.0.0.
- Installation de toutes les dépendances verrouillées, `uv pip check` et imports
  réels de torch/torchvision, transformers, diffusers, accelerate, InsightFace,
  ONNX Runtime, FaceXLib, timm et llama-cpp-python.
- Archive extraite et installée en production dans `/tmp`, dossier complet
  déplacé vers un chemin contenant des espaces : refus du vieux `.venv`,
  recréation réussie et suppression d'une sentinelle de l'ancien environnement.
  Après réparation : `/health` et `/models` HTTP 200 sans SDXL, puis
  `/v1/embeddings` HTTP 200 avec BGE Metal (1024 dimensions).
- `scripts/inspect_models.py --show-cache-env --fail-on-internal-cache
  --allow-missing-sdxl` réussi sur les modèles existants en lecture seule, avec
  les sorties de vérification redirigées dans `/tmp`.
- BGE-M3 Q8_0 existant chargé en lecture seule depuis le SSD : **25/25 couches
  transférées sur MTL0 (Apple M1 Max)**, embedding réel de **1024 dimensions**,
  contexte **8192 tokens**. Le test a nécessité l'accès GPU hors sandbox.
- Régressions couvertes : dossiers techniques exclus, disparition d'un dossier,
  liens de fichiers valides, cycles de liens, attributs de jonction Windows,
  refus du Python système, manifeste périmé, profils et recréation de `.venv`,
  démarrage sans SDXL, refus d'une wheel CPU pour une demande GPU.
- Résolution croisée Windows x64 : PyTorch 2.13.0+cu130 et Torchvision 0.28.0+cu130,
  wheel llama-cpp-python CUDA 13.0 ; aucune wheel macOS requise pour ce parcours.

Restent à vérifier sur Windows natif : installation sans outils préinstallés, jonction NTFS cassée,
DLL/driver CUDA, calcul BGE CUDA et déplacement. Aucune génération SDXL lourde
n'a été effectuée dans cette tranche. Aucun commit ni push n'est automatique.


## Correctif du bootstrap Windows et isolation

Un premier test Windows sur `783337726a2dde7a71f459da5c16f61f3077c271`
a installé uv 0.12.10, puis s'est arrêté sur un faux échec. Le bootstrap lisait
`$LASTEXITCODE` immédiatement après `Invoke-Expression`, qui ne garantit pas de
mettre à jour ce code natif. Il vérifie désormais l'existence de uv puis exécute
`uv --version` et contrôle son véritable code de sortie et la version épinglée.
Voir la [sémantique PowerShell de LASTEXITCODE](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_automatic_variables?view=powershell-5.1#lastexitcode).

`tests/test_windows_bootstrap.py` exécute le bloc PowerShell réel avec un
installateur local simulé : état initial vide ou périmé, binaire absent, mauvaise
version, binaire en échec, erreur de téléchargement, réutilisation et remplacement.
Les huit cas passent avec PowerShell 7.5.2 portable sur macOS. Le test sélectionne
Windows PowerShell 5.1 quand il est disponible ; sinon il utilise `pwsh` ou
`PULID_TEST_POWERSHELL`. Sans PowerShell, ces cas sont explicitement ignorés.

L'installation conserve le téléchargement verrouillé, sans utiliser les paquets
Python globaux. Les lanceurs passent `-I`, uv reçoit `--no-config`, les variables
de recherche de bibliothèques héritées sont neutralisées, et les chemins CUDA
sont limités à la wheel locale et au pilote. Les tests des lanceurs injectent un
`PYTHONHOME` invalide et un `sitecustomize` global : ils restent ignorés. Le
frontend ne se rabat plus sur un Python système.

Validation locale de ce correctif : **250 tests réussis, 3 tests d'intégration
lourde ignorés**. Une installation de production dans `/tmp` a également réussi
avec `PYTHONHOME`, `PYTHONPATH`, `PYTHONUSERBASE` et une configuration uv globale
volontairement invalides. Les modèles et l'environnement utilisateur existant
n'ont pas été modifiés. Le parcours complet Windows natif reste à retester.


## Déplacement sans réinstallation

Les nouvelles installations créent `.venv` avec `uv venv --relocatable`. Le
fichier `.venv/pulid-python-path` conserve un chemin relatif lorsque Python est
sous le projet, absolu lorsqu’il est externe. Les lanceurs backend et frontend
appellent ce Python géré en mode isolé pour réajuster les chemins de `.venv`
avant sa première utilisation. Cette étape utilise exclusivement la bibliothèque
standard : ni uv, ni réseau, ni résolution de paquets. Les versions et le verrou
sont contrôlés avant toute modification. Les métadonnées sont remplacées
atomiquement et ne sont pas réécrites si elles sont déjà correctes.

Le profil développement utilise un chemin `.pth` relatif vers les sources. Le
profil production conserve le paquet installé. Les deux helpers natifs sont
obligatoires dans les archives. Une ancienne installation encore à son emplacement
initial peut enregistrer les métadonnées portables au premier démarrage avec les
nouveaux lanceurs, sans repasser par l’installation.

Validation macOS : installation de production complète en Python 3.11.16/uv
0.12.10, puis déplacement du dossier **avec son Python géré** vers un chemin
contenant espaces et accent. Le script d’installation était rendu indisponible
avant le déplacement. Après démarrage par le lanceur, `/health` et `/models`
répondent HTTP 200 sans SDXL, et `/v1/embeddings` répond HTTP 200 avec un embedding
BGE Metal de 1024 dimensions. Les téléchargements Hugging Face étaient désactivés
et le proxy de téléchargement inaccessible. Les **36 683 fichiers des paquets**
(hors caches bytecode) ont conservé taille et date de modification ; une sentinelle
dans `.venv` a été conservée. Aucun poids ni dossier utilisateur n’a été modifié.
L’inventaire du dossier de test, sans poids PuLID/AntelopeV2, signale correctement
ces deux absences avec un code non nul, sans erreur de parcours du Python déplacé.
L’inventaire des modèles existants sur le SSD a également réussi depuis ce Python
déplacé, en lecture seule et avec les sorties de contrôle dans `/tmp`.

Suite : **265 tests réussis, 3 tests d’intégration lourde ignorés**. Les tests
couvrent deux déplacements successifs, backend/frontend, production/développement,
Python interne/externe, métadonnées Windows, refus des versions/verrous incompatibles
et sélection du Python par le helper PowerShell réel (PowerShell 7.5.2 sur macOS).
Le démarrage du Python Windows et les DLL CUDA après déplacement restent à vérifier
sur Windows natif. Le déplacement entre
OS/architectures et celui indépendant d’un dossier de modèles externe ne sont pas
pris en charge automatiquement.

## Correctif du trampoline Windows après déplacement

Le test Windows utilisateur du commit `1696690e1019afe7b9170c065e4646509b80d68e`
valide l'installation complète sans SDXL, Python géré 3.11.16, uv 0.12.10,
PyTorch 2.13.0+cu130, le calcul BGE CUDA (1024 dimensions, contexte 8192), doctor,
l'inventaire et le premier démarrage réseau sur RTX 4070 SUPER. Après déplacement,
le démarrage échoue avec `uv trampoline failed to spawn Python child process`.
Ce nouvel échec est distinct du parcours d'inventaire : celui-ci réussit dans le log.
La cause exacte du WinError 3 initial chez l'autre utilisateur reste non confirmée.

Dans [uv 0.12.10](https://github.com/astral-sh/uv/blob/0.12.10/crates/uv-virtualenv/src/virtualenv.rs),
le lanceur Windows peut être un trampoline vers la jonction Python mineure.
Son chemin est intégré à l'exécutable ; réécrire `home` dans `pyvenv.cfg` ne suffit
pas. Les tests précédents de déplacement utilisaient `venv.EnvBuilder` et ne
couvraient pas ce trampoline uv.

La préparation du runtime installe désormais les deux redirecteurs CPython du
Python géré (`Lib/venv/scripts/nt/python.exe` et `pythonw.exe`) dans `.venv/Scripts`.
Leur [mode VENV_REDIRECT](https://github.com/python/cpython/blob/v3.11.16/PC/launcher.c)
lit `home` dans `pyvenv.cfg`. Les alias `python3.exe`/`python3.11.exe` existants
sont corrigés aussi. Les binaires identiques sont conservés sans réécriture.
L'absence d'un redirecteur embarqué provoque une erreur avant la réparation ;
aucun Python, DLL ou lanceur système n'est utilisé. La migration des anciennes
installations quitte d'abord le trampoline avant de le remplacer, car Windows
verrouille les exécutables en cours d'utilisation.

Contrôle local : l'archive Windows Python 3.11.16 du build standalone 20260901 a
été téléchargée dans `/tmp` et son SHA-256 vérifié contre les métadonnées uv
0.12.10. Elle contient bien les deux redirecteurs ; leur copie a été vérifiée
octet par octet. Les tests couvrent le remplacement, les alias, l'absence de
redirecteur, la conservation des exécutables déjà corrects et les chemins internes
ou externes. Aucun binaire Windows n'a été exécuté sur macOS.
Suite locale : **267 tests réussis, 4 ignorés** (trois intégrations lourdes et le
nouveau test Windows natif). La migration PowerShell a également été exécutée
avec PowerShell 7.5.2 portable pour vérifier que la réparation démarre depuis le
Python géré après avoir quitté le processus `.venv`.

Un test natif optionnel reproduit un trampoline uv cassé, déplace une copie privée
du Python géré, puis vérifie deux démarrages après déplacement, l'isolation et la
conservation de `.venv`. Sous Windows, depuis l'installation de développement :

```powershell
$env:PULID_TEST_UV = (Resolve-Path .\PuLID_models\other\uv-windows-bin\uv.exe).Path
.\.venv\Scripts\python.exe -I -m pytest -q tests/test_windows_runtime_native.py
```

Adapter seulement le chemin uv si les modèles sont externes. Ce test est sans
réseau ni modèle et requiert quelques centaines de Mo temporaires. La réparation
native et le redémarrage BGE CUDA après déplacement restent à valider sur Windows.
