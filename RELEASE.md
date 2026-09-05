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

La recette [WINDOWS_INSTALL_TEST.md](WINDOWS_INSTALL_TEST.md) reste à exécuter
sur Windows natif : installation sans outils préinstallés, jonction NTFS cassée,
DLL/driver CUDA, calcul BGE CUDA et déplacement. Aucune génération SDXL lourde
n'a été effectuée dans cette tranche. Aucun commit ni push n'est automatique.
