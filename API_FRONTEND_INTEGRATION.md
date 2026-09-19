# API PuLID — intégration frontend

Ce serveur HTTP local expose huit routes applicatives. Trois routes de découverte
stables sont prévues pour les lanceurs, installateurs et clients gérés :

- `GET /health` confirme que le processus HTTP répond ;
- `GET /version` expose la version PuLID et celle du contrat API ;
- `GET /capabilities` décrit les fonctions et routes disponibles.

Les cinq routes métier sont :

- `GET /models` liste séparément les checkpoints SDXL, les méthodes de sampling
  et les courbes de sigmas compatibles ;
- `POST /generate` génère une image SDXL avec identité et renvoie le PNG ;
- `POST /generate/krea2` génère une image Krea v2 sans identité et renvoie le PNG ;
- `GET /v1/models` expose le modèle d'embedding de texte local au format OpenAI ;
- `POST /v1/embeddings` calcule un ou plusieurs embeddings au format OpenAI.

Les trois routes de découverte ne vérifient, ne chargent et ne téléchargent
aucun modèle. Elles restent donc légères même lorsque les checkpoints sont
absents ou temporairement indisponibles. La version PuLID provient des
métadonnées du paquet générées depuis `project.version` dans `pyproject.toml`.
Le contrat HTTP est versionné indépendamment en SemVer ; cette tranche expose
`1.1.0`.

La génération HTTP est séparée du chemin CLI avec sauvegarde. Elle ne crée ni
image dans `outputs/`, ni manifeste JSON. L'image de référence, le
conditionnement PuLID et le PNG de réponse restent en mémoire. Le petit embedding
ArcFace est créé ou réutilisé sous `cache/identity/`, avec une clé dérivée des
pixels décodés et du nom du personnage. Le frontend reste responsable du
téléchargement ou du stockage du PNG reçu.

Le modèle GGUF de texte est chargé paresseusement depuis `models_root`. Il
utilise Metal/CUDA par défaut, ou le CPU avec l'option de démarrage `--CPU`.
Les embeddings ne créent aucun fichier dans le projet.

## Démarrage du serveur

Installer les dépendances du serveur avec celles de l'inférence :

```bash
uv pip install -e '.[inference,pulid,server,embeddings]'
```

Démarrer sur l'interface locale, comportement par défaut :

```bash
pulid-server --port 12693 --device mps
```

Si le frontend est servi depuis une autre origin, l'autoriser explicitement :

```bash
pulid-server \
  --host 127.0.0.1 \
  --port 12693 \
  --device mps \
  --cors-origin http://localhost:3000
```

`--cors-origin` est répétable. Par défaut, aucune origin distante n'est
autorisée. Le serveur n'implémente pas d'authentification et doit rester lié à
`127.0.0.1`, sauf si une protection réseau adaptée est ajoutée.

L’écoute sur toutes les interfaces et CORS ouvert sont des options avancées
séparées et explicites :

```bash
pulid-server --network --port 12693
```

`--network` est le raccourci officiel pour `--host 0.0.0.0 --cors-origin "*"`.

Ne combinez ces options que sur un réseau privé de confiance. Sous Windows,
`start_windows.bat` reste sur `127.0.0.1`, tandis que
`start_windows.bat --network` active ce mode avancé. La règle de pare-feu privée
n’est proposée que par `install_windows.bat --network`.

La politique mémoire BGE/SDXL se choisit au démarrage :

```text
aucune option  BGE et SDXL concurrents sur CUDA, aucun offload
--serialized-cuda  BGE et SDXL sérialisés sur GPU, aucun offload
--partial      BGE GPU, CLIP et VAE SDXL déplacés sur CPU
--full         BGE GPU, pipeline SDXL entièrement déchargé
--CPU          BGE CPU, aucun offload SDXL
```

Ces options sont mutuellement exclusives. Sur CUDA, la concurrence est activée
par défaut. Si elle provoque un OOM lors d'un pic mémoire,
`--serialized-cuda` restaure le verrou commun sans offload.
`start_pulid_server.sh` et `start_windows.bat` transmettent leurs arguments au
serveur.

URL de base utilisée dans les exemples :

```text
http://127.0.0.1:12693
```

## Découverte, santé et compatibilité

### Santé du processus

```http
GET /health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "api_contract_version": "1.1.0"
}
```

Un statut `200` signifie que le processus HTTP répond. Il ne garantit pas la
présence des modèles ; utilisez ensuite les routes métier ou `pulid-gen doctor`
pour un diagnostic des actifs locaux.

### Versions

```http
GET /version
```

```json
{
  "component": "pulid",
  "version": "0.1.0",
  "api_contract_version": "1.1.0"
}
```

Un client doit comparer séparément la version PuLID exacte et le major du
contrat API.

### Capacités

```http
GET /capabilities
```

```json
{
  "component": "pulid",
  "version": "0.1.0",
  "api_contract_version": "1.1.0",
  "capabilities": {
    "krea2_generation": {
      "enabled": true,
      "generation_endpoint": "/generate/krea2",
      "identity_transfer": false,
      "samplers": ["euler"],
      "schedulers": ["beta"]
    },
    "image_generation": {
      "enabled": true,
      "catalog_endpoint": "/models",
      "generation_endpoint": "/generate"
    },
    "text_embeddings": {
      "enabled": true,
      "models_endpoint": "/v1/models",
      "embeddings_endpoint": "/v1/embeddings",
      "model": "text-embedding-bge-m3",
      "dimensions": 1024
    }
  }
}
```

`enabled` décrit la configuration annoncée, pas l’état de chargement d’un poids.
Les valeurs `model` et `dimensions` sont `null` si les embeddings ne sont pas
configurés.

## Embeddings de texte OpenAI compatibles

Le modèle est configuré avec un chemin relatif à `models_root`, portable entre
le SSD macOS et le dossier `PuLID_models` placé à la racine du projet Windows :

```yaml
text_embedding:
  checkpoint: text_embedding/bge-m3-Q8_0.gguf
  model_id: text-embedding-bge-m3
  dimensions: 1024
  context_size: 8192
  batch_size: 8192
  threads: 0  # automatique
```

En mode GPU, `n_gpu_layers=-1`, `offload_kqv=true`, `op_offload=true` et
`flash_attn=true`. Avec `--CPU`, ces offloads sont désactivés. Ces valeurs ne
peuvent pas être modifiées par une requête HTTP. BGE-M3 étant un encodeur
bidirectionnel, `batch_size` doit être supérieur
ou égal à `context_size` : llama.cpp ne peut pas découper une séquence
d'embedding en micro-lots indépendants. La configuration est refusée au
démarrage si cette contrainte n'est pas respectée, au lieu de laisser
llama.cpp interrompre nativement Python sur une entrée longue. La fenêtre reste
à 8192 tokens dans les cinq modes. Avec `--CPU` et `threads: 0`,
`llama-cpp-python` utilise tous les CPU logiques pour le traitement par lots ;
une valeur positive permet de limiter manuellement cette charge.

### Lister le modèle d'embedding

```http
GET /v1/models
```

```bash
curl http://127.0.0.1:12693/v1/models
```

Réponse `200 application/json` :

```json
{
  "object": "list",
  "data": [
    {
      "id": "text-embedding-bge-m3",
      "object": "model",
      "created": 0,
      "owned_by": "pulid-local"
    }
  ]
}
```

Cette route vérifie la présence du GGUF sans le charger. Un fichier absent est
signalé en `503` avec son chemin résolu.

### Calculer des embeddings

```http
POST /v1/embeddings
Content-Type: application/json
```

Corps pour un texte :

```json
{
  "model": "text-embedding-bge-m3",
  "input": "Un souvenir important du personnage."
}
```

Le champ `input` accepte également un tableau de chaînes. `encoding_format`
peut être omis ou valoir `float`; le format Base64 n'est pas pris en charge.

```bash
curl --fail-with-body http://127.0.0.1:12693/v1/embeddings \
  --header "Content-Type: application/json" \
  --data '{"model":"text-embedding-bge-m3","input":["bonjour","au revoir"]}'
```

Réponse `200 application/json` — vecteur volontairement abrégé dans l'exemple :

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789]
    }
  ],
  "model": "text-embedding-bge-m3",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

Les vecteurs BGE-M3 comportent 1024 composantes. Le backend vérifie leur
dimension et leurs valeurs puis applique une normalisation L2. Les limites HTTP sont de 64 textes,
32 000 caractères par texte et 128 000 caractères cumulés par requête. Un texte
vide ou un identifiant de modèle différent est refusé en `422`.
Un texte dépassant les 8192 jetons configurés est également refusé avant le
calcul ; il n'est jamais tronqué silencieusement.

## 1. Lister les modèles, méthodes et sigmas

### Requête

```http
GET /models
```

```bash
curl http://127.0.0.1:12693/models
```

### Réponse `200 application/json`

```json
{
  "models": [
    {
      "name": "realvisxlV50_v50LightningBakedvae",
      "filename": "realvisxlV50_v50LightningBakedvae.safetensors",
      "default": true
    },
    {
      "name": "reaxl_v30",
      "filename": "reaxl_v30.safetensors",
      "default": false
    }
  ],
  "sampling_methods": [
    {
      "name": "default",
      "label": "Scheduler du checkpoint",
      "default": true,
      "supported_sigma_schedules": ["normal"]
    },
    {
      "name": "dpmpp_2m",
      "label": "DPM++ 2M",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "dpmpp_2m_sde",
      "label": "DPM++ 2M SDE",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "dpmpp_3m_sde",
      "label": "DPM++ 3M SDE",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "euler",
      "label": "Euler",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "euler_ancestral",
      "label": "Euler ancestral",
      "default": false,
      "supported_sigma_schedules": ["normal"]
    },
    {
      "name": "heun",
      "label": "Heun",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "lms",
      "label": "LMS",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "ddim",
      "label": "DDIM",
      "default": false,
      "supported_sigma_schedules": ["normal"]
    }
  ],
  "sigma_schedules": [
    {
      "name": "normal",
      "label": "Normal / natif",
      "default": true,
      "supported_sampling_methods": [
        "default", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
        "euler", "euler_ancestral", "heun", "lms", "ddim"
      ]
    },
    {
      "name": "karras",
      "label": "Karras",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    },
    {
      "name": "exponential",
      "label": "Exponentiel",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    },
    {
      "name": "beta",
      "label": "Beta",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    }
  ]
}
```

Le frontend doit envoyer la propriété `name`, sans extension, dans le champ
`model` de `/generate`. Il doit envoyer séparément le `name` d'une méthode dans
`method` et le `name` d'une courbe dans `sigmas`.

La méthode `default` conserve le scheduler fourni par le checkpoint et accepte
uniquement `normal`. Pour alimenter la liste des sigmas, filtrer
`sigma_schedules` avec `samplingMethod.supported_sigma_schedules`. Le champ
inverse `sigmaSchedule.supported_sampling_methods` est fourni pour les interfaces
qui partent d'abord du choix de sigmas. L'API refuse une combinaison incompatible
en `422` avant de charger le pipeline.

Migration de l'ancienne valeur combinée :

```text
method=dpmpp_2m_sde_karras
            devient
method=dpmpp_2m_sde + sigmas=karras
```

`dpmpp_2m_sde_karras` n'est plus une valeur acceptée pour `method`.

L'inventaire des modèles est relu à chaque appel de `GET /models` depuis :

```text
/Volumes/SSD/Documents/PuLID_models/checkpoints/*.safetensors
```

Un checkpoint ajouté dans ce dossier apparaît donc sans redémarrage du serveur.

## 2. Générer une image

### Requête

```http
POST /generate
Content-Type: multipart/form-data
```

Champs du formulaire :

| Champ | Type | Requis | Défaut | Description |
|---|---:|:---:|---:|---|
| `reference` | fichier | oui | — | JPEG, PNG, WebP, BMP ou TIFF, 20 Mio maximum |
| `character` | texte | oui | — | Nom du personnage, 1 à 100 caractères |
| `prompt` | texte | oui | — | Prompt positif, 1 à 4000 caractères et 255 jetons CLIP utiles maximum |
| `negative_prompt` | texte | non | prompt négatif du pipeline | Prompt négatif, 4000 caractères et 255 jetons CLIP utiles maximum ; une chaîne vide le désactive |
| `clip_skip_2` | booléen | non | `false` | Active « Clip skip 2 » selon la convention A1111 pour les deux encodeurs CLIP de SDXL |
| `model` | texte | oui | — | `name` renvoyé par `GET /models` |
| `cfg` | nombre | non | `7.0` | CFG entre 0 et 30 |
| `steps` | entier | non | `20` | Nombre de steps entre 1 et 200 |
| `strength` | nombre | non | `0.8` | Force d'identité PuLID, finie et supérieure ou égale à 0 |
| `method` | texte | non | `default` | `sampling_methods[].name` renvoyé par `GET /models` |
| `sigmas` | texte | non | `normal` | `sigma_schedules[].name` compatible avec `method` |
| `seed` | entier | non | `0` | `0` ou `-1` = aléatoire ; sinon 1 à 2^63−1 |
| `width` | entier | non | `1024` | Largeur de génération, de 64 à 2048 pixels et multiple de 8 |
| `height` | entier | non | `1024` | Hauteur de génération, de 64 à 2048 pixels et multiple de 8 |

`width` et `height` sont facultatifs afin de préserver les clients existants :
leur omission conserve la résolution historique `1024 × 1024`. Les résolutions
SDXL recommandées proposées par le frontend sont `1024 × 1024`, `896 × 1152`,
`832 × 1216`, `768 × 1344`, `640 × 1536` et leurs équivalents paysage.

Si `negative_prompt` est omis, le prompt négatif par défaut du pipeline est
appliqué. Envoyer une chaîne vide permet de le désactiver explicitement.
`clip_skip_2` accepte les valeurs booléennes de formulaire reconnues par
FastAPI, notamment `true` et `false`. La valeur `true` est traduite en
`clip_skip=1` pour Diffusers : elle sélectionne `hidden_states[-3]`. La valeur
`false` conserve le comportement SDXL natif, `hidden_states[-2]`.

Le backend compte séparément les jetons produits par les deux tokenizers de
SDXL, sans inclure les marqueurs BOS/EOS. Jusqu'à 75 jetons utiles, l'encodage
natif Diffusers est conservé. De 76 à 255 jetons, le prompt est segmenté en
blocs CLIP puis leurs embeddings sont concaténés sans troncature. Si l'un des
deux tokenizers dépasse 255 jetons utiles, l'API répond en `422` avec
`PromptTooLongError`. Le nombre de jetons n'est pas équivalent au nombre de
mots ou de caractères ; le frontend peut donc conserver la limite de 4000
caractères et afficher le message précis renvoyé par l'API.

Exemple `curl` :

```bash
curl --fail-with-body \
  --output generation.png \
  --dump-header response-headers.txt \
  --form reference=@inputs/noemie.webp \
  --form character=noemie \
  --form 'prompt=cinematic portrait of a woman standing in Tokyo at night' \
  --form 'negative_prompt=bad anatomy, watermark' \
  --form clip_skip_2=true \
  --form model=reaxl_v30 \
  --form cfg=4.5 \
  --form steps=20 \
  --form strength=1.25 \
  --form method=dpmpp_2m_sde \
  --form sigmas=karras \
  --form seed=0 \
  --form width=832 \
  --form height=1216 \
  http://127.0.0.1:12693/generate
```

### Réponse réussie

Le corps de la réponse est directement le fichier PNG :

```http
HTTP/1.1 200 OK
Content-Type: image/png
Content-Disposition: attachment; filename="noemie_20260813T200947_123456Z.png"
Cache-Control: no-store
X-Generation-Seed: 987654321
X-SDXL-Model: reaxl_v30
X-Sampling-Method: dpmpp_2m_sde
X-Sigma-Schedule: karras
```

Le nom est composé du personnage normalisé et de la date/heure UTC :

```text
<personnage>_YYYYMMDDTHHMMSS_microsecondesZ.png
```

Pour une seed `0` ou `-1`, `X-Generation-Seed` contient la seed aléatoire
réellement utilisée. Le frontend doit la conserver s'il veut permettre de
reproduire la génération.

### Exemple JavaScript/TypeScript

```ts
type GenerateInput = {
  reference: File;
  character: string;
  prompt: string;
  negativePrompt?: string;
  clipSkip2?: boolean;
  model: string;
  cfg: number;
  steps: number;
  strength: number;
  method: string;
  sigmas: string;
  seed: number;
  width?: number;
  height?: number;
};

export async function generateImage(input: GenerateInput) {
  const form = new FormData();
  form.append("reference", input.reference);
  form.append("character", input.character);
  form.append("prompt", input.prompt);
  if (input.negativePrompt !== undefined) {
    form.append("negative_prompt", input.negativePrompt);
  }
  form.append("clip_skip_2", String(input.clipSkip2 ?? false));
  form.append("model", input.model);
  form.append("cfg", String(input.cfg));
  form.append("steps", String(input.steps));
  form.append("strength", String(input.strength));
  form.append("method", input.method);
  form.append("sigmas", input.sigmas);
  form.append("seed", String(input.seed));
  if (input.width !== undefined) form.append("width", String(input.width));
  if (input.height !== undefined) form.append("height", String(input.height));

  const response = await fetch("http://127.0.0.1:12693/generate", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(
      error.detail?.message ?? `Génération impossible (${response.status})`,
    );
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename =
    disposition.match(/filename="([^"]+)"/)?.[1] ?? "generation.png";

  return {
    blob,
    objectUrl: URL.createObjectURL(blob),
    filename,
    seed: Number(response.headers.get("X-Generation-Seed")),
    model: response.headers.get("X-SDXL-Model"),
    method: response.headers.get("X-Sampling-Method"),
    sigmas: response.headers.get("X-Sigma-Schedule"),
  };
}
```

Pour le contrôle frontend, utiliser `0.8` comme valeur initiale et un champ
numérique avec `min=0`. Un pas de `0.05` ou `0.1` est adapté à l'interface, mais
l'API accepte toute valeur finie supérieure ou égale à zéro. `String(number)`
produit le séparateur décimal `.` attendu dans le `FormData`, indépendamment de
la langue de l'interface.

Types conseillés pour la réponse du GET :

```ts
type NamedOption = {
  name: string;
  label: string;
  default: boolean;
};

type SamplingMethod = NamedOption & {
  supported_sigma_schedules: string[];
};

type SigmaSchedule = NamedOption & {
  supported_sampling_methods: string[];
};

type ModelsResponse = {
  models: Array<{ name: string; filename: string; default: boolean }>;
  sampling_methods: SamplingMethod[];
  sigma_schedules: SigmaSchedule[];
};
```

Lorsqu'un utilisateur change de méthode, conserver les sigmas seulement s'ils
figurent encore dans `supported_sigma_schedules`; sinon revenir à la première
valeur compatible, normalement `normal` :

```ts
const method = inventory.sampling_methods.find((item) => item.name === methodName);
const allowed = new Set(method?.supported_sigma_schedules ?? ["normal"]);
const visibleSigmas = inventory.sigma_schedules.filter((item) => allowed.has(item.name));
const nextSigmas = allowed.has(currentSigmas) ? currentSigmas : visibleSigmas[0].name;
```

Lorsque l'aperçu n'est plus utilisé, libérer son URL :

```ts
URL.revokeObjectURL(result.objectUrl);
```

Pour télécharger le résultat côté navigateur :

```ts
const link = document.createElement("a");
link.href = result.objectUrl;
link.download = result.filename;
link.click();
```

## Erreurs

Les erreurs métier utilisent cette forme :

```json
{
  "detail": {
    "error": "ModelNotFoundError",
    "message": "Checkpoint SDXL introuvable: ..."
  }
}
```

Statuts usuels :

- `422` : formulaire invalide (dont `strength` négatif ou non fini), prompt
  dépassant 255 jetons CLIP utiles,
  modèle/méthode/sigmas inconnus, combinaison
  méthode-sigmas incompatible, image illisible, aucun visage ou plusieurs visages ;
- `422` sur `/v1/embeddings` : modèle inconnu, entrée vide, lot ou texte trop grand ;
- `503` sur les routes `/v1` : GGUF local absent ou runtime
  `llama-cpp-python` indisponible ;
- `500` : échec de chargement d'un modèle ou erreur pendant l'inférence.

Les erreurs de validation FastAPI utilisent un tableau standard dans `detail`.

## Cycle de vie et concurrence

- une seule génération est exécutée à la fois ; une requête concurrente attend
  la fin de la précédente afin de protéger la mémoire MPS/CUDA ;
- un seul lot d'embeddings est calculé à la fois ;
- sur CUDA, les embeddings GPU et SDXL calculent simultanément par défaut ;
  `--serialized-cuda` rétablit un verrou commun, tandis que `--CPU` autorise un
  embedding CPU pendant SDXL ;
- le checkpoint choisi est chargé localement avec les téléchargements désactivés ;
- l'embedding ArcFace est créé ou réutilisé dans le cache NPZ configuré ;
- le PNG est encodé dans un buffer mémoire puis renvoyé immédiatement ;
- sur CUDA, le générateur et son pipeline SDXL restent chargés après la réponse
  tant que les requêtes suivantes utilisent le même checkpoint ; sélectionner un
  autre modèle ferme d'abord le générateur précédent afin de libérer sa VRAM ;
- sur MPS et CPU, les composants du pipeline sont fermés après chaque réponse ;
- le générateur CUDA encore actif est fermé à l'arrêt du serveur ;
- sans option, avec `--serialized-cuda` ou `--CPU`, le GGUF reste chargé jusqu'à
  l'arrêt ; avec `--partial` ou `--full`, il est également fermé avant une
  génération SDXL ;
- aucun PNG ni manifeste JSON n'est créé par l'application serveur.

Une génération peut prendre plusieurs dizaines de secondes. Le proxy ou le
client HTTP du frontend doit utiliser un timeout adapté, par exemple deux à cinq
minutes selon le matériel et le nombre de steps.

### Runtime BGE et disponibilité sans SDXL

Le démarrage et les routes de supervision restent disponibles sans checkpoint
SDXL. L’inventaire récursif du CLI n’intervient pas dans le démarrage du backend.
Sur macOS, le lanceur demande BGE sur GPU Metal ; si la wheel installée ne fournit
pas le backend GPU demandé, le chargement produit une `ModelLoadError` via la
gestion d’erreur existante des embeddings, sans repli CPU silencieux. Réinstaller
PuLID depuis une archive complète restaure la wheel Metal précompilée. Les routes,
corps, en-têtes et formats d’erreur existants restent identiques.

Les lanceurs fournis utilisent maintenant Python en mode isolé. Sous Windows,
les DLL CUDA applicatives proviennent du `.venv` PuLID ; le pilote NVIDIA reste
requis, et les variables d'un CUDA Toolkit global ne sont plus utilisées. En cas
de DLL manquante, réparer l'installation PuLID ou son pilote ; les formats des
réponses HTTP restent inchangés.


## Génération Krea v2 sans identité

```http
POST /generate/krea2
Content-Type: multipart/form-data
```

Les formulaires `application/x-www-form-urlencoded` sont également acceptés.
Le frontend léger utilise le proxy `POST /api/generate/krea2` ;
`GET /api/capabilities` relaie la découverte du backend.

| Champ | Défaut | Validation |
|---|---|---|
| `prompt` | obligatoire | 1 à 4000 caractères, non vide après trim |
| `width` | 1248 | entier, 64 à 2048, multiple de 16 |
| `height` | 832 | entier, 64 à 2048, multiple de 16 |
| `seed` | 0 | -1 ou 0 = aléatoire ; sinon 1 à 2^63−1 |
| `steps` | 10 | entier, 1 à 200 |
| `cfg` | 1 | nombre fini, 0 à 30 ; convention CFG ComfyUI |
| `sampler` | `euler` | seule valeur implémentée : `euler` |
| `scheduler` | `beta` | seule valeur implémentée : `beta` |
| `denoise` | 1 | nombre fini, 0,01 à 1 |

Les champs supplémentaires sont refusés, notamment `reference`, `character`,
`strength`, `negative_prompt` et `model`. Le checkpoint se choisit côté serveur
via `krea2.checkpoint`. Le conditionnement négatif est vide, comme dans le workflow.
Le tokenizer utilise une fenêtre de 512 tokens et tronque les prompts plus longs.
Aucun encodage facial, transfert d'identité ou cache d'identité n'est exécuté.

Le scheduler beta reprend la grille de 10000 temps du workflow, avec
alpha=beta=0,6 et un shift exponentiel fixe mu=1,15. Pour `denoise < 1`,
les derniers intervalles du planning allongé sont utilisés et le bruit initial
est réduit à la première sigma retenue. Ce paramètre n'ajoute pas de mode img2img.
Le CFG est traduit en `guidance_scale = cfg - 1` pour Diffusers ; `cfg=1`
ne calcule donc pas la branche négative. Les sorties ne sont pas garanties
identiques pixel par pixel à ComfyUI.

Succès : `200 image/png`, avec les en-têtes :

- `Content-Disposition: attachment; filename="krea2_<horodatage>.png"` ;
- `Cache-Control: no-store` ;
- `X-Generation-Model: krea2` ;
- `X-Generation-Seed` : seed effective, à conserver comme chaîne en JavaScript ;
- `X-Sampling-Method: euler` et `X-Sigma-Schedule: beta`.

Ces en-têtes sont exposés par CORS et relayés par le proxy du frontend.
**L'endpoint ne crée aucun PNG, JSON ou autre sortie locale.** Le PNG reste en
mémoire ; rp-bot est responsable de sa conservation. Le frontend léger conserve
seulement l'aperçu en mémoire, avec un téléchargement manuel facultatif.

Erreurs : `422` pour paramètres invalides, fichiers absents ou device indisponible ;
`500` pour chargement incompatible/corrompu, panne d'inférence ou mémoire insuffisante.
Les erreurs métier suivent `{"detail":{"error":"...","message":"..."}}` ;
les erreurs de validation FastAPI utilisent une liste dans `detail`.
Les messages de modèles absents/incompatibles indiquent les chemins à corriger.
Aucun téléchargement n'est effectué pendant une requête HTTP.

Le chargement accepte aussi les poids NVFP4 Comfy et FP8 E4M3 scaled déclarés
dans leurs métadonnées. Ils restent compactés en mémoire : déquantification
temporaire par couche, ou calcul FP8 natif sur CUDA compatible. Le serveur
ne crée aucun fichier de poids converti. Cette prise en charge ne change pas
les paramètres ni la réponse HTTP. Les formats non pris en charge ou les
échelles manquantes suivent l'erreur de chargement `500` décrite ci-dessus.

Sur Windows/CUDA, le décodage requiert les noyaux Comfy Kitchen ajoutés à une
installation existante par `install_windows.bat --update`, sans recréer `.venv`.
Une dépendance absente/inutilisable produit également
une erreur de chargement `500` avec la commande corrective. Le terminal trace
les phases et les durées ; la réponse HTTP reste un PNG unique à la fin.
L'attention du modèle Krea sur CUDA impose un noyau fusionné et journalise
son nom (Flash, memory-efficient ou cuDNN). Aucun repli silencieux vers
le backend `math` n'est permis. Si aucun noyau n'est compatible, l'API renvoie
`500` avec `detail.error = "GenerationError"` et un message indiquant les
dimensions et la commande `scripts/benchmark_krea2_cuda.py --attention`.
Pour une carte de 12 Go, lancer `start_windows.bat --offload model_cpu_offload`.
Le serveur, y compris l'exécutable direct `pulid-server`, écoute désormais sur
`127.0.0.1:12693` par défaut, comme le proxy du frontend. Un port différent exige
de régler aussi l'URL du backend du frontend.

Krea est sérialisé avec toutes les générations SDXL et tous les embeddings BGE,
y compris en mode CUDA concurrent. Après les requêtes en cours, SDXL et BGE sont
déchargés avant Krea. Krea est libéré à la fin de chaque requête, succès ou erreur.
Les routes SDXL/BGE rechargent ensuite leurs composants à la demande.

```bash
curl --fail-with-body http://127.0.0.1:12693/generate/krea2 \
  -F 'prompt=Un phare au bord de la mer, lumière du matin' \
  -F width=1248 -F height=832 -F seed=42 \
  -F steps=10 -F cfg=1 -F sampler=euler -F scheduler=beta -F denoise=1 \
  --output krea2.png
```

Ici `--output` enregistre le PNG sur la machine du client curl, pas sur le serveur.
Voir le [guide d'installation Krea v2](README.md#krea-v2-sans-identité) pour les
formats de poids compatibles et les chemins configurables.
