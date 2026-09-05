"""Commande installable ``pulid-gen`` et diagnostics locaux."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from pulid_app import __version__
from pulid_app.config import AppConfig, ConfigError, PROJECT_ROOT, load_config
from pulid_app.doctor import build_doctor_report, print_doctor_report
from pulid_app.exceptions import (
    ExternalDriveNotMountedError,
    ModelNotFoundError,
    PuLIDAppError,
    actionable_error,
)
from pulid_app.models.sdxl import (
    SAMPLING_METHOD_SPECS,
    SIGMA_SCHEDULE_ARGUMENTS,
)
from pulid_app.paths import (
    cache_env_violations,
    configure_external_model_caches,
    ensure_writable_directory,
    external_cache_paths,
    inspect_models,
    require_models_root,
    resolve_sdxl_checkpoint,
)


DEFAULT_NEGATIVE_PROMPT = (
    "flaws in the eyes, flaws in the face, low quality, worst quality, "
    "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
)
SAMPLING_METHODS = tuple(SAMPLING_METHOD_SPECS)
SIGMA_SCHEDULES = tuple(SIGMA_SCHEDULE_ARGUMENTS)
OFFLOAD_STRATEGIES = ("none", "model_cpu_offload")


def _print_actionable_error(console: Console, exc: BaseException) -> None:
    label, cause = actionable_error(exc)
    console.print(f"[bold red]{label}:[/]\n{cause}")


def build_inspection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspecte les checkpoints locaux requis par PuLID."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Fichier YAML à utiliser à la place de config/default.yaml.",
    )
    parser.add_argument(
        "--show-cache-env",
        action="store_true",
        help="Affiche les emplacements effectifs des caches de modèles.",
    )
    parser.add_argument(
        "--fail-on-internal-cache",
        action="store_true",
        help="Échoue si un cache effectif se trouve hors de models_root.",
    )
    parser.add_argument(
        "--allow-missing-sdxl",
        action="store_true",
        help="Tolère un checkpoint SDXL différé pendant la validation d'installation.",
    )
    return parser


def _format_bytes(value: int) -> str:
    units = ("o", "Kio", "Mio", "Gio", "Tio")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} Tio"


def _print_cache_env(config: AppConfig, console: Console) -> None:
    table = Table(title="Caches de modèles effectifs")
    table.add_column("Variable")
    table.add_column("Chemin", overflow="fold")
    for name in external_cache_paths(config.models_root):
        table.add_row(name, os.environ.get(name, "absent"))
    console.print(table)


def run_inspection(
    config_path: Path | None,
    console: Console,
    *,
    show_cache_env: bool = False,
    fail_on_internal_cache: bool = False,
    allow_missing_sdxl: bool = False,
) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration invalide :[/] {exc}")
        return 2

    configure_external_model_caches(config.models_root)
    console.print(f"Configuration : [cyan]{config.source_path}[/]")
    console.print(f"Racine modèles : [cyan]{config.models_root}[/]")

    failures: list[str] = []
    if show_cache_env or fail_on_internal_cache:
        _print_cache_env(config, console)
    cache_failures = cache_env_violations(config.models_root)
    if fail_on_internal_cache and cache_failures:
        console.print("[red]✗ Cache(s) hors du SSD configuré :[/]")
        for failure in cache_failures:
            console.print(f"  • {failure}")
        failures.append("cache_env")
    elif fail_on_internal_cache:
        console.print("[green]✓ Tous les caches sont sous models_root.[/]")
    if not config.models_root.is_dir():
        console.print("[red]✗ La racine des modèles n'existe pas ou n'est pas un dossier.[/]")
        failures.append("models_root")
    else:
        usage = shutil.disk_usage(config.models_root)
        console.print(
            f"[green]✓ Racine disponible[/] — espace libre : "
            f"[bold]{_format_bytes(usage.free)}[/] / {_format_bytes(usage.total)}"
        )

    inventory = inspect_models(config)
    for warning in inventory.warnings:
        console.print(f"Inventaire partiel — {warning}", style="yellow", markup=False)

    if inventory.pulid_checkpoints:
        console.print("[green]✓ Checkpoint(s) PuLID :[/]")
        for path in inventory.pulid_checkpoints:
            console.print(f"  • {path}")
    else:
        console.print(f"[red]✗ Checkpoint PuLID introuvable :[/] {config.pulid.checkpoint}")
        failures.append("pulid")

    if inventory.antelope_dir is None:
        console.print(
            f"[red]✗ AntelopeV2 introuvable :[/] {config.insightface.model_dir}"
        )
        failures.append("antelopev2")
    elif inventory.antelope_missing_files:
        console.print(f"[red]✗ AntelopeV2 incomplet :[/] {inventory.antelope_dir}")
        for name in inventory.antelope_missing_files:
            console.print(f"  • fichier manquant : {name}")
        failures.append("antelopev2")
    else:
        console.print(f"[green]✓ AntelopeV2 complet :[/] {inventory.antelope_dir}")

    table = Table(title="Candidats SDXL (.safetensors)")
    table.add_column("Chemin", overflow="fold")
    table.add_column("Configuré", justify="center")
    for path in inventory.sdxl_candidates:
        configured = "✓" if path == config.sdxl.checkpoint else ""
        table.add_row(str(path), configured)
    if inventory.sdxl_candidates:
        console.print(table)
    elif allow_missing_sdxl:
        console.print(
            "[yellow]⚠ Checkpoint SDXL non installé pour le moment.[/] "
            "Relancez `pulid-install` avant toute génération d'image."
        )
    else:
        console.print("[red]✗ Aucun candidat SDXL .safetensors détecté.[/]")
        failures.append("sdxl")

    if not config.sdxl.checkpoint.is_file():
        if not (allow_missing_sdxl and not inventory.sdxl_candidates):
            console.print(
                f"[red]✗ Checkpoint SDXL configuré absent :[/] "
                f"{config.sdxl.checkpoint}"
            )
            failures.append("sdxl_configured")
    else:
        console.print(
            "[green]✓ Checkpoint SDXL configuré (VAE intégré) :[/] "
            f"{config.sdxl.checkpoint}"
        )

    try:
        ensure_writable_directory(config.outputs_dir)
    except (OSError, PermissionError) as exc:
        console.print(f"[red]✗ Sorties non accessibles :[/] {exc}")
        failures.append("outputs")
    else:
        console.print(f"[green]✓ Sorties accessibles en écriture :[/] {config.outputs_dir}")

    if failures:
        console.print(f"[bold red]Inspection échouée ({len(failures)} contrôle(s)).[/]")
        return 1
    console.print("[bold green]Inspection réussie.[/]")
    return 0


def _project_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="Fichier YAML alternatif, résolu depuis la racine du projet.",
    )


def _add_identity_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Image JPEG, PNG, WebP, BMP ou TIFF.",
    )
    parser.add_argument(
        "--character",
        help="Identifiant du personnage ; nom du fichier sans extension par défaut.",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        help="Index du visage à utiliser lorsqu'il y en a plusieurs (base 0).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur de la commande installable de phase 11."""

    parser = argparse.ArgumentParser(
        prog="pulid-gen",
        description="Génération locale SDXL conditionnée par PuLID.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Vérifie le SSD, les modèles, les caches, le device et les dépendances.",
    )
    _add_config_option(doctor)
    doctor.add_argument(
        "--allow-missing-sdxl",
        action="store_true",
        help="Tolère un checkpoint SDXL différé pendant la validation d'installation.",
    )
    doctor.set_defaults(handler=_handle_doctor)

    inspection = subparsers.add_parser(
        "inspect-models",
        help="Inventorie les checkpoints locaux sans les charger.",
    )
    _add_config_option(inspection)
    inspection.add_argument("--show-cache-env", action="store_true")
    inspection.add_argument("--fail-on-internal-cache", action="store_true")
    inspection.add_argument("--allow-missing-sdxl", action="store_true")
    inspection.set_defaults(handler=_handle_inspection)

    encode = subparsers.add_parser(
        "encode",
        help="Crée ou réutilise le cache ArcFace d'une identité.",
    )
    _add_config_option(encode)
    _add_identity_options(encode)
    encode.add_argument(
        "--force",
        action="store_true",
        help="Ignore le cache existant et recalcule l'embedding.",
    )
    encode.set_defaults(handler=_handle_encode)

    generate = subparsers.add_parser(
        "generate",
        help="Encode une référence puis génère un PNG et son JSON.",
    )
    _add_config_option(generate)
    _add_identity_options(generate)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--strength", type=float, default=0.8)
    generate.add_argument("--steps", type=int, default=20)
    generate.add_argument(
        "--guidance-scale",
        "--cfg",
        dest="guidance_scale",
        type=float,
        default=7.0,
        help="CFG numérique (7.0 par défaut).",
    )
    generate.add_argument("--width", type=int, default=1024)
    generate.add_argument("--height", type=int, default=1024)
    generate.add_argument("--device", choices=("mps", "cuda", "cpu"))
    generate.add_argument("--dtype", choices=("float16", "float32"))
    generate.add_argument(
        "--offload",
        choices=OFFLOAD_STRATEGIES,
        help="Stratégie mémoire SDXL ; model_cpu_offload est réservé à CUDA.",
    )
    generate.add_argument(
        "--model",
        help=(
            "Nom d'un checkpoint SDXL du dossier checkpoints configuré, "
            "sans l'extension .safetensors."
        ),
    )
    generate.add_argument("--method", choices=SAMPLING_METHODS)
    generate.add_argument("--sigmas", choices=SIGMA_SCHEDULES, default="normal")
    generate.add_argument(
        "--force-identity",
        action="store_true",
        help="Recalcule le cache ArcFace de la référence.",
    )
    generate.set_defaults(handler=_handle_generate)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Mesure séparément les étapes du pipeline complet.",
    )
    _add_config_option(benchmark)
    _add_identity_options(benchmark)
    benchmark.add_argument("--prompt", required=True)
    benchmark.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--strength", type=float, default=0.8)
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument(
        "--guidance-scale",
        "--cfg",
        dest="guidance_scale",
        type=float,
        default=7.0,
    )
    benchmark.add_argument("--width", type=int, default=1024)
    benchmark.add_argument("--height", type=int, default=1024)
    benchmark.add_argument("--device", choices=("mps", "cuda", "cpu"))
    benchmark.add_argument("--dtype", choices=("float16", "float32"))
    benchmark.add_argument("--offload", choices=OFFLOAD_STRATEGIES)
    benchmark.add_argument(
        "--model",
        help=(
            "Nom d'un checkpoint SDXL du dossier checkpoints configuré, "
            "sans l'extension .safetensors."
        ),
    )
    benchmark.add_argument("--method", choices=SAMPLING_METHODS)
    benchmark.add_argument("--sigmas", choices=SIGMA_SCHEDULES, default="normal")
    benchmark.set_defaults(handler=_handle_benchmark)
    return parser


def _load_command_config(
    config_path: Path | None,
    console: Console,
) -> AppConfig | None:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration invalide :[/] {exc}")
        return None


def run_doctor(
    config_path: Path | None,
    console: Console,
    *,
    allow_missing_sdxl: bool = False,
) -> int:
    config = _load_command_config(config_path, console)
    if config is None:
        return 2
    report = build_doctor_report(config, allow_missing_sdxl=allow_missing_sdxl)
    print_doctor_report(report, console)
    return 0 if report.healthy else 1


def run_encode(
    args: argparse.Namespace,
    console: Console,
    *,
    encoder_factory: Callable[[AppConfig], Any] | None = None,
) -> int:
    config = _load_command_config(args.config, console)
    if config is None:
        return 2
    try:
        require_models_root(config.models_root)
    except ExternalDriveNotMountedError as exc:
        _print_actionable_error(console, exc)
        return 2
    configure_external_model_caches(config.models_root)
    reference = _project_path(args.reference)
    character = (args.character or reference.stem).strip()

    from pulid_app.models.identity_encoder import IdentityEncoder, IdentityEncoderError

    factory = encoder_factory or IdentityEncoder.from_config
    try:
        encoder = factory(config)
        cache_path = encoder.cache_path_for(
            reference,
            identity_id=character,
            face_index=args.face_index,
        )
        cache_hit = cache_path.is_file() and not args.force
        identity = encoder.encode_image(
            reference,
            identity_id=character,
            face_index=args.face_index,
            force_recompute=args.force,
        )
    except (PuLIDAppError, OSError, RuntimeError, ValueError) as exc:
        _print_actionable_error(console, exc)
        return 1

    state = "réutilisé" if cache_hit else "créé"
    console.print(f"Personnage : [bold]{identity.id}[/]")
    console.print(f"Référence : {identity.source_images[0]}")
    console.print(f"Cache {state} : [cyan]{cache_path}[/]")
    console.print(f"Embedding : shape={identity.face_embedding.shape}")
    return 0


def run_generate(
    args: argparse.Namespace,
    console: Console,
    *,
    generator_factory: Callable[..., Any] | None = None,
) -> int:
    config = _load_command_config(args.config, console)
    if config is None:
        return 2
    try:
        require_models_root(config.models_root)
    except ExternalDriveNotMountedError as exc:
        _print_actionable_error(console, exc)
        return 2
    try:
        checkpoint = resolve_sdxl_checkpoint(config, args.model)
    except ValueError as exc:
        console.print(f"[bold red]Modèle SDXL invalide :[/] {exc}")
        return 2
    if not checkpoint.is_file():
        _print_actionable_error(
            console,
            ModelNotFoundError(
                f"Checkpoint SDXL introuvable : {checkpoint}. "
                "Corrigez sdxl.checkpoint, utilisez --model avec un nom local "
                "existant, ou omettez --model pour revenir au modèle configuré."
            ),
        )
        return 2
    config = replace(config, sdxl=replace(config.sdxl, checkpoint=checkpoint))
    configure_external_model_caches(config.models_root)
    reference = _project_path(args.reference)
    character = (args.character or reference.stem).strip()

    from pulid_app.pipeline.generator import ImageGenerator, ImageGeneratorError

    factory = generator_factory or ImageGenerator
    generator: Any | None = None
    cleanup_error: ImageGeneratorError | None = None
    try:
        generator = factory(
            config,
            device=args.device,
            dtype_name=args.dtype,
            offload_strategy=args.offload,
            allow_downloads=False,
        )
        console.print(f"Device : [bold]{generator.device}[/]")
        console.print(f"Modèle SDXL : [cyan]{checkpoint.name}[/]")
        identity = generator.encode_identity(
            reference,
            identity_id=character,
            face_index=args.face_index,
            force_recompute=args.force_identity,
        )
        generated = generator.generate(
            prompt=args.prompt,
            identity=identity,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            identity_strength=args.strength,
            guidance_scale=args.guidance_scale,
            sampling_method=args.method,
            sigma_schedule=args.sigmas,
        )
    except PuLIDAppError as exc:
        _print_actionable_error(console, exc)
        return 1
    finally:
        if generator is not None:
            try:
                generator.close()
            except ImageGeneratorError as exc:
                cleanup_error = exc

    if cleanup_error is not None:
        console.print(f"[bold red]Nettoyage mémoire impossible :[/] {cleanup_error}")
        return 1
    console.print(f"[bold green]Image générée :[/] {generated.png_path}")
    console.print(f"[bold green]Métadonnées :[/] {generated.json_path}")
    return 0


def run_benchmark(
    args: argparse.Namespace,
    console: Console,
    *,
    runner_factory: Callable[..., Any] | None = None,
) -> int:
    config = _load_command_config(args.config, console)
    if config is None:
        return 2
    try:
        require_models_root(config.models_root)
    except ExternalDriveNotMountedError as exc:
        _print_actionable_error(console, exc)
        return 2
    try:
        checkpoint = resolve_sdxl_checkpoint(config, args.model)
    except ValueError as exc:
        console.print(f"[bold red]Modèle SDXL invalide :[/] {exc}")
        return 2
    if not checkpoint.is_file():
        _print_actionable_error(
            console,
            ModelNotFoundError(
                f"Checkpoint SDXL introuvable : {checkpoint}. "
                "Sélectionnez un nom local existant avec --model."
            ),
        )
        return 2
    config = replace(config, sdxl=replace(config.sdxl, checkpoint=checkpoint))
    configure_external_model_caches(config.models_root)
    reference = _project_path(args.reference)

    from pulid_app.pipeline.benchmark import BenchmarkError, BenchmarkRunner

    factory = runner_factory or BenchmarkRunner
    try:
        runner = factory(
            config,
            device=args.device,
            dtype_name=args.dtype,
            offload_strategy=args.offload,
        )
        result = runner.run(
            reference=reference,
            prompt=args.prompt,
            identity_id=args.character,
            face_index=args.face_index,
            runs=args.runs,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            identity_strength=args.strength,
            guidance_scale=args.guidance_scale,
            sampling_method=args.method,
            sigma_schedule=args.sigmas,
        )
    except BenchmarkError as exc:
        _print_actionable_error(console, exc)
        return 1

    console.print(f"[bold green]Benchmark enregistré :[/] {result.json_path}")
    summary = result.report["summary_seconds"]
    table = Table(title="Durées moyennes")
    table.add_column("Étape")
    table.add_column("Secondes", justify="right")
    for name, statistics_ in summary.items():
        table.add_row(name, f"{statistics_['mean']:.3f}")
    console.print(table)
    return 0


def _handle_doctor(args: argparse.Namespace, console: Console) -> int:
    return run_doctor(
        args.config,
        console,
        allow_missing_sdxl=args.allow_missing_sdxl,
    )


def _handle_inspection(args: argparse.Namespace, console: Console) -> int:
    return run_inspection(
        args.config,
        console,
        show_cache_env=args.show_cache_env,
        fail_on_internal_cache=args.fail_on_internal_cache,
        allow_missing_sdxl=args.allow_missing_sdxl,
    )


def _handle_encode(args: argparse.Namespace, console: Console) -> int:
    return run_encode(args, console)


def _handle_generate(args: argparse.Namespace, console: Console) -> int:
    return run_generate(args, console)


def _handle_benchmark(args: argparse.Namespace, console: Console) -> int:
    return run_benchmark(args, console)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace, Console], int] = args.handler
    return handler(args, Console())


def inspect_main(argv: list[str] | None = None) -> int:
    """Point d'entrée historique de ``pulid-inspect-models`` et du script."""

    args = build_inspection_parser().parse_args(argv)
    return run_inspection(
        args.config,
        Console(),
        show_cache_env=args.show_cache_env,
        fail_on_internal_cache=args.fail_on_internal_cache,
        allow_missing_sdxl=args.allow_missing_sdxl,
    )


if __name__ == "__main__":
    raise SystemExit(main())
