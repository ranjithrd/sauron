#!/usr/bin/env python3
"""
Sauron Edge — Greengrass deployment script.

Usage (from the edge/ directory):
    uv run --extra deploy python scripts/deploy.py --version 1.0.0

What this script does (in order, with a confirmation gate before each step):
  1. Validate environment / configuration
  2. Build a clean zip artifact from src/
  3. Upload the zip to S3:  s3://<bucket>/artifacts/sauron-edge/<version>/sauron-edge-<version>.zip
  4. Upload the component recipe to S3 (optional, for traceability)
  5. Create a new Greengrass component version via greengrassv2:CreateComponentVersion
  6. Create (or update) a Greengrass deployment via greengrassv2:CreateDeployment

SAFETY RULES:
  - This script NEVER auto-runs.  Every step asks "Proceed? [y/N]" before calling AWS.
  - Pass --dry-run to preview everything without touching AWS or S3.
  - AWS credentials are read from the standard boto3 chain (profile, env vars, instance role).
    No credentials are ever embedded in this file or printed to stdout.
  - The script exits non-zero on any failure — suitable for CI if you add --yes flag.

Required environment variables (read from scripts/.env.deploy or environment):
    AWS_PROFILE              boto3 named profile (optional if using env vars / instance role)
    AWS_REGION               e.g. us-east-1
    S3_ARTIFACT_BUCKET       bucket name only, no s3:// prefix
    GG_COMPONENT_NAME        e.g. com.sauron.edge
    GG_TARGET_ARN            thing group or core device ARN for deployment
    GG_DEPLOYMENT_NAME       human-readable name for the deployment (e.g. sauron-edge-prod)

Optional environment variables:
    GG_NUCLEUS_VERSION_REQ   Greengrass nucleus semver requirement (default: >=2.9.0)
    DASHBOARD_PORT           dashboard port baked into recipe (default: 5000)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Bootstrap: load .env.deploy if present
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_EDGE_DIR = _SCRIPT_DIR.parent
_ENV_FILE = _SCRIPT_DIR / ".env.deploy"

if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
        print(f"[env] Loaded {_ENV_FILE}")
    except ImportError:
        # parse manually — dotenv may not be installed yet
        with open(_ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        print(f"[env] Loaded {_ENV_FILE} (manual parse)")
else:
    print(f"[env] {_ENV_FILE} not found — reading from environment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _h(msg: str) -> str:
    return f"{BOLD}{msg}{RESET}"


def _ok(msg: str) -> str:
    return f"{GREEN}✔  {msg}{RESET}"


def _warn(msg: str) -> str:
    return f"{YELLOW}⚠  {msg}{RESET}"


def _err(msg: str) -> str:
    return f"{RED}✘  {msg}{RESET}"


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{BOLD}[{n}/{total}] {msg}{RESET}")


def _confirm(prompt: str, auto_yes: bool, dry_run: bool) -> bool:
    """
    Ask the user to confirm before proceeding.
    Returns True if the step should run, False to skip.
    dry_run always returns False (skip the actual call).
    """
    if dry_run:
        print(f"  {YELLOW}[DRY RUN] Would: {prompt}{RESET}")
        return False
    if auto_yes:
        print(f"  {YELLOW}[--yes]   Auto-confirming: {prompt}{RESET}")
        return True
    answer = input(f"  {BOLD}Proceed? [y/N]{RESET} ").strip().lower()
    return answer in ("y", "yes")


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(_err(f"Required environment variable {name!r} is not set."))
        print("       Set it in scripts/.env.deploy or in your shell.")
        sys.exit(1)
    return val


def _semver_re() -> re.Pattern:
    return re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------------------
# Step 1 — Validate environment
# ---------------------------------------------------------------------------


def step_validate(version: str) -> Dict[str, str]:
    """Collect and validate all required config.  Returns a config dict."""
    print(_h("\nValidating configuration..."))

    if not _semver_re().match(version):
        print(_err(f"Version {version!r} must be in MAJOR.MINOR.PATCH format."))
        sys.exit(1)

    cfg: Dict[str, str] = {
        "version": version,
        "aws_region": _require_env("AWS_REGION"),
        "s3_bucket": _require_env("S3_ARTIFACT_BUCKET"),
        "component_name": _require_env("GG_COMPONENT_NAME"),
        "target_arn": _require_env("GG_TARGET_ARN"),
        "deployment_name": _require_env("GG_DEPLOYMENT_NAME"),
        "aws_profile": os.environ.get("AWS_PROFILE", "").strip(),
        "nucleus_req": os.environ.get("GG_NUCLEUS_VERSION_REQ", ">=2.9.0").strip(),
        "dashboard_port": os.environ.get("DASHBOARD_PORT", "5000").strip(),
    }

    # Validate S3 bucket name (basic)
    if not re.match(r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$", cfg["s3_bucket"]):
        print(
            _warn(
                f"S3 bucket name {cfg['s3_bucket']!r} looks unusual — continuing anyway."
            )
        )

    # Validate target ARN
    if not cfg["target_arn"].startswith("arn:aws"):
        print(_err(f"GG_TARGET_ARN must be a full ARN, got: {cfg['target_arn']!r}"))
        sys.exit(1)

    print(_ok("Configuration valid"))
    print(textwrap.dedent(f"""
          Component  : {cfg['component_name']} @ {cfg['version']}
          Region     : {cfg['aws_region']}
          Bucket     : s3://{cfg['s3_bucket']}/artifacts/sauron-edge/{cfg['version']}/
          Target ARN : {cfg['target_arn']}
          Deployment : {cfg['deployment_name']}
          AWS Profile: {cfg['aws_profile'] or '(default / env vars)'}
          Dashboard  : port {cfg['dashboard_port']}
        """).rstrip())
    return cfg


# ---------------------------------------------------------------------------
# Step 2 — Build zip artifact
# ---------------------------------------------------------------------------


def step_build_zip(cfg: Dict[str, str]) -> Path:
    """Create a zip of src/ + pyproject.toml in memory, write to dist/."""
    version = cfg["version"]
    dist_dir = _EDGE_DIR / "dist"
    dist_dir.mkdir(exist_ok=True)

    zip_name = f"sauron-edge-{version}.zip"
    zip_path = dist_dir / zip_name

    print(_h(f"\nBuilding artifact: {zip_path.relative_to(_EDGE_DIR)}"))

    # Paths to include in the zip
    include_roots = [
        _EDGE_DIR / "src",
        _EDGE_DIR / "pyproject.toml",
        _EDGE_DIR / "README.md",
        _EDGE_DIR / "config.example.yaml",
        _EDGE_DIR / ".env.example",
    ]

    # Exclude patterns
    exclude = {".venv", "__pycache__", ".mypy_cache", "dist", ".git", "*.pyc", "*.pyo"}

    def _should_exclude(p: Path) -> bool:
        for part in p.parts:
            if part in exclude or part.endswith(".pyc") or part.endswith(".pyo"):
                return True
        return False

    files_added = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.is_file():
                arcname = root.relative_to(_EDGE_DIR)
                zf.write(root, arcname)
                files_added += 1
                print(f"    + {arcname}")
            elif root.is_dir():
                for f in sorted(root.rglob("*")):
                    if f.is_file() and not _should_exclude(f.relative_to(_EDGE_DIR)):
                        arcname = f.relative_to(_EDGE_DIR)
                        zf.write(f, arcname)
                        files_added += 1
                        print(f"    + {arcname}")

    size_kb = zip_path.stat().st_size // 1024
    print(_ok(f"Built {zip_name} — {files_added} files, {size_kb} KB"))
    return zip_path


# ---------------------------------------------------------------------------
# Step 3 — Upload artifact to S3
# ---------------------------------------------------------------------------


def step_upload_s3(cfg: Dict[str, str], zip_path: Path, boto_session) -> str:
    """Upload the zip to S3 and return the S3 URI."""
    version = cfg["version"]
    bucket = cfg["s3_bucket"]
    key = f"artifacts/sauron-edge/{version}/{zip_path.name}"
    s3_uri = f"s3://{bucket}/{key}"

    print(_h(f"\nUploading to {s3_uri}"))
    print(f"  Source : {zip_path}")
    print(f"  Size   : {zip_path.stat().st_size // 1024} KB")

    s3 = boto_session.client("s3")
    s3.upload_file(
        str(zip_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/zip"},
    )
    print(_ok(f"Uploaded: {s3_uri}"))
    return s3_uri


# ---------------------------------------------------------------------------
# Step 4 — Build and upload recipe
# ---------------------------------------------------------------------------


def _build_recipe(cfg: Dict[str, str], artifact_uri: str) -> Dict[str, Any]:
    """Construct the Greengrass component recipe as a Python dict."""
    return {
        "RecipeFormatVersion": "2020-01-25",
        "ComponentName": cfg["component_name"],
        "ComponentVersion": cfg["version"],
        "ComponentDescription": (
            "Sauron edge surveillance component. Captures video, detects moving objects "
            "via MOG2 background subtraction, reads GPS and IMU sensors, and publishes "
            "telemetry to AWS IoT Core via Greengrass IPC."
        ),
        "ComponentPublisher": "Sauron",
        "ComponentConfiguration": {
            "DefaultConfiguration": {
                "configFilePath": "~/.config/sauron/config.yaml",
                "accessControl": {
                    "aws.greengrass.ipc.mqttproxy": {
                        f"{cfg['component_name']}:mqttproxy:1": {
                            "policyDescription": (
                                "Allow the edge component to publish telemetry to its device topic."
                            ),
                            "operations": ["aws.greengrass#PublishToIoTCore"],
                            "resources": ["devices/*/telemetry"],
                        }
                    }
                },
            }
        },
        "ComponentDependencies": {
            "aws.greengrass.Nucleus": {
                "VersionRequirement": cfg["nucleus_req"],
                "DependencyType": "SOFT",
            }
        },
        "Manifests": [
            {
                "Platform": {"os": "linux"},
                "Lifecycle": {
                    "Install": {
                        "RequiresPrivilege": True,
                        "Script": textwrap.dedent(f"""\
                            set -e
                            
                            # Install system-level C libraries required by PiWheels (NumPy/OpenCV)
                            apt-get update
                            apt-get install -y libopenblas-dev libatlas-base-dev
                            
                            if ! command -v uv &>/dev/null; then
                              curl -LsSf https://astral.sh/uv/install.sh | sh
                              export PATH="$HOME/.local/bin:$PATH"
                            fi
                            cd {{artifacts:decompressedPath}}/sauron-edge-{cfg['version']}
                            uv sync --python 3.11.2 --extra-index-url https://www.piwheels.org/simple
                        """),
                    },
                    "Run": {
                        "RequiresPrivilege": True,
                        "Script": textwrap.dedent(f"""\
                            set -e
                            export PATH="$HOME/.local/bin:$PATH"
                            export CONFIGURATION_FILE_PATH={{configuration:/configFilePath}}
                            cd {{artifacts:decompressedPath}}/sauron-edge-{cfg['version']}
                            libcamerify uv run sauron-edge
                        """),
                    },
                },
                "Artifacts": [
                    {
                        "URI": artifact_uri,
                        "Unarchive": "ZIP",
                    }
                ],
            }
        ],
    }


def step_create_component_version(
    cfg: Dict[str, str],
    artifact_uri: str,
    boto_session,
) -> str:
    """Call greengrassv2:CreateComponentVersion and return the component ARN."""
    recipe = _build_recipe(cfg, artifact_uri)
    recipe_json = json.dumps(recipe, indent=2)

    print(_h("\nCreating Greengrass component version"))
    print(f"  Component : {cfg['component_name']} @ {cfg['version']}")
    print("  Recipe preview (first 30 lines):")
    for line in recipe_json.splitlines()[:30]:
        print(f"    {line}")
    print("    ...")

    gg = boto_session.client("greengrassv2")
    response = gg.create_component_version(
        inlineRecipe=recipe_json.encode("utf-8"),
    )

    arn = response["arn"]
    status = response.get("status", {})
    print(_ok("Component version created"))
    print(f"  ARN    : {arn}")
    print(f"  Status : {status.get('componentState', 'unknown')}")
    return arn


# ---------------------------------------------------------------------------
# Step 5 — Create Greengrass deployment
# ---------------------------------------------------------------------------


def step_create_deployment(
    cfg: Dict[str, str],
    component_arn: str,
    boto_session,
) -> str:
    """Create a Greengrass deployment targeting cfg['target_arn']. Returns deployment ID."""
    component_name = cfg["component_name"]
    version = cfg["version"]
    target_arn = cfg["target_arn"]
    deployment_name = cfg["deployment_name"]

    print(_h("\nCreating Greengrass deployment"))
    print(f"  Name       : {deployment_name}")
    print(f"  Target ARN : {target_arn}")
    print(f"  Component  : {component_name} @ {version}")

    components = {
        component_name: {
            "componentVersion": version,
            "configurationUpdate": {
                "merge": json.dumps(
                    {
                        "configFilePath": "/home/pi/.config/sauron/config.yaml",
                    }
                )
            },
        }
    }

    gg = boto_session.client("greengrassv2")
    response = gg.create_deployment(
        targetArn=target_arn,
        deploymentName=deployment_name,
        components=components,
        deploymentPolicies={
            "failureHandlingPolicy": "ROLLBACK",
            "componentUpdatePolicy": {
                "action": "NOTIFY_COMPONENTS",
                "timeoutInSeconds": 60,
            },
        },
    )

    deployment_id = response["deploymentId"]
    print(_ok("Deployment created"))
    print(f"  Deployment ID : {deployment_id}")
    print(f"  IoT Job ID    : {response.get('iotJobId', 'N/A')}")
    print(f"  IoT Job ARN   : {response.get('iotJobArn', 'N/A')}")
    return deployment_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy Sauron Edge to AWS Greengrass v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full dry-run preview
          uv run --extra deploy python scripts/deploy.py --version 1.0.0 --dry-run

          # Interactive deploy (confirms each step)
          uv run --extra deploy python scripts/deploy.py --version 1.0.0

          # Non-interactive (CI / scripted)
          uv run --extra deploy python scripts/deploy.py --version 1.0.0 --yes

          # Skip Greengrass steps (artifact upload only)
          uv run --extra deploy python scripts/deploy.py --version 1.0.0 --skip-deployment
        """),
    )
    parser.add_argument(
        "--version",
        "-v",
        required=True,
        help="Component version to deploy (e.g. 1.0.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without calling AWS",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm all steps (non-interactive)",
    )
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Stop after creating the component version; do not create a deployment",
    )
    parser.add_argument(
        "--skip-component-version",
        action="store_true",
        help="Skip CreateComponentVersion (re-use an already-created version)",
    )
    args = parser.parse_args()

    TOTAL_STEPS = 5 if not args.skip_deployment else 4

    print(f"\n{BOLD}{'=' * 56}")
    print("  Sauron Edge — Greengrass Deploy Script")
    print(f"{'=' * 56}{RESET}")
    if args.dry_run:
        print(f"\n{YELLOW}  DRY RUN MODE — no AWS calls will be made{RESET}")

    # --- Step 1: Validate ---
    _step(1, TOTAL_STEPS, "Validate environment")
    cfg = step_validate(args.version)

    # --- boto3 session (created once, shared) ---
    boto_session = None
    if not args.dry_run:
        try:
            import boto3
        except ImportError:
            print(_err("boto3 is not installed. Run: uv pip install boto3"))
            print("       Or install the deploy extras: uv sync --extra deploy")
            sys.exit(1)

        session_kwargs: Dict[str, Any] = {"region_name": cfg["aws_region"]}
        if cfg["aws_profile"]:
            session_kwargs["profile_name"] = cfg["aws_profile"]

        boto_session = boto3.Session(**session_kwargs)

        # Validate credentials early
        try:
            sts = boto_session.client("sts")
            identity = sts.get_caller_identity()
            print(
                _ok(
                    f"AWS credentials valid — Account: {identity['Account']} "
                    f"User: {identity['Arn'].split('/')[-1]}"
                )
            )
        except Exception as exc:
            print(_err(f"AWS credentials invalid or STS unreachable: {exc}"))
            sys.exit(1)

    # --- Step 2: Build zip ---
    _step(2, TOTAL_STEPS, "Build artifact zip")
    if _confirm("Build zip artifact from src/", args.yes, args.dry_run):
        zip_path = step_build_zip(cfg)
    elif args.dry_run:
        zip_path = _EDGE_DIR / "dist" / f"sauron-edge-{cfg['version']}.zip"
        print(f"  [DRY RUN] Would build: {zip_path.name}")
    else:
        print("  Skipped by user.")
        sys.exit(0)

    # --- Step 3: Upload to S3 ---
    _step(3, TOTAL_STEPS, "Upload artifact to S3")
    artifact_uri = (
        f"s3://{cfg['s3_bucket']}/artifacts/sauron-edge/{cfg['version']}/"
        f"sauron-edge-{cfg['version']}.zip"
    )
    if _confirm(f"Upload {zip_path.name} to {artifact_uri}", args.yes, args.dry_run):
        artifact_uri = step_upload_s3(cfg, zip_path, boto_session)
    else:
        print(f"  [DRY RUN] artifact URI would be: {artifact_uri}")

    # --- Step 4: Create component version ---
    component_arn: Optional[str] = None
    if not args.skip_component_version:
        _step(4, TOTAL_STEPS, "Create Greengrass component version")
        if _confirm(
            f"Call greengrassv2:CreateComponentVersion for {cfg['component_name']} @ {cfg['version']}",
            args.yes,
            args.dry_run,
        ):
            component_arn = step_create_component_version(
                cfg, artifact_uri, boto_session
            )
        else:
            print(
                f"  [DRY RUN] Would create component {cfg['component_name']}:{cfg['version']}"
            )
    else:
        print(
            f"\n{YELLOW}  Skipping CreateComponentVersion (--skip-component-version){RESET}"
        )

    # --- Step 5: Create deployment ---
    if not args.skip_deployment:
        _step(5, TOTAL_STEPS, "Create Greengrass deployment")
        if _confirm(
            f"Call greengrassv2:CreateDeployment targeting {cfg['target_arn']}",
            args.yes,
            args.dry_run,
        ):
            step_create_deployment(cfg, component_arn or "", boto_session)
        else:
            print("  [DRY RUN] Would create deployment")
    else:
        print(f"\n{YELLOW}  Skipping CreateDeployment (--skip-deployment){RESET}")

    # --- Done ---
    print(f"\n{BOLD}{GREEN}{'=' * 56}")
    print("  Done!")
    print(f"{'=' * 56}{RESET}\n")


if __name__ == "__main__":
    main()
