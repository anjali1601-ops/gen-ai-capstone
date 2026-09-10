"""Local username/password authentication (no cloud IdP).

Demo accounts are listed in ``config/users.example.yaml``. On first run the
app writes ``config/users.yaml`` with PBKDF2-SHA256 hashes. That hashed file
is gitignored. Passwords are never logged or shown in the UI.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

EXAMPLE_USERS_PATH = PROJECT_ROOT / "config" / "users.example.yaml"
USERS_PATH = PROJECT_ROOT / "config" / "users.yaml"

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 120_000
ADMIN_ROLE = "admin"
KNOWN_ROLES = frozenset({"admin", "analyst", "user"})
_PLACEHOLDER_SECRETS = frozenset({"", "changeme", "xxx", "todo", "your_app_secret"})

MAX_LOGIN_ATTEMPTS = 12


@dataclass(frozen=True)
class AuthUser:
    username: str
    name: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN_ROLE

    @property
    def role_label(self) -> str:
        labels = {"admin": "Admin", "analyst": "Analyst", "user": "User"}
        return labels.get(self.role, self.role.title())


@dataclass
class UserStore:
    users: list[AuthUser]
    password_hashes: dict[str, str]
    path: Path
    error: str | None = None

    def get(self, username: str) -> AuthUser | None:
        key = _norm_username(username)
        for user in self.users:
            if _norm_username(user.username) == key:
                return user
        return None


def _norm_username(value: str) -> str:
    return value.strip().lower()


def _pepper() -> str:
    raw = os.getenv("APP_SECRET", "").strip()
    if raw.lower() in _PLACEHOLDER_SECRETS:
        return ""
    return raw


def hash_password(password: str, *, pepper: str | None = None) -> str:
    if pepper is None:
        pepper = _pepper()
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        (pepper + password).encode("utf-8"),
        bytes.fromhex(salt),
        ITERATIONS,
    )
    return f"{ALGORITHM}${ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str, *, pepper: str | None = None) -> bool:
    if pepper is None:
        pepper = _pepper()
    try:
        algorithm, iterations, salt, digest = stored.split("$", 3)
        if algorithm != ALGORITHM:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            (pepper + password).encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        )
        return hmac.compare_digest(candidate.hex(), digest)
    except (TypeError, ValueError):
        return False


def _normalise_role(value: object) -> str:
    role = str(value or "").strip().lower()
    if role in KNOWN_ROLES:
        return role
    return "user"


def _load_yaml(path: Path) -> tuple[Any, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"Could not read {path.name}: {exc}"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, f"Could not parse {path.name}: {exc}"
    return data, None


def _rows_from_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        rows = data.get("users", [])
    elif isinstance(data, list):
        rows = data
    else:
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _parse_users(data: Any) -> tuple[list[AuthUser], dict[str, str], dict[str, str], bool]:
    """Return users, hashes, leftover plaintext passwords, and whether a rewrite is needed."""
    users: list[AuthUser] = []
    hashes: dict[str, str] = {}
    plaintext: dict[str, str] = {}
    seen: set[str] = set()
    needs_rewrite = False

    for row in _rows_from_payload(data):
        username = str(row.get("username") or "").strip()
        if not username:
            continue
        key = _norm_username(username)
        if key in seen:
            logger.warning("Skipping duplicate username in user file: %s", username)
            continue
        seen.add(key)
        name = str(row.get("name") or username).strip() or username
        role = _normalise_role(row.get("role"))
        stored_hash = str(row.get("password_hash") or "").strip()
        password = str(row.get("password") or "")
        if stored_hash:
            hashes[key] = stored_hash
            if password:
                needs_rewrite = True
        elif password:
            plaintext[key] = password
            needs_rewrite = True
        else:
            logger.warning("Skipping user %s: no password or password_hash", username)
            continue
        users.append(AuthUser(username=username, name=name, role=role))

    return users, hashes, plaintext, needs_rewrite


def _dump_users(path: Path, users: list[AuthUser], hashes: dict[str, str]) -> str | None:
    payload = {
        "users": [
            {
                "username": user.username,
                "name": user.name,
                "role": user.role,
                "password_hash": hashes[_norm_username(user.username)],
            }
            for user in users
            if _norm_username(user.username) in hashes
        ]
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as exc:
        return f"Could not write hashed user file: {exc}"
    return None


def _hash_plaintext(plaintext: dict[str, str]) -> dict[str, str]:
    return {key: hash_password(password) for key, password in plaintext.items()}


def load_user_store(path: Path | None = None) -> UserStore:
    """Load accounts, hashing any leftover plaintext passwords into users.yaml."""
    hashed_path = path or USERS_PATH
    source = hashed_path if hashed_path.is_file() else EXAMPLE_USERS_PATH
    if not source.is_file():
        return UserStore(
            users=[],
            password_hashes={},
            path=hashed_path,
            error=(
                "No user file found. Copy config/users.example.yaml to "
                "config/users.yaml and restart the app."
            ),
        )

    data, error = _load_yaml(source)
    if error:
        return UserStore(users=[], password_hashes={}, path=hashed_path, error=error)

    users, hashes, plaintext, needs_rewrite = _parse_users(data)
    if not users:
        return UserStore(
            users=[],
            password_hashes={},
            path=hashed_path,
            error=f"No valid user accounts in {source.name}.",
        )

    if plaintext:
        hashes.update(_hash_plaintext(plaintext))

    if needs_rewrite or source != hashed_path:
        write_error = _dump_users(hashed_path, users, hashes)
        if write_error:
            if plaintext and source == hashed_path:
                return UserStore(
                    users=[],
                    password_hashes={},
                    path=hashed_path,
                    error=write_error,
                )
            logger.warning("Could not persist hashed users: %s", write_error)

    return UserStore(users=users, password_hashes=hashes, path=hashed_path)


def authenticate(username: str, password: str, store: UserStore | None = None) -> AuthUser | None:
    if store is None:
        store = load_user_store()
    if store.error or not username.strip() or password == "":
        return None
    user = store.get(username)
    if user is None:
        return None
    stored = store.password_hashes.get(_norm_username(username), "")
    if not stored or not verify_password(password, stored):
        return None
    return user


def directory_rows(store: UserStore | None = None) -> list[dict[str, str]]:
    if store is None:
        store = load_user_store()
    if store.error:
        return []
    return [
        {"username": user.username, "name": user.name, "role": user.role_label}
        for user in store.users
    ]
