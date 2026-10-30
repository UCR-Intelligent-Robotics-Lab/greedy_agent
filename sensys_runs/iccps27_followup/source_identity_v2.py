"""Strict source identities across approved Git refs and LF/CRLF checkouts.

This module does not edit Git configuration, files, or the legacy probe path.
The caller supplies the approved source ref and verifies the manifest file itself.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess


class SourceIdentityError(ValueError):
    """The approved source identity could not be established."""


def _git(repo, *arguments):
    completed = subprocess.run(["git", "-C", str(repo), *arguments],
                               capture_output=True, check=False)
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SourceIdentityError("Git source verification failed: " + detail)
    return completed.stdout


def _text(repo, *arguments):
    return _git(repo, *arguments).decode("ascii").strip()


def _source_path(repo, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise SourceIdentityError("Source names must be repository-relative POSIX paths")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
        raise SourceIdentityError("Unsafe source path: " + name)
    path = repo.joinpath(*parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo):
        raise SourceIdentityError("Source is absent, symbolic or outside the repository: " + name)
    return path


def accepted_representation(working, committed, filtered_blob_id, committed_blob_id):
    """Return the proven byte representation; permit no general normalization."""
    if working == committed:
        return "exact_committed_bytes"
    if (b"\r" not in committed and
            working == committed.replace(b"\n", b"\r\n") and
            filtered_blob_id == committed_blob_id):
        return "uniform_CRLF_checkout_of_LF_blob_with_matching_Git_filter"
    raise SourceIdentityError("Working source is not an accepted committed-byte representation")


def verified_source_identity(repo, source_ref, source_files, protected_pins,
                             protected_manifest_sha256, expected_identity=None,
                             audit=None):
    """Verify current bytes against an explicit approved historical/current ref.

    The return value preserves reset_probe.source_git_identity's schema. HEAD
    may differ from source_ref, but must not move during verification. An old
    receipt therefore retains its real old commit instead of being relabelled.
    Protected files always use exact SHA256 bytes, never newline conversion.
    The manifest digest is recorded here; the caller must verify that manifest
    file against this digest before relying on its supplied protected pins.
    """
    repo = Path(repo).resolve()
    names = tuple(source_files)
    pins = dict(protected_pins)
    if (not names or len(set(names)) != len(names) or not set(pins).issubset(names)):
        raise SourceIdentityError("Source coverage is empty, duplicated or omits protected files")
    if not isinstance(source_ref, str) or not source_ref or source_ref.startswith("-"):
        raise SourceIdentityError("An explicit approved source ref is required")
    if not re.fullmatch(r"[0-9a-f]{64}", protected_manifest_sha256):
        raise SourceIdentityError("Invalid protected manifest SHA256")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values()):
        raise SourceIdentityError("Invalid protected source SHA256")
    paths = {name: _source_path(repo, name) for name in names}
    head_before = _text(repo, "rev-parse", "HEAD")
    ref_argument = source_ref + "^{commit}"
    commit = _text(repo, "rev-parse", "--verify", "--end-of-options", ref_argument)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise SourceIdentityError("Approved source ref did not resolve to a commit")
    blobs, digests, representations = {}, {}, {}
    for name in names:
        raw = paths[name].read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        digests[name] = digest
        if name in pins:
            if digest != pins[name]:
                raise SourceIdentityError("Protected source hash mismatch: " + name)
            representations[name] = "exact_protected_SHA256"
            continue
        object_name = commit + ":" + name
        blob_id = _text(repo, "rev-parse", object_name)
        committed = _git(repo, "show", object_name)
        # A committed CRLF file can be changed by the Git clean filter even
        # though the checkout already has exactly the approved blob bytes.
        filtered = None if raw == committed else _text(
            repo, "hash-object", "--path=" + name, str(paths[name]))
        try:
            representations[name] = accepted_representation(raw, committed, filtered, blob_id)
        except SourceIdentityError as exc:
            raise SourceIdentityError(str(exc) + ": " + name) from exc
        blobs[name] = blob_id
    if _text(repo, "rev-parse", "--verify", "--end-of-options", ref_argument) != commit:
        raise SourceIdentityError("Approved source ref moved during verification")
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digests[name]:
            raise SourceIdentityError("Source bytes changed during verification: " + name)
    if _text(repo, "rev-parse", "HEAD") != head_before:
        raise SourceIdentityError("Git HEAD changed during source verification")
    identity = dict(commit=commit, blobs=blobs, protected_sha256=pins,
                    protected_manifest_sha256=protected_manifest_sha256)
    if expected_identity is not None and identity != expected_identity:
        raise SourceIdentityError("Verified source identity differs from the supplied receipt")
    if audit is not None:
        audit.update(source_ref=source_ref, source_commit=commit,
                     checkout_commit=head_before, raw_sha256=digests,
                     representations=representations)
    return identity


@contextmanager
def source_identity_scope(module, **verification_arguments):
    """Use this verifier only in the caller's new entry point, restoring it.

    This changes one module attribute in memory. It does not alter SOURCE_FILES,
    the legacy module file, or any Git setting. Callers should keep their work
    inside this scope and must not run a concurrent legacy probe in the process.
    """
    verified_source_identity(**verification_arguments)
    original = module.source_git_identity

    def current_verified_identity():
        return verified_source_identity(**verification_arguments)

    module.source_git_identity = current_verified_identity
    try:
        yield current_verified_identity
    finally:
        module.source_git_identity = original
