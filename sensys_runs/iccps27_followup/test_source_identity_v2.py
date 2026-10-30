"""Source representation and receipt-integrity tests without changing Git config."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from sensys_runs.iccps27_followup import source_identity_v2 as identity


OLD = "a" * 40
CURRENT = "b" * 40
MANIFEST = "c" * 64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


class FakeGit:
    def __init__(self, committed, filtered=None):
        self.committed = committed
        self.filtered = blob_id(committed) if filtered is None else filtered
        self.head_reads = self.ref_reads = 0
        self.calls = []
        self.move_head = self.move_ref = False
        self.on_final_ref = None

    def __call__(self, repo, *args):
        self.calls.append(args)
        if args == ("rev-parse", "HEAD"):
            self.head_reads += 1
            return (CURRENT if not self.move_head or self.head_reads == 1 else OLD).encode()
        if args[:3] == ("rev-parse", "--verify", "--end-of-options"):
            if args[3] != "approved-old^{commit}":
                raise identity.SourceIdentityError("Unknown approved ref")
            self.ref_reads += 1
            if self.ref_reads % 2 == 0 and self.on_final_ref:
                self.on_final_ref()
            return (OLD if not self.move_ref or self.ref_reads == 1 else CURRENT).encode()
        if args == ("rev-parse", OLD + ":module.txt"):
            return blob_id(self.committed).encode()
        if args == ("show", OLD + ":module.txt"):
            return self.committed
        if args[:2] == ("hash-object", "--path=module.txt"):
            return self.filtered.encode()
        raise AssertionError("Unexpected or mutating Git command: " + repr(args))


@pytest.fixture
def setup_identity(tmp_path, monkeypatch):
    def make(committed=b"first\nsecond\n", working=None, filtered=None):
        raw = committed if working is None else working
        (tmp_path / "module.txt").write_bytes(raw)
        (tmp_path / "protected.txt").write_bytes(b"protected\n")
        git = FakeGit(committed, filtered)
        monkeypatch.setattr(identity, "_git", git)
        kwargs = dict(repo=tmp_path, source_ref="approved-old",
                      source_files=("module.txt", "protected.txt"),
                      protected_pins={"protected.txt": digest(b"protected\n")},
                      protected_manifest_sha256=MANIFEST)
        expected = dict(commit=OLD, blobs={"module.txt": blob_id(committed)},
                        protected_sha256=kwargs["protected_pins"],
                        protected_manifest_sha256=MANIFEST)
        return kwargs, git, expected
    return make


def test_exact_committed_crlf_is_accepted_without_clean_filter(setup_identity):
    kwargs, git, expected = setup_identity(b"first\r\nsecond\r\n", filtered="wrong_clean_hash")
    audit = {}
    assert identity.verified_source_identity(**kwargs, expected_identity=expected, audit=audit) == expected
    assert audit["representations"]["module.txt"] == "exact_committed_bytes"
    assert not any(call[0] == "hash-object" for call in git.calls)


def test_uniform_crlf_checkout_requires_matching_filter_and_keeps_historical_commit(setup_identity):
    kwargs, git, expected = setup_identity(working=b"first\r\nsecond\r\n")
    audit = {}
    assert identity.verified_source_identity(**kwargs, expected_identity=expected, audit=audit) == expected
    assert audit["checkout_commit"] == CURRENT
    assert audit["source_commit"] == OLD
    assert audit["raw_sha256"]["module.txt"] == digest(b"first\r\nsecond\r\n")
    assert audit["representations"]["module.txt"] == "uniform_CRLF_checkout_of_LF_blob_with_matching_Git_filter"
    assert sum(call[0] == "hash-object" for call in git.calls) == 1


@pytest.mark.parametrize("working", [
    b"FIRST\r\nsecond\r\n", b"first\r\nsecond\n",
    b"first\r\nsecond \r\n", b"\xef\xbb\xbffirst\r\nsecond\r\n",
    b"first\r\nsecond", b"first\r\r\nsecond\r\n",
])
def test_content_or_nonuniform_representation_is_rejected_even_with_matching_filter(setup_identity, working):
    kwargs, _, _ = setup_identity(working=working)
    with pytest.raises(identity.SourceIdentityError, match="accepted committed-byte representation"):
        identity.verified_source_identity(**kwargs)


def test_uniform_crlf_with_wrong_filtered_hash_is_rejected(setup_identity):
    kwargs, _, _ = setup_identity(working=b"first\r\nsecond\r\n", filtered="d" * 40)
    with pytest.raises(identity.SourceIdentityError, match="accepted committed-byte representation"):
        identity.verified_source_identity(**kwargs)


def test_reverse_crlf_normalization_is_not_an_allowed_fallback(setup_identity):
    kwargs, _, _ = setup_identity(b"first\r\nsecond\r\n", working=b"first\nsecond\n")
    with pytest.raises(identity.SourceIdentityError, match="accepted committed-byte representation"):
        identity.verified_source_identity(**kwargs)


def test_protected_files_are_not_normalized(setup_identity):
    kwargs, _, _ = setup_identity()
    (kwargs["repo"] / "protected.txt").write_bytes(b"protected\r\n")
    with pytest.raises(identity.SourceIdentityError, match="Protected source hash mismatch"):
        identity.verified_source_identity(**kwargs)


def test_unknown_ref_is_rejected(setup_identity):
    kwargs, _, _ = setup_identity()
    kwargs["source_ref"] = "not-an-approved-ref"
    with pytest.raises(identity.SourceIdentityError, match="Unknown approved ref"):
        identity.verified_source_identity(**kwargs)


def test_head_moving_during_verification_is_rejected(setup_identity):
    kwargs, git, _ = setup_identity()
    git.move_head = True
    with pytest.raises(identity.SourceIdentityError, match="HEAD changed"):
        identity.verified_source_identity(**kwargs)


def test_source_ref_moving_during_verification_is_rejected(setup_identity):
    kwargs, git, _ = setup_identity()
    git.move_ref = True
    with pytest.raises(identity.SourceIdentityError, match="source ref moved"):
        identity.verified_source_identity(**kwargs)


def test_working_source_changing_during_verification_is_rejected(setup_identity):
    kwargs, git, _ = setup_identity()
    git.on_final_ref = lambda: (kwargs["repo"] / "module.txt").write_bytes(b"changed\n")
    with pytest.raises(identity.SourceIdentityError, match="Source bytes changed"):
        identity.verified_source_identity(**kwargs)


def test_receipt_identity_mismatch_is_rejected(setup_identity):
    kwargs, _, expected = setup_identity()
    expected["commit"] = CURRENT
    with pytest.raises(identity.SourceIdentityError, match="supplied receipt"):
        identity.verified_source_identity(**kwargs, expected_identity=expected)


def test_protected_coverage_and_unsafe_paths_are_rejected(setup_identity):
    kwargs, _, _ = setup_identity()
    kwargs["source_files"] = ("module.txt",)
    with pytest.raises(identity.SourceIdentityError, match="omits protected"):
        identity.verified_source_identity(**kwargs)
    kwargs["source_files"] = ("../module.txt", "protected.txt")
    with pytest.raises(identity.SourceIdentityError, match="Unsafe source path"):
        identity.verified_source_identity(**kwargs)


def test_scoped_check_restores_legacy_function_after_exception(setup_identity):
    kwargs, _, expected = setup_identity()
    original = lambda: "legacy sentinel"
    module = SimpleNamespace(source_git_identity=original)
    with pytest.raises(RuntimeError, match="downstream failure"):
        with identity.source_identity_scope(module, **kwargs, expected_identity=expected):
            assert module.source_git_identity() == expected
            raise RuntimeError("downstream failure")
    assert module.source_git_identity is original
    assert module.source_git_identity() == "legacy sentinel"
