from dradar.patch_guard import (
    POMPEII_PATCH_MAX_BYTES,
    check_pompeii_patch,
    format_patch_guard_report,
    inspect_patch,
)


def _text_patch(path: str = "model_answer.json", value: bytes = b'{"edges":[]}') -> bytes:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 9e26dfe..e9cfe4b 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-{}\n"
    ).encode() + b"+" + value + b"\n"


def test_accepts_only_small_text_model_answer_patch():
    result = check_pompeii_patch(_text_patch())

    assert result.accepted is True
    assert result.violations == ()
    assert result.inspection.files[0].display_path == "model_answer.json"
    assert result.inspection.files[0].binary is False


def test_reports_unexpected_binary_file_and_its_patch_contribution():
    patch = _text_patch() + (
        b"diff --git a/generated/reconstruction.png "
        b"b/generated/reconstruction.png\n"
        b"new file mode 100644\n"
        b"index 0000000..1111111\n"
        b"GIT binary patch\n"
        b"literal 4\nLc${NkU|;|M00aO5\n\n"
    )

    result = check_pompeii_patch(patch)
    report = "\n".join(format_patch_guard_report(result))

    assert result.accepted is False
    assert "generated/reconstruction.png" in report
    assert "binary" in report
    assert "patch contribution" in report
    assert "changes 2 files" in report


def test_reports_oversized_model_answer_before_upload():
    patch = _text_patch(value=b'"' + b"x" * POMPEII_PATCH_MAX_BYTES + b'"')

    result = check_pompeii_patch(patch)

    assert result.accepted is False
    assert any("local Pompeii limit" in item for item in result.violations)


def test_malformed_patch_fails_closed_without_content_dump():
    inspection = inspect_patch(b"not a patch\n")

    assert inspection.files == ()
    assert inspection.parse_error is not None
