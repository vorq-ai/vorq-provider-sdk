"""Mapping template engine + JSONPath extraction."""

from __future__ import annotations

import re

import pytest

from vorqd._templates import extract, render


def ctx(**kw):
    base = {"input": {}, "job": {}, "base_url": "http://b", "api_key": "k", "model": "m"}
    base.update(kw)
    return base


def test_object_body_substitution():
    body = {"prompt": "{input.prompt}", "max_tokens": "{job.units_out}"}
    out = render(body, ctx(input={"prompt": "hi"}, job={"units_out": 128}))
    assert out == {"prompt": "hi", "max_tokens": 128}  # type preserved (int)


def test_embedded_string_stringifies():
    out = render("Key {api_key}", ctx(api_key="sekret"))
    assert out == "Key sekret"


def test_top_level_array_with_uuid_unique():
    body = [{"taskType": "submit", "taskUUID": "{uuid}"}]
    a = render(body, ctx())
    b = render(body, ctx())
    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    assert uuid_re.match(a[0]["taskUUID"])
    assert a[0]["taskUUID"] != b[0]["taskUUID"]


def test_submit_scope_with_indexing():
    c = ctx(submit={"data": [{"taskUUID": "abc-123"}]})
    assert render("{submit.data[0].taskUUID}", c) == "abc-123"


def test_jsonpath_left_intact_by_render():
    # $.-prefixed markers are extraction targets, not template tokens
    assert render("$.status_url", ctx()) == "$.status_url"


def test_unresolved_token_raises():
    with pytest.raises(Exception):
        render("{input.missing}", ctx(input={}))


def test_extract_scalar():
    body = {"choices": [{"message": {"content": "hello"}}], "usage": {"completion_tokens": 9}}
    assert extract("$.choices[0].message.content", body) == "hello"
    assert extract("$.usage.completion_tokens", body) == 9
    assert extract("$.nope", body) is None


def test_extract_wildcard_list():
    body = {"images": [{"url": "a"}, {"url": "b"}]}
    assert extract("$.images[*].url", body) == ["a", "b"]


# --- optional tokens, and byte-carrying fields --------------------------------


def test_a_data_uri_renders_a_reference_into_the_field_a_backend_expects():
    """A reference arrives as bytes inside the sealed payload, and the prevailing
    interface takes a URL. A data URI is the bridge, and it needs no new
    primitive: base64 contains no braces, so embedded substitution is safe.
    """
    body = render({"image_url": "data:{input.image.media_type};base64,{input.image.b64}"},
                  {"input": {"image": {"media_type": "image/png", "b64": "QUJD"}}})
    assert body["image_url"] == "data:image/png;base64,QUJD"


def test_a_whole_token_keeps_the_value_it_resolved_to():
    body = render({"image": "{input.image.b64}"},
                  {"input": {"image": {"b64": "QUJD"}}})
    assert body["image"] == "QUJD"


def test_an_optional_token_drops_its_key_when_nothing_resolves():
    """An end frame is optional, and there is no way to say so without this: a
    request that carries none must send no `end_image_url` at all, not a null and
    not the literal token.
    """
    body = render({"image_url": "{input.image.b64}", "end_image_url": "{input.end_image.b64?}"},
                  {"input": {"image": {"b64": "QUJD"}}})
    assert body == {"image_url": "QUJD"}


def test_an_optional_token_that_does_resolve_is_kept():
    body = render({"end_image_url": "{input.end_image.b64?}"},
                  {"input": {"end_image": {"b64": "WFla"}}})
    assert body["end_image_url"] == "WFla"


def test_an_optional_token_embedded_in_a_string_drops_the_key_too():
    """The data URI is the form a reference is sent in, so it is the form an
    optional one has to be expressible in. Rendering the missing half as nothing
    would send `data:;base64,` — a field that is present and wrong, which a backend
    answers with a 400 where an absent one is simply no end frame.
    """
    body = render({"end": "data:{input.end_image.media_type?};base64,{input.end_image.b64?}"},
                  {"input": {}})
    assert body == {}


def test_an_embedded_optional_token_that_resolves_is_rendered_in_place():
    body = render({"end": "data:{input.end_image.media_type?};base64,{input.end_image.b64?}"},
                  {"input": {"end_image": {"media_type": "image/png", "b64": "WFla"}}})
    assert body == {"end": "data:image/png;base64,WFla"}


def test_an_optional_token_in_a_list_leaves_no_element_behind():
    """Some backends take their frames as an array. The sentinel that marks a
    dropped value must never be what is serialised in its place."""
    body = render({"frames": ["{input.image.b64}", "{input.end_image.b64?}"]},
                  {"input": {"image": {"b64": "QUJD"}}})
    assert body == {"frames": ["QUJD"]}


def test_a_fragment_that_is_nothing_but_a_missing_optional_token_is_an_error():
    """A key can be dropped and so can an element. A whole URL or a whole body
    cannot: there is nothing to drop it from."""
    with pytest.raises(KeyError, match="end_image"):
        render("{input.end_image.b64?}", {"input": {}})


def test_a_required_token_still_raises_so_a_typo_is_not_silently_dropped():
    with pytest.raises(KeyError, match="end_image"):
        render({"end_image_url": "{input.end_image.b64}"}, {"input": {}})


# --- a result nested inside a JSON string ---------------------------------------


def test_a_path_that_continues_past_a_string_reads_it_as_json():
    """Some task APIs answer with their result serialized *into* a string field. A
    path that goes on below a string can match nothing any other way, so going on
    is the instruction to parse it."""
    body = {"data": {"state": "success",
                     "result": '{"urls": ["http://cdn/a.mp4", "http://cdn/b.mp4"], "n": 2}'}}
    assert extract("$.data.result.urls[*]", body) == ["http://cdn/a.mp4", "http://cdn/b.mp4"]
    assert extract("$.data.result.n", body) == 2


def test_a_string_is_never_parsed_when_the_path_stops_at_it():
    body = {"data": {"result": '{"urls": []}'}}
    assert extract("$.data.result", body) == '{"urls": []}'


def test_a_string_that_is_not_json_matches_nothing_rather_than_raising():
    assert extract("$.data.result.urls[*]", {"data": {"result": "{not json"}}) == []
    assert extract("$.data.result.n", {"data": {"result": "plain"}}) is None


# --- list-valued tokens ---------------------------------------------------------


def test_a_wildcard_token_renders_the_whole_list():
    ctx = {"prepare": {"refs": [{"url": "http://f/1"}, {"url": "http://f/2"}]}}
    assert render({"urls": "{prepare.refs[*].url}"}, ctx) == {"urls": ["http://f/1", "http://f/2"]}


def test_an_optional_wildcard_token_with_nothing_to_list_drops_its_key():
    """An empty list is not the same request as no list: a backend that validates
    the field refuses `[]` where it would have accepted silence."""
    assert render({"p": "x", "urls": "{prepare.refs[*].url?}"}, {"prepare": {}}) == {"p": "x"}
    assert render({"p": "x", "urls": "{prepare.refs[*].url?}"}, {"prepare": {"refs": []}}) == {"p": "x"}


def test_a_wildcard_token_inside_a_list_is_spliced_into_it():
    """A backend takes one list of clip URLs; this network offers a singular
    `video` and a listed `reference_videos`. One list has to hold both."""
    ctx = {"prepare": {"clip": {"url": "http://f/0"}, "clips": [{"url": "http://f/1"}, {"url": "http://f/2"}]}}
    body = render({"urls": ["{prepare.clip.url?}", "{prepare.clips[*].url?}"]}, ctx)
    assert body == {"urls": ["http://f/0", "http://f/1", "http://f/2"]}


def test_a_list_whose_every_element_dropped_takes_its_key_with_it():
    body = render({"p": "x", "urls": ["{prepare.clip.url?}", "{prepare.clips[*].url?}"]}, {"prepare": {}})
    assert body == {"p": "x"}
    assert render({"tags": []}, {}) == {"tags": []}       # an empty list the operator wrote stays
