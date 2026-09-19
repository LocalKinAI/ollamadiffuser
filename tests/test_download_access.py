"""What happens when the hub says no.

A permission error is not a flaky connection, but the download path treated
every exception the same: five attempts with growing sleeps, the reason
logged once at WARNING, and the progress bars redrawing over it. A gated
pack therefore came back looking like a finished download of nothing — 21
files, 0 MB, exit 0, no model directory — which is a worse outcome than a
crash, because it looks like success.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

from ollamadiffuser.core.utils.download_utils import (
    access_problem,
    robust_snapshot_download,
)


def hub_error(cls, message, *, status_code=403, repo_id=None):
    """Build a hub error the way the installed huggingface_hub wants it.

    hub 1.x made ``response`` a required keyword-only argument on
    HfHubHTTPError; older versions took the message alone. Constructing these
    by hand in each test pinned us to one of those, so do it in one place and
    let either shape work.
    """
    try:
        import httpx
        response = httpx.Response(
            status_code, request=httpx.Request("GET", "https://huggingface.co")
        )
        error = cls(message, response=response)
    except TypeError:                       # pre-1.x: message only
        error = cls(message)
    if repo_id is not None:
        error.repo_id = repo_id
    return error


class TestAccessProblem:
    def test_a_gated_repo_says_where_to_ask(self):
        error = hub_error(GatedRepoError, '403 Client Error',
                          repo_id='dgrauet/ltx-2.5-mlx-q8')
        advice = access_problem(error)
        assert 'huggingface.co/dgrauet/ltx-2.5-mlx-q8' in advice
        assert 'Request access' in advice
        assert 'hf auth login' in advice        # not the retired huggingface-cli

    def test_a_missing_repo_says_check_the_id(self):
        assert 'repo_id' in access_problem(
            hub_error(RepositoryNotFoundError, '404', status_code=404))

    def test_a_missing_file_points_at_the_patterns(self):
        assert 'allow_patterns' in access_problem(
            hub_error(EntryNotFoundError, '404', status_code=404))

    def test_an_ordinary_failure_is_not_one_of_these(self):
        assert access_problem(ConnectionResetError('reset by peer')) is None
        assert access_problem(TimeoutError('timed out')) is None


class TestNoRetryOnRefusal:
    def _refuse(self, error, expect=RuntimeError):
        """Run a download that always hits `error`; report what it did.

        An ordinary failure still comes back as itself after the retries —
        only a refusal is re-raised as a RuntimeError carrying the advice.
        """
        calls = {'n': 0}

        def fake_download(**kwargs):
            calls['n'] += 1
            raise error

        with patch('ollamadiffuser.core.utils.download_utils.snapshot_download',
                   side_effect=fake_download), \
             patch('ollamadiffuser.core.utils.download_utils.get_repo_file_list',
                   return_value={}), \
             patch('ollamadiffuser.core.utils.download_utils.time.sleep') as slept:
            with pytest.raises(expect) as caught:
                robust_snapshot_download(repo_id='dgrauet/ltx-2.5-mlx-q8',
                                         local_dir='/tmp/nope-does-not-exist',
                                         max_retries=5)
        return calls['n'], slept, str(caught.value)

    def test_a_gated_repo_is_tried_once(self):
        error = hub_error(GatedRepoError, '403',
                          repo_id='dgrauet/ltx-2.5-mlx-q8')
        tries, slept, message = self._refuse(error)
        assert tries == 1, 'retrying a 403 only buries the reason'
        slept.assert_not_called()
        assert 'Request access' in message

    def test_the_message_names_the_repo(self):
        error = hub_error(GatedRepoError, '403',
                          repo_id='dgrauet/ltx-2.5-mlx-q8')
        _, _, message = self._refuse(error)
        assert message.startswith('dgrauet/ltx-2.5-mlx-q8:')

    def test_an_ordinary_error_still_retries(self):
        tries, slept, _ = self._refuse(ConnectionResetError('reset by peer'),
                                       expect=ConnectionResetError)
        assert tries == 5
        assert slept.call_count == 4
