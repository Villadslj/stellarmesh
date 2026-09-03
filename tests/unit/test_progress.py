import logging
import time

from stellarmesh._progress import log_progress, progress_heartbeat


def test_progress_heartbeat(caplog):
    logger = logging.getLogger("stellarmesh.progress-test")
    with caplog.at_level(logging.INFO):
        with progress_heartbeat(logger, "Test operation", interval=0.01):
            time.sleep(0.025)

    messages = [record.message for record in caplog.records]
    assert "Test operation started." in messages
    assert any("Test operation is still running" in message for message in messages)
    assert any("Test operation completed in" in message for message in messages)


def test_log_progress(caplog):
    logger = logging.getLogger("stellarmesh.progress-test")
    with caplog.at_level(logging.INFO):
        for completed in range(1, 101):
            log_progress(logger, "Processing", completed, 100)

    messages = [record.message for record in caplog.records]
    assert "Processing: 1/100 (1%)." in messages
    assert "Processing: 50/100 (50%)." in messages
    assert "Processing: 100/100 (100%)." in messages
