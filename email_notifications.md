# Email Notification System Documentation

This document summarizes the configuration, debugging process, and resolution for the email notification feature in the Loop News application.

## Goal

Configure and ensure the reliable operation of email notifications for users subscribed to topic updates, preventing duplicate emails and ensuring timely delivery based on user preferences (hourly/daily).

## Key Configuration Points (`config.py` / Environment Variables)

-   **`MAIL_USERNAME`**: Sender email address (e.g., Gmail account).
-   **`MAIL_PASSWORD`**: **App Password** generated from the email provider (e.g., Gmail App Password), **not** the regular account password. Requires 2-Step Verification to be enabled for the provider.
-   **`MAIL_SERVER`**: SMTP server address (e.g., `smtp.gmail.com`).
-   **`MAIL_PORT`**: SMTP server port (e.g., `587` for TLS).
-   **`MAIL_USE_TLS`**: Boolean (`True` or `False`) indicating whether to use TLS encryption. Usually `True` for port 587.
-   **`MAIL_SENDER_NAME`**: Display name for the sender in the email (e.g., "Loop News").

## Issues Encountered & Debugging Steps

1.  **Email Configuration Clarification:** Identified that `MAIL_PASSWORD` requires a provider-specific **App Password**, not the standard login password.
2.  **`AttributeError` in `save_notification_settings`:**
    *   **Issue:** Code was attempting `.get()` on a `tuple` instead of a `dict` value when iterating through `history.items()`.
    *   **Fix:** Correctly unpacked the dictionary key/value (`topic_key, entry_data`) in the loop within `app/services/notifications.py`.
3.  **Duplicate Notifications (Initial):**
    *   **Issue:** The notification scheduler (`schedule_notification_checks`) was being initialized within the Flask app factory (`create_app` in `app/__init__.py`). The Flask development server's reloader caused `create_app` to run twice, starting two scheduler threads.
    *   **Fix:** Moved the scheduler initialization call (`schedule_notification_checks(app)`) to `run.py`, executing it *after* `app = create_app(...)` but *before* `app.run(...)`.
4.  **Duplicate Notifications (Persistent):**
    *   **Issue:** Duplicate emails persisted even after moving the scheduler start. Suspected the Flask development server's reloader might still be interfering subtly.
    *   **Fix 1 (Mitigation):** Modified `run.py` to explicitly disable the reloader during development: `app.run(host=host, port=port, debug=debug, use_reloader=False)`. *Note: This requires manual server restarts during development.*
    *   **Fix 2 (Diagnosis & Verification):**
        *   Reverted temporary 1-minute timing changes in `app/services/notifications.py` back to the standard 15-minute check / hourly/daily logic to debug the intended behavior.
        *   Added detailed logging (timestamps, thread IDs, decision points, state previews) to `check_and_send_notifications` and `process_history_file` in `app/services/notifications.py`.
        *   Manually edited `search_history.json` to set an old `last_sent` timestamp, forcing an email send condition.
        *   Analyzed logs across two consecutive check cycles:
            *   **Check 1:** Confirmed email was sent, `last_sent` was updated in memory, and the history file containing the *new* timestamp was successfully saved to disk.
            *   **Check 2:** Confirmed the system read the *new* timestamp from the file and correctly determined *not* to send a duplicate email.

## Final State (Resolved)

-   Email configuration uses an **App Password** for `MAIL_PASSWORD`.
-   The `AttributeError` in `save_notification_settings` is fixed.
-   The notification scheduler is initialized **once** in `run.py`.
-   The Flask development server runs with `use_reloader=False` in `run.py` to prevent process-related issues.
-   The logic for updating and reading the `last_sent` timestamp in `search_history.json` is functioning correctly, preventing duplicate notifications.
-   Standard notification timing (15-minute checks, hourly/daily evaluation) is active.
-   *Note: Detailed diagnostic logging added during debugging remains in `app/services/notifications.py`. This can be left for monitoring or cleaned up later.*

The email notification system is now considered stable and working as intended. 