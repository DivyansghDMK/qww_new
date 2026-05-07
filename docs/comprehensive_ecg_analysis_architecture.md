# Comprehensive ECG Analysis Architecture

## Overview

The "Comprehensive ECG Analysis" feature is the Holter-style workflow in CardioX.
It supports:

- starting a live multi-lead recording from the dashboard
- capturing and writing ECG packets continuously
- running background analysis on streamed chunks
- building a session summary from saved metrics
- generating a final PDF report
- viewing completed sessions from History / Replay

The feature is centered around the Holter pipeline:

`Dashboard -> ECG page -> Holter writer -> analysis worker -> report generator -> history/replay`

The most important design rule is that the system should stay ordered where clinical state matters.
Do not blindly parallelize the whole analysis pipeline:

- QRS detection and other pure signal-processing steps can be parallelized per chunk
- RR history, rhythm aggregation, AF burden, template clustering, and event fusion should remain in one ordered consumer

Think of it as a parallel map, serial reduce pattern.

Another key rule: clinical thresholds should live in configuration, not hardcoded `if` statements.
That keeps the codebase tunable for different patient profiles without rewriting Python.

## End-to-End Flow

### 1. Entry from Dashboard

The dashboard exposes the Comprehensive ECG Analysis button.
When clicked, it can:

- start a live Holter session
- open the previous-recording workspace

Main entry point:

- [`qww_new/src/dashboard/dashboard.py`](../src/dashboard/dashboard.py)

### 2. Holter Workspace / Main UI

The Holter UI is the main orchestration layer.
It owns:

- session loading
- live status updates
- tab navigation
- report generation
- replay of completed sessions

Main file:

- [`qww_new/src/ecg/holter/holter_ui.py`](../src/ecg/holter/holter_ui.py)

### 3. Live Capture

During live analysis, the ECG page pushes packets into the Holter stream writer.
That writer:

- saves packets to disk as `.ecgh`
- stores per-chunk metrics in `metrics.jsonl`
- maintains a RAM display buffer
- enqueues chunks for background analysis

Main files:

- [`qww_new/src/ecg/twelve_lead_test.py`](../src/ecg/twelve_lead_test.py)
- [`qww_new/src/ecg/holter/stream_writer.py`](../src/ecg/holter/stream_writer.py)

### 4. Background Analysis

Chunks are processed by a daemon analysis thread.
It computes:

- heart rate
- RR intervals
- PR / QRS / QT / QTc
- signal quality
- arrhythmia detection
- beat classification
- ST tendency

Important nuance:

- the worker should preserve chunk order
- rhythm state should never be recombined out of order
- the UI should not assume worker completion means clinical correctness unless the ordered reducer has accepted the chunk

Main files:

- [`qww_new/src/ecg/holter/analysis_worker.py`](../src/ecg/holter/analysis_worker.py)
- [`qww_new/src/ecg/holter/analysis_pipeline.py`](../src/ecg/holter/analysis_pipeline.py)
- [`qww_new/src/ecg/arrhythmia_detector.py`](../src/ecg/arrhythmia_detector.py)
- [`qww_new/src/ecg/ecg_calculations.py`](../src/ecg/ecg_calculations.py)
- [`qww_new/src/ecg/signal_quality.py`](../src/ecg/signal_quality.py)

### 5. Session Summary

The Holter main window builds a session summary from:

- `metrics.jsonl`
- `recording.ecgh`
- patient metadata

It uses that summary to populate:

- overview cards
- HRV tab
- Lorenz plot
- histogram
- AF analysis
- ST tendency
- report table
- waveform replay

Main file:

- [`qww_new/src/ecg/holter/holter_ui.py`](../src/ecg/holter/holter_ui.py)

### 6. Report Generation

When the user stops the recording, the report generator creates the final PDF.
It reads:

- `metrics.jsonl`
- `recording.ecgh`
- patient info
- session summary

It outputs:

- `holter_report_YYYYMMDD_HHMMSS.pdf`

Main file:

- [`qww_new/src/ecg/holter/report_generator.py`](../src/ecg/holter/report_generator.py)

### 7. History and Replay

After generation, the report is stored in the history system.
Completed sessions can later be reopened in the history UI and replayed.

Main files:

- [`qww_new/src/dashboard/history_window.py`](../src/dashboard/history_window.py)
- [`qww_new/src/dashboard/history_dialog.py`](../src/dashboard/history_dialog.py)
- [`qww_new/src/ecg/holter/replay_engine.py`](../src/ecg/holter/replay_engine.py)
- [`qww_new/src/ecg/holter/file_format.py`](../src/ecg/holter/file_format.py)

## Runtime Lifecycle

### Live Session Lifecycle

1. User clicks Comprehensive ECG Analysis in the dashboard.
2. The app switches to the ECG page.
3. Live mode starts and packets begin flowing into the Holter writer.
4. The writer saves the session to disk and feeds analysis chunks.
5. The analysis worker updates the session metrics in the background.
6. The Holter UI refreshes the live summary panels.
7. The user stops recording.
8. The session is finalized and the PDF report is generated.
9. The report is added to History.

### Previous Recording Lifecycle

1. User chooses "View Previous Recording".
2. The Holter UI loads the session folder.
3. It reads `metrics.jsonl` and `recording.ecgh`.
4. It reconstructs the summary and replay data.
5. The user can browse tabs, replay strips, and regenerate the report.

## Key Features

- live packet streaming at ECG acquisition time
- persistent recording to `.ecgh`
- incremental metrics in `metrics.jsonl`
- background chunk analysis
- arrhythmia detection
- HRV metrics
- ST tendency tracking
- replay of completed recordings
- history integration
- PDF report generation
- support for live and previous sessions

## Execution Model

### Safe Parallelism

The only safe place to parallelize aggressively is inside stateless signal processing.
Examples:

- QRS detection
- filtering
- feature extraction on a single chunk

Keep these serial:

- RR accumulation
- AF burden calculation
- beat template history
- event aggregation
- final clinical interpretation

### Fault Domains

The system should fail in isolated domains:

- capture
- analysis
- storage
- replay
- report generation

If one domain fails, the session should be preserved and the UI should tell the user exactly what failed.

### Shutdown Rules

The writer and analysis worker should flush cleanly on exit.
Daemon-only shutdown is risky for medical recording because it can cut off a session mid-write.

## Important Files

### UI and Flow

- [`qww_new/src/dashboard/dashboard.py`](../src/dashboard/dashboard.py)
- [`qww_new/src/dashboard/history_window.py`](../src/dashboard/history_window.py)
- [`qww_new/src/dashboard/history_dialog.py`](../src/dashboard/history_dialog.py)

### Live ECG Source

- [`qww_new/src/ecg/twelve_lead_test.py`](../src/ecg/twelve_lead_test.py)
- [`qww_new/src/ecg/demo_manager.py`](../src/ecg/demo_manager.py)
- [`qww_new/src/ecg/recording.py`](../src/ecg/recording.py)

### Holter / Comprehensive ECG Analysis Core

- [`qww_new/src/ecg/holter/holter_ui.py`](../src/ecg/holter/holter_ui.py)
- [`qww_new/src/ecg/holter/stream_writer.py`](../src/ecg/holter/stream_writer.py)
- [`qww_new/src/ecg/holter/analysis_worker.py`](../src/ecg/holter/analysis_worker.py)
- [`qww_new/src/ecg/holter/analysis_pipeline.py`](../src/ecg/holter/analysis_pipeline.py)
- [`qww_new/src/ecg/holter/replay_engine.py`](../src/ecg/holter/replay_engine.py)
- [`qww_new/src/ecg/holter/file_format.py`](../src/ecg/holter/file_format.py)
- [`qww_new/src/ecg/holter/report_generator.py`](../src/ecg/holter/report_generator.py)

### Supporting Shared Utilities

- [`qww_new/src/utils/pdf_process_runner.py`](../src/utils/pdf_process_runner.py)
- [`qww_new/src/utils/session_recorder.py`](../src/utils/session_recorder.py)
- [`qww_new/src/utils/settings_manager.py`](../src/utils/settings_manager.py)
- [`qww_new/src/ecg/ecg_calculations.py`](../src/ecg/ecg_calculations.py)
- [`qww_new/src/ecg/arrhythmia_detector.py`](../src/ecg/arrhythmia_detector.py)
- [`qww_new/src/ecg/signal_quality.py`](../src/ecg/signal_quality.py)

## Short Summary

Comprehensive ECG Analysis in CardioX is a full Holter pipeline:

- the dashboard opens the workflow
- the ECG page streams packets
- the Holter writer saves and buffers the session
- the analysis worker computes metrics in the background
- the UI shows live summary tabs
- the report generator creates the final PDF
- history stores the result and replay can reopen it later

## Refactor Roadmap

### Phase 1 - Stability

1. Add a clinical config layer such as `clinical_config.yaml`.
2. Split `holter_ui.py` into smaller controllers.
3. Add clear error boundaries between capture, analysis, storage, replay, and report.
4. Implement graceful worker shutdown and flush-on-exit.

### Phase 2 - Scalability

5. Replace the current flat session storage with layered storage:
   - `recording.ecgh` for waveform data
   - `events.db` for indexed events
   - `metrics_cache.bin` for fast summaries
   - `session.json` for metadata
6. Keep ordered analysis single-threaded at the aggregation layer.
7. Move to a beat-centric model so PVC burden, AF burden, and morphology clustering can evolve cleanly.
8. Build indexed replay so large recordings do not require full rescans.

### Phase 3 - Clinical Quality

9. Add event timelines and episode merging.
10. Add beat annotation editing for clinician review.
11. Track per-lead SQI and lead-off detection.
12. Add confidence scores to every detection.

### Phase 4 - Commercial Holter Grade

13. Cache report assets incrementally during recording.
14. Improve morphology template clustering.
15. Add export formats such as EDF+ or HL7 aECG if required.
16. Keep the analysis pipeline modular so PDF, export, and replay stay separate outputs from the same core session data.

## Candidate Clinical Config Keys

These are the kinds of values that belong in a config file rather than hardcoded in Python:

- VT rate threshold
- PVC burden window
- BBB QRS threshold
- AF RR irregularity threshold
- minimum SQI for acceptance
- pause duration threshold
- tachycardia / bradycardia cutoffs
- morphology cluster similarity threshold

This is the architectural lever that makes the rest of the system easier to tune safely.
