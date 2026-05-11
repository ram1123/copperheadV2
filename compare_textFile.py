import pandas as pd

# Helper to read and extract run, lumi, event from a file
def load_events(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    # Skip header and separator
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("=") and not line.startswith("run")]

    events = set()
    for line in data_lines:
        parts = line.split()
        try:
            run = int(parts[0])
            lumi = int(parts[1])
            event = int(parts[2])
            events.add((run, lumi, event))
        except (ValueError, IndexError):
            continue

    return events

# Load both files
anusree_events = load_events("AnusreeFile.txt")
sync_events = load_events("sync_output_file.txt")

# Compare sets
only_in_anusree = anusree_events - sync_events
only_in_sync = sync_events - anusree_events
common_events = anusree_events & sync_events

# Print summary
print(f"Total events in AnusreeFile.txt: {len(anusree_events)}")
print(f"Total events in sync_output_file.txt: {len(sync_events)}")
print(f"Common events: {len(common_events)}")
print(f"Events only in AnusreeFile.txt: {len(only_in_anusree)}")
print(f"Events only in sync_output_file.txt: {len(only_in_sync)}")

# Optional: print a few differences
print("\nExamples only in AnusreeFile.txt:")
for e in list(only_in_anusree)[:5]:
    print(e)

print("\nExamples only in sync_output_file.txt:")
for e in list(only_in_sync)[:5]:
    print(e)
