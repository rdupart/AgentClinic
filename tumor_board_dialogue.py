import json

# Load the original dialogues.json file
with open("dialogues.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a new list of reformatted entries
formatted_data = []
for entry in data:
    specialty_case = entry.get("role", "Unknown Topic")
    transcript = entry.get("dialogue", [])
    formatted_entry = {
        "specialty_case": specialty_case,
        "transcript": transcript
    }
    formatted_data.append(formatted_entry)

# Save the output as JSONL (one JSON object per line)
with open("formatted_dialogues.jsonl", "w", encoding="utf-8") as f_out:
    for entry in formatted_data:
        json.dump(entry, f_out)
        f_out.write("\n")

print("Conversion complete. Output saved to formatted_dialogues.jsonl")
