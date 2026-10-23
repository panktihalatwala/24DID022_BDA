import csv
from pathlib import Path

source = Path("data/uci/SMSSpamCollection")
destination = Path("data/sms_spam.csv")

destination.parent.mkdir(parents=True, exist_ok=True)

total = 0

with source.open("r", encoding="utf-8", errors="replace") as infile, \
     destination.open("w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow(["label", "message_text"])

    for line in infile:
        line = line.rstrip("\r\n")

        if "\t" not in line:
            continue

        label, message = line.split("\t", 1)
        label = label.strip().lower()

        if label not in {"ham", "spam"}:
            continue

        writer.writerow([label, message])
        total += 1

print("Dataset conversion completed.")
print("Total SMS records:", total)
print("Output file:", destination)

