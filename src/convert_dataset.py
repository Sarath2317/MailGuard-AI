import os
import email
import pandas as pd


def extract_emails(folder, label):
    records = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "rb") as file:
                msg = email.message_from_binary_file(file)

            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)

                        if payload:
                            body += payload.decode(
                                "utf-8",
                                errors="ignore"
                            )

            else:
                payload = msg.get_payload(decode=True)

                if payload:
                    body = payload.decode(
                        "utf-8",
                        errors="ignore"
                    )

            body = body.strip()

            if body:
                records.append({
                    "label": label,
                    "message": body
                })

        except Exception as error:
            print(f"Skipped {filename}: {error}")

    return records


print("Reading Easy Ham...")
easy_ham = extract_emails(
    "../data/raw/easy_ham",
    "ham"
)

print("Reading Hard Ham...")
hard_ham = extract_emails(
    "../data/raw/hard_ham",
    "ham"
)

print("Reading Spam...")
spam = extract_emails(
    "../data/raw/spam",
    "spam"
)


# Combine all emails
records = easy_ham + hard_ham + spam

# Create DataFrame
df = pd.DataFrame(records)

# Save CSV
df.to_csv(
    "../data/spam_dataset.csv",
    index=False,
    encoding="utf-8"
)


print("\n==============================")
print("Dataset conversion complete!")
print("==============================")

print("Total emails:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nCSV saved to:")
print("../data/spam_dataset.csv")