import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

NOTE_FILE="data/NOTEEVENTS.csv"
ADMISSION_FILE="data/ADMISSIONS.csv"
OUTLOC="data"

def load_discharges(notes_file=NOTE_FILE):
    notes = pd.read_csv(notes_file)
    discharge_data = notes[notes.CATEGORY == "Discharge summary"]
    return discharge_data

def load_readmit_times(admit_file=ADMISSION_FILE):
    admits = pd.read_csv(admit_file)
    admits.index = admits.HADM_ID
    by_patient = admits.groupby("SUBJECT_ID")
    def time_to_readmit(df):
        df = df.sort_values(by="ADMITTIME")
        return (pd.to_datetime(df.ADMITTIME).shift(-1) - pd.to_datetime(df.DISCHTIME)).apply(lambda f: f.days)
    readmit = by_patient.apply(time_to_readmit)
    readmit.index = readmit.index.droplevel("SUBJECT_ID")
    readmit.name = "DAYS_TO_READMIT"
    return readmit

def run():
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument("--notes", "-n", help="the location of the note events", default=NOTE_FILE)
    parser.add_argument("--admissions", "-a", help="the location of the admission events", default=ADMISSION_FILE)
    parser.add_argument("--out", "-o", help="the location to output the data", default=OUTLOC)
    parser.add_argument("--split", "-s", help="the split size of the test data", type=float, default=0.1)

    args = parser.parse_args()
    discharges = load_discharges(args.notes)
    readmits = load_readmit_times(args.admissions)
    data = discharges.join(readmits, on="HADM_ID")
    train, test = train_test_split(data, test_size=args.split, random_state=6181982)
    with open(f"{args.out}/train.csv", 'w') as f:
        train.to_csv(f)
    with open(f"{args.out}/test.csv", 'w') as f:
        test.to_csv(f)

if __name__ == "__main__":
    run()