import pandas as pd
import argparse

NOTE_FILE="data/NOTEEVENTS.csv"
ADMISSION_FILE="data/ADMISSIONS.csv"
READMIT_FILE="data/readmit.csv"
DISCARGES="data/discharges.csv"

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
    readmit.name = "days"
    return readmit

def run():
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument("--notes", "-n", help="the location of the note events", default=NOTE_FILE)
    parser.add_argument("--admissions", "-a", help="the location of the admission events", default=ADMISSION_FILE)
    parser.add_argument("--readmits", "-r", help="the output for the label file", default=READMIT_FILE)
    parser.add_argument("--discharges", "-d", help="the output for the discharge data", default=DISCARGES)

    args = parser.parse_args()
    discharges = load_discharges(args.notes)
    with open(args.discharges, 'w') as f:
        discharges.to_csv(f)

    readmits = load_readmit_times(args.admissions)
    with open(args.readmits, 'w') as f:
        readmits.to_csv(f)

if __name__ == "__main__":
    run()