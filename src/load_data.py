import pandas as pd

def load_projects(filepath="data/ClimateData_2023_2024.csv"):
    """
    Load project data from CSV and apply basic cleaning.
    Update 05.08.2026: File already cleaned, so no further cleaning is applied.

    - Fills missing descriptions with titles.
    - Drops projects that are not marked for climate mitigation/adaptation.
    - Removes duplicates.
    """
    df = pd.read_csv(filepath)
    
    # Fill missing descriptions
    #df["Description"] = df["Description"].fillna(df["Title"])

    # Keep only rows marked for mitigation or adaptation
    #df = df.dropna(subset=["Climat change - mitigation", "Climat change - adaptation"])

    # Drop duplicates
    #df.drop_duplicates(inplace=True, ignore_index=True)

    return df


def load_definitions(filepath="data/Definitions.xlsx"):
    df_short = pd.read_excel(filepath, sheet_name="Definitions")
    df_detailed = pd.read_excel(filepath, sheet_name="Definition_Detail", header=None)

    cb_main = df_short[df_short["Tool"] == "CB"]["Definition"].iloc[0]
    tt_main = df_short[df_short["Tool"] == "TT"]["Definition"].iloc[0]

    def format_details(tool):
        rows = df_detailed[df_detailed.iloc[:, 0] == tool].iloc[:, 1].tolist()
        return "\n".join(f"- {r}" for r in rows)

    definitions = {
        "CB": f"{cb_main}\n\nProjects can promote:\n{format_details('CB')}",
        "TT": f"{tt_main}\n\nProjects can promote:\n{format_details('TT')}",
    }
    return definitions


def load_labeled_examples(filepath="data/Final_TT_CB_Labels.xlsx", sheet_name="Train"):
    """
    Load labeled examples from the training sheet for use in few-shot prompting.
    Removes non-relevant columns.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Drop unnecessary columns if present
    df = df.drop(columns=["Partical Action"], errors='ignore')

    return df
