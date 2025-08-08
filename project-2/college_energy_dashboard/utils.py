import pandas as pd

DATA_PATH = "data/college_energy_data.csv"

def get_block_list():
    return ["Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Block 6"]

def get_block_data(block_name):
    df = pd.read_csv(DATA_PATH)
    block_data = df[df["block"] == block_name]
    return block_data

def get_suggestions(data):
    suggestions = []
    if data["idle_appliances"].sum() > 2:
        suggestions.append("Switch off idle appliances when not in use.")
    if data["consumption"].mean() > 30:
        suggestions.append("Use appliances with higher energy ratings.")
    if data["consumption"].max() > 35:
        suggestions.append("Check for faulty equipment consuming excess power.")
    if not suggestions:
        suggestions.append("Energy usage is optimal. Good job!")
    return suggestions
