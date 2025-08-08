def predict_next_month_bill(data, save_energy=False):
    total_consumption = data["consumption"].sum()
    if save_energy:
        total_consumption *= 0.8  # save 20%
    return total_consumption * 7
