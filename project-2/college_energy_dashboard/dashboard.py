import streamlit as st
from utils import get_block_list, get_block_data, get_suggestions

def show_dashboard():
    st.sidebar.title("Blocks")
    blocks = get_block_list()
    selected_block = st.sidebar.radio("Select a block", blocks)

    data = get_block_data(selected_block)
    total_consumption = data["consumption"].sum()  # in kWh
    total_idle = data["idle_appliances"].sum()
    total_bill = total_consumption * 10  # ₹10 per kWh
    predicted_bill = total_bill * 1.05  # 5% increase if same usage continues
    optimized_bill = total_bill * 0.85  # 15% saving if improved usage

    st.title("Energy Consumption Report")
    st.subheader(f"{selected_block} - Summary")
    st.write(f"🔢 **Total Energy Consumption (kWh)**: {total_consumption:.2f}")
    st.write(f"💰 **Estimated Bill (₹)**: {total_bill:.2f}")

    if total_bill > 7000:
        st.markdown("<span style='color: red; font-weight: bold;'>⚠️ High Energy Consumption! Please save energy.</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: green;'>✅ Consumption under control.</span>", unsafe_allow_html=True)

    st.write(f"📅 **Predicted Bill Next Month (if same usage)**: ₹{predicted_bill:.2f}")
    st.write(f"💡 **With Energy Saving: ₹{optimized_bill:.2f}**")

    st.subheader("Idle Appliances Detected")
    st.write(f"🔌 **Total Idle Appliances**: {total_idle}")
    for i in range(total_idle):
        st.toggle(f"Turn Off Idle Appliance {i+1}", value=True)

    st.subheader("Suggestions")
    for suggestion in get_suggestions(data):
        st.write("•", suggestion)
